from random import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Sentence


INIT_WEIGHT: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'kaimingnormal': lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'),
    'kaiminguniform': lambda x: nn.init.kaiming_uniform_(x, nonlinearity='relu'),
    'xaviernormal': nn.init.xavier_normal_,
    'xavieruniform': nn.init.xavier_uniform_
}


def tanh_cube(input: torch.Tensor) -> torch.Tensor:
    """Applies the tanh-cube function g(l) = tanh(l^3 + l) element-wise
    on a PyTorch tensor."""
    return F.tanh(input.pow(3).add(input))


class TanhCube(nn.Module):
    """Applies the tanh-cube function g(l) = tanh(l^3 + l) element-wise.
    This class extends PyTorch's Module."""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return tanh_cube(input)


ACTIVATION_FUNC: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'relu': F.relu,
    'tanh': F.tanh,
    'tanhcube': tanh_cube
}


ACTIVATION_MODULE: dict[str, nn.Module] = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'tanhcube': TanhCube
}


def get_fasttext_embeddings(word_map: dict[str, int], fasttext_file: str) -> torch.Tensor:
    """Returns a pretrained fastText word embedding tensor.
    
    Args:
        word_map: A dictionary mapping a word to index.
        fasttext_file: A string path to the pretrained fastText binary (.bin) file.
    
    Returns:
        A PyTorch tensor of pretrained word embeddings.
    """
    # import fasttext only when called
    from fasttext import FastText
    
    print(f'Creating pretrained fastText word embedding from \'{fasttext_file}\'')
    ft = FastText.load_model(fasttext_file)
    words = {v: k for k, v in word_map.items()}
    t = torch.empty(0, ft.get_dimension())
    for i in range(len(words)):
        v = torch.from_numpy(ft.get_word_vector(words[i]))
        t = torch.cat((t, v[None, :]), 0)
    print('*FINISH*')
    return t


class DeepContextualWordEmbedding:
    """This class implements deep contextual word embeddings that supports any pretrained
    BERT or compatible BERT-like transformer model from Hugging Face.
    
    Attributes:
        config: A transformers configuration class.
        tokenizer: A transformers tokenizer class.
        model: A transformers model class.
        vectors: A dictionary with a word string key and the embedding tensor as the value.
        sum_from: An int for the range of layers to sum from.
            Default: 4.
        sum_to: An int for the range of layers to sum to (inclusive).
            Default: 8.
        dropout: A float for the embedding dropout.
            Default: 0.33.
    """

    __slots__ = ('config',
                 'tokenizer',
                 'model',
                 'vectors',
                 'sum_from',
                 'sum_to',
                 'dropout')
    
    def __init__(self,
                 model_name: str = 'bert-base-multilingual-cased',
                 cache_dir: str | None = None,
                 sum_from: int = 4,
                 sum_to: int = 8,
                 dropout: float = 0.33) -> None:
        """Initializes a new DeepContextualWordEmbedding object.
        
        Args:
            model_name: A string, the model id of a pretrained model hosted inside a model repo
            on huggingface.co. Valid model ids can be located at the root-level,
            like 'bert-base-uncased', or namespaced under a user or organization name,
            like 'dbmdz/bert-base-german-cased'.
                Default: 'bert-base-multilingual-cased'.
            cache_dir: Path to a directory in which a downloaded pretrained model configuration
            should be cached if the standard cache should not be used.
                Default: None.
            sum_from: An int for the range of layers to sum from.
                Default: 4.
            sum_to: An int for the range of layers to sum to (inclusive).
                Default: 8.
            dropout: A float for the embedding dropout.
                Default: 0.33.
        """
        # import transformers only when initialized
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        
        if cache_dir:
            self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name,
                                                   cache_dir=cache_dir,
                                                   output_hidden_states=True)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name,
                                                   output_hidden_states=True)
        self.model.eval()
        self.vectors = None
        self.sum_from = sum_from
        self.sum_to = sum_to
        self.dropout = dropout
    
    def init_embedding(self, sentence: Sentence) -> None:
        """Initializes an embedding for a new Sentence."""
        # Reference: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        self.vectors = dict[str, torch.Tensor]()
        text = '[CLS] ' + ' '.join([t.form for t in sentence[1:]]) + ' [SEP]'
        toks = self.tokenizer.tokenize(text)
        idx = self.tokenizer.convert_tokens_to_ids(toks)
        seg = [1] * len(toks)
        toks_tensor = torch.tensor([idx])
        seg_tensor = torch.tensor([seg])
        
        with torch.no_grad():
            output = self.model(toks_tensor, seg_tensor)
            tok_embeds = torch.stack(output[2], 0)
            tok_embeds = torch.squeeze(tok_embeds, 1)
            tok_embeds = tok_embeds.permute(1, 0, 2)
            subwords = ''
            incomplete = False
            vec = None
            for i in range(len(toks) - 2, 0, -1):
                tok = toks[i]
                if len(tok) > 2 and tok[:2] == '##':
                    subwords = tok[2:] + subwords
                    if vec is None:
                        vec = tok_embeds[i]
                    else:
                        vec = torch.add(vec, tok_embeds[i])
                    incomplete = True
                elif incomplete:
                    subwords = tok + subwords
                    self.vectors[subwords] = torch.sum(
                        vec[self.sum_from:(self.sum_to + 1)], 0
                    )
                    subwords = ''
                    incomplete = False
                    vec = None
                else:
                    self.vectors[tok] = torch.sum(
                        tok_embeds[i, self.sum_from:(self.sum_to + 1)], 0
                    )
    
    def get_embedding(self, word: str, training: bool) -> torch.Tensor:
        """Returns the embedding for a word.
        Args:
            word: A string containing the word.
            training: A bool indicating if model is in training mode.
        """
        if word in self.vectors and (not training or random() > self.dropout):
            return self.vectors[word].unsqueeze(dim=0)
        else:
            return torch.zeros((1, self.config.hidden_size))
    
    def get_hidden_size(self) -> int:
        """Returns an int of the dimensionality of the hidden layers."""
        return self.config.hidden_size


class CanineCharacterEmbedding:
    """This class implements deep contextual character embeddings that supports any pretrained
    CANINE transformer model from Hugging Face.
    
    Attributes:
        config: A CANINE transformers configuration class.
        model: A CANINE transformers model class.
        dropout: A float for the embedding dropout.
            Default: 0.33.
    """

    __slots__ = ('config',
                 'tokenizer',
                 'model',
                 'dropout')
    
    def __init__(self,
                 model_name: str = 'google/canine-c',
                 cache_dir: str | None = None,
                 dropout: float = 0.33) -> None:
        """Initializes a new DeepContextualWordEmbedding object.
        
        Args:
            model_name: A string, the model id of a pretrained model hosted inside a model repo
            on huggingface.co. Valid model ids are 'google/canine-c' and 'google/canine-s'.
                Default: 'google/canine-c'.
            cache_dir: Path to a directory in which a downloaded pretrained model configuration
            should be cached if the standard cache should not be used.
                Default: None.
            dropout: A float for the embedding dropout.
                Default: 0.33.
        """
        # import transformers only when initialized
        from transformers import CanineConfig, CanineModel
        
        if cache_dir:
            self.config = CanineConfig.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = CanineModel.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            self.config = CanineConfig.from_pretrained(model_name)
            self.model = CanineModel.from_pretrained(model_name)
        self.model.eval()
        self.dropout = dropout
    
    def get_embedding(self, word: str, training: bool) -> torch.Tensor:
        """Returns the character embedding for a word.
        Args:
            word: A string containing the word.
            training: A bool indicating if model is in training mode.
        """
        if not training or random() > self.dropout:
            input_ids = torch.tensor([[ord(char) for char in word]])
            with torch.no_grad():
                output = self.model(input_ids)
                return torch.sum(output.last_hidden_state, 1)
        else:
            return torch.zeros((1, self.config.hidden_size))
    
    def get_hidden_size(self) -> int:
        """Returns an int of the dimensionality of the hidden layers."""
        return self.config.hidden_size