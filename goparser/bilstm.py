"""This module contains a BiLSTM neural network for transition- and graph-based
dependency parsing.
"""
from dataclasses import dataclass
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_features import NNFeatureMap, NNFeatures
from nn_utils import ACTIVATION_FUNC, INIT_WEIGHT, get_fasttext_embeddings
from utils import NULL, ROOT, UNK, Sentence


@dataclass(eq=False, slots=True)
class BiLSTMVectors:
    """This dataclass stores the forward and backward BiLSTM word vectors.
    
    Attributes:
        forward: The ouput tensor of the forward pass.
        backward: The ouput tensor of the backward pass.
        unknown: A bool to indicate if the word is unknown.
            Default: False.
    """

    forward: torch.Tensor = None
    backward: torch.Tensor = None
    unknown: bool = False

    def get_bilstm_vector(self) -> torch.Tensor:
        """Returns the concatenated forward and backward BiLSTM ouput tensors."""
        return torch.cat((self.forward, self.backward), 1)


class BiLSTM(nn.Module):
    """This is a BiLSTM neural network for transition- and graph-based dependency parsing.

    Attributes:
        feature_map: An NNFeatureMap object for feature and label mapping.
        device: The device that the model will run on.
        lstm_in_size: An int with the LSTM input dimension.
        mlp_in_size: An int with the MLP input dimension.
        out_size: An int with the output dimension.
    """
    
    def __init__(self,
                 feature_map: NNFeatureMap,
                 fasttext_file: str | None = None,
                 num_features: int = 2,
                 word_dim: int = 100,
                 pos_dim: int = 25,
                 mlp_dim: int = 100,
                 bilstm_num_layers: int = 2,
                 bilstm_dims: int = 125,
                 word_dropout: float = 0.25,
                 init_weight: str = 'xavieruniform',
                 activation: str = 'tanh',
                 gpu: str | None = None) -> None:
        """Inits a BiLSTM neural network for dependency parsing.

        Args:
            feature_map: An NNFeatureMap object with the feature-to-index maps.
            fasttext_file: An optional string path to the pretrained fastText binary (.bin) file.
                Default: None.
            num_features: An int for the number of features.
                Default: 2.
            word_dim: An int for the dimension of the word embedding.
                Default: 100.
                Will be overridden to the size of the pretrained embedding if given.
            pos_dim: An int for the dimension of the POS embedding.
                Default: 25.
            mlp_dim: An int for the dimension of the MLP.
                Default: 100.
            bilstm_num_layers: An int for the number of BiLSTM layers.
                Default: 2.
            bilstm_dims: An int for the dimensions of the BiLSTM layers.
                Default: 125.
            word_dropout: A float for the word dropout ⍺.
                Default: 0.25.
            init_weight: An INIT_WEIGHT string for a function that initializes tensor weights.
                Default: 'xavieruniform'
            activation: An ACTIVATION_FUNC string for a non-linear activation function.
                Default: 'tanh'
            gpu: An optional string to specify which GPU to use.
                Default: None.
        """
        super().__init__()
        self.feature_map = feature_map
        self.bilstm_num_layers = bilstm_num_layers
        self.bilstm_dims = bilstm_dims
        self.word_dropout = word_dropout
        self.init_weight = INIT_WEIGHT[init_weight]
        self.activation = ACTIVATION_FUNC[activation]
        gpu = 'cuda' if gpu is None else 'cuda:' + gpu
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        
        # word embedding layer
        if fasttext_file is None:
            self.word_embeds = nn.Embedding(len(feature_map.word_map), word_dim)
            self.init_weight(self.word_embeds.weight)
        else:
            self.word_embeds = nn.Embedding.from_pretrained(
                get_fasttext_embeddings(feature_map.word_map, fasttext_file),
                freeze=False
            )
            word_dim = self.word_embeds.embedding_dim
        
        # part-of-speech embedding layer
        self.pos_embeds = nn.Embedding(len(feature_map.pos_map), pos_dim)
        self.init_weight(self.pos_embeds.weight)
        
        # bilstm word vectors
        self.bilstm_vectors = None
        
        # input and output size
        self.lstm_in_size = word_dim + pos_dim
        self.mlp_in_size = (2 * bilstm_dims) * num_features
        self.out_size = len(feature_map.label_map)
        
        # add forward lstm layers
        self.forward_lstm = nn.LSTM(input_size=self.lstm_in_size,
                                    hidden_size=bilstm_dims,
                                    num_layers=bilstm_num_layers)
        
        # add backward lstm layers
        self.backward_lstm = nn.LSTM(input_size=self.lstm_in_size,
                                     hidden_size=bilstm_dims,
                                     num_layers=bilstm_num_layers)
        
        # set lstm weights using Xavier initialization with uniform distribution
        # set forget gate bias to 1.0
        # set all other biases to 0.0
        for lstm in (self.forward_lstm, self.backward_lstm):
            for layer in range(bilstm_num_layers):
                for weight in lstm._all_weights[layer]:
                    if 'weight' in weight:
                        self.init_weight(getattr(lstm, weight))
                    elif 'bias' in weight:
                        bias = getattr(lstm, weight)
                        n = bias.size(0)
                        f_start, f_end = n // 4, n // 2
                        bias.data[f_start:f_end].fill_(1.0)
                        bias.data[:f_start].fill_(0.0)
                        bias.data[f_end:].fill_(0.0)
        
        # add MLP hidden layer
        self.MLP_h = nn.Linear(self.mlp_in_size, mlp_dim)
        self.init_weight(self.MLP_h.weight)
        nn.init.zeros_(self.MLP_h.bias)
        
        # add MLP output layer
        self.MLP_o = nn.Linear(mlp_dim, self.out_size)
        self.init_weight(self.MLP_o.weight)
        nn.init.zeros_(self.MLP_o.bias)

    def init_state(self) -> None:
        """Initializes a new BiLSTM hidden state and cell state."""
        self.forward_h_n = torch.zeros((self.bilstm_num_layers, self.bilstm_dims),
                                       device=self.device)
        self.forward_c_n = torch.zeros((self.bilstm_num_layers, self.bilstm_dims),
                                       device=self.device)
        self.backward_h_n = torch.zeros((self.bilstm_num_layers, self.bilstm_dims),
                                        device=self.device)
        self.backward_c_n = torch.zeros((self.bilstm_num_layers, self.bilstm_dims),
                                        device=self.device)

    def detach_state(self) -> None:
        """Detatches the BiLSTM hidden state and cell state."""
        self.forward_h_n.detach()
        self.forward_c_n.detach()
        self.backward_h_n.detach()
        self.backward_c_n.detach()
    
    def init_bilstm_vectors(self, sentence: Sentence) -> None:
        """Initializes the BiLSTM word vectors given a Sentence."""
        self.bilstm_vectors = dict[str, BiLSTMVectors]()
        
        # forward LSTM
        for forw in sentence:
            w = forw.form
            if w not in self.feature_map.word_map:
                w = UNK
            elif self.training and self.word_dropout > 0.0 and w!= ROOT:
                p_unk = (self.word_dropout /
                         (self.word_dropout
                          + self.feature_map.wordfreq_map[self.feature_map.word_map[w]]))
                if p_unk > random():
                    w = UNK
            word_index = torch.tensor([self.feature_map.word_map[w]], device=self.device)
            
            p = forw.upos
            if p not in self.feature_map.pos_map:
                p = UNK
            pos_index = torch.tensor([self.feature_map.pos_map[p]], device=self.device)
            
            x = torch.cat((self.word_embeds(word_index), self.pos_embeds(pos_index)), 1)
            x, (self.forward_h_n,
                self.forward_c_n) = self.forward_lstm(x, (self.forward_h_n,
                                                          self.forward_c_n))
            
            vecs = BiLSTMVectors()
            vecs.forward = x
            if w == UNK:
                vecs.unknown = True
            
            self.bilstm_vectors[forw.form] = vecs
        
        # backward LSTM
        for backw in reversed(sentence):
            w = backw.form
            if self.bilstm_vectors[w].unknown == True:
                w = UNK
            word_index = torch.tensor([self.feature_map.word_map[w]], device=self.device)
            
            p = backw.upos
            if p not in self.feature_map.pos_map:
                p = UNK
            pos_index = torch.tensor([self.feature_map.pos_map[p]], device=self.device)
            
            x = torch.cat((self.word_embeds(word_index), self.pos_embeds(pos_index)), 1)
            x, (self.backward_h_n,
                self.backward_c_n) = self.backward_lstm(x, (self.backward_h_n,
                                                            self.backward_c_n))
            
            self.bilstm_vectors[backw.form].backward = x

    
    def create_input_tensor(self, features: NNFeatures) -> torch.Tensor:
        """Creates the input tensor.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor containing the input features.
        """
        x = torch.empty((1, 0), device=self.device)
        for w in features.words:
            if w == NULL:
                word_index = torch.tensor([self.feature_map.word_map[NULL]], device=self.device)
                pos_index = torch.tensor([self.feature_map.pos_map[NULL]], device=self.device)
                null_word = self.word_embeds(word_index)
                null_pos = self.pos_embeds(pos_index)
                null = torch.cat((null_word, null_pos, null_word, null_pos), 1)
                # truncate tensor if larger than input size
                if null.size(dim=1) > (2 * self.bilstm_dims):
                    null = null[:, :(2 * self.bilstm_dims)]
                # pad tensor if smaller than input size
                elif null.size(dim=1) < (2 * self.bilstm_dims):
                    pad = (0, (2 * self.bilstm_dims) - null.size(dim=1))
                    null = F.pad(null, pad)
                x = torch.cat((x, null), 1)
            else:
                x = torch.cat((x, self.bilstm_vectors[w].get_bilstm_vector()), 1)
        return x

    def forward(self, features: NNFeatures) -> torch.Tensor:
        """Runs a forward pass.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor of the result.
        """
        x = self.create_input_tensor(features)
        x = self.MLP_h(x)
        x = self.activation(x)
        x = self.MLP_o(x)
        return x


class GBiLSTM(BiLSTM):
    """This is a BiLSTM neural network for graph-based dependency parsing.
    Predicts arcs and labels separately.

    Attributes:
        feature_map: An NNFeatureMap object for feature and label mapping.
        device: The device that the model will run on.
        lstm_in_size: An int with the LSTM input dimension.
        mlp_in_size: An int with the MLP input dimension.
        out_size: An int with the output dimension.
    """
    
    def __init__(self,
                 feature_map: NNFeatureMap,
                 fasttext_file: str | None = None,
                 num_features: int = 2,
                 word_dim: int = 100,
                 pos_dim: int = 25,
                 mlp_arc_dim: int = 100,
                 mlp_lbl_dim: int = 100,
                 bilstm_num_layers: int = 2,
                 bilstm_dims: int = 125,
                 word_dropout: float = 0.25,
                 init_weight: str = 'xavieruniform',
                 activation: str = 'tanh',
                 gpu: str | None = None) -> None:
        """Inits a BiLSTM neural network for graph-based dependency parsing.

        Args:
            feature_map: An NNFeatureMap object with the feature-to-index maps.
            fasttext_file: An optional string path to the pretrained fastText binary (.bin) file.
                Default: None.
            num_features: An int for the number of features.
                Default: 2.
            word_dim: An int for the dimension of the word embedding.
                Default: 100.
                Will be overridden to the size of the pretrained embedding if given.
            pos_dim: An int for the dimension of the POS embedding.
                Default: 25.
            mlp_arc_dim: An int for the dimension of the MLP for predicting arcs.
                Default: 100.
            mlp_lbl_dim: An int for the dimension of the MLP for predicting deprel labels.
                Default: 100.
            bilstm_num_layers: An int for the number of BiLSTM layers.
                Default: 2.
            bilstm_dims: An int for the dimensions of the BiLSTM layers.
                Default: 125.
            hidden_dropout: A float for the dropout probability between hidden layers.
                Default: 0.33.
            word_dropout: A float for the word dropout ⍺.
                Default: 0.25.
            gpu: An optional string to specify which GPU to use.
                Default: None.
        """
        super().__init__(feature_map,
                         fasttext_file,
                         num_features,
                         word_dim,
                         pos_dim,
                         mlp_lbl_dim,
                         bilstm_num_layers,
                         bilstm_dims,
                         word_dropout,
                         init_weight,
                         activation,
                         gpu)
        
        # add arc MLP hidden layer
        self.MLP_arc_h = nn.Linear(self.mlp_in_size, mlp_arc_dim)
        self.init_weight(self.MLP_arc_h.weight)
        nn.init.zeros_(self.MLP_arc_h.bias)
        
        # add arc MLP output layer
        self.MLP_arc_o = nn.Linear(mlp_arc_dim, 1)
        self.init_weight(self.MLP_arc_o.weight)
        nn.init.zeros_(self.MLP_arc_o.bias)
    
    def forward(self, features: NNFeatures, labels: bool) -> torch.Tensor:
        """Runs a forward pass.
        
        Args:
            features: An NNFeatures object with features of the current state.
            labels: An bool to set label prediction mode.
        
        Returns:
            A tensor of the result.
        """
        x = self.create_input_tensor(features)
        
        # label MLP
        if labels:
            x = self.MLP_h(x)
            x = self.activation(x)
            x = self.MLP_o(x)
        
        # arc MLP
        else:
            x = self.MLP_arc_h(x)
            x = self.activation(x)
            x = self.MLP_arc_o(x)
        
        return x