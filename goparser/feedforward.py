"""This module contains a feedforward neural network for transition- and graph-based
dependency parsing.
"""
from collections.abc import Sequence
from random import random

import torch
import torch.nn as nn

from nn_features import NNFeatureMap, NNFeatures
from nn_utils import ACTIVATION_MODULE, INIT_WEIGHT, get_fasttext_embeddings
from utils import NULL, ROOT, UNK

from graph.graph_feats import MAX_DISTANCE


class GFeedforward(nn.Module):
    """This is a feedforward neural network for graph-based dependency parsing.
    Predicts arcs and labels together.
    
    Attributes:
        feature_map: An NNFeatureMap object for feature and label mapping.
        device: The device that the model will run on.
        in_size: An int with the input dimension.
        out_size: An int with the output dimension.
    """
    
    def __init__(self,
                 feature_map: NNFeatureMap,
                 fasttext_file: str | None = None,
                 num_features: int = 10,
                 word_dim: int = 64,
                 pos_dim: int = 32,
                 dis_dim: int = 32,
                 hidden_dims: Sequence[int] = (256,),
                 hidden_dropout: float = 0.5,
                 word_dropout: float = 0.25,
                 init_weight: str = 'xavieruniform',
                 activation: str = 'tanhcube',
                 gpu: str | None = None) -> None:
        """Inits a feedforward neural network for graph-based dependency parsing.
        
        Args:
            feature_map: An NNFeatureMap object with the feature-to-index maps.
            fasttext_file: An optional string path to the pretrained fastText binary (.bin) file.
                Default: None.
            num_features: An int for the number of features (excluding distance).
                Default: 10.
            word_dim: An int for the dimension of the word embedding.
                Default: 64.
                Will be overridden to the size of the pretrained embedding if given.
            pos_dim: An int for the dimension of the POS embedding.
                Default: 32.
            dis_dim: An int for the dimension of the distance embedding.
                Default: 32.
            hidden_dims: A Sequence of ints for the dimensions of each hidden layer.
                Default: (256,).
                A hidden layer of the given size will be added for every int in the Sequence.
            hidden_dropout: A float for the dropout probability between hidden layers.
                Default: 0.5.
            word_dropout: A float for the word dropout ⍺.
                Default: 0.25.
            init_weight: An INIT_WEIGHT string for a function that initializes tensor weights.
                Default: 'xavieruniform'
            activation: An ACTIVATION_MODULE string for a non-linear activation module.
                Default: 'tanhcube'
            gpu: An optional string to specify which GPU to use.
                Default: None.
        """
        super().__init__()
        self.feature_map = feature_map
        self.word_dropout = word_dropout
        self.init_weight = INIT_WEIGHT[init_weight]
        self.activation = ACTIVATION_MODULE[activation]
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
        
        # distance embedding layer
        self.dis_embeds = nn.Embedding(feature_map.deprel_map[MAX_DISTANCE], dis_dim)
        self.init_weight(self.dis_embeds.weight)
        
        # calculate in/out size and build network layers
        self.in_size = (num_features * word_dim) + (num_features * pos_dim) + dis_dim
        self.out_size = len(feature_map.label_map)
        layers = []
        
        # add hidden layers + activation function
        h_size = self.in_size
        for i in range(len(hidden_dims)):
            h = nn.Linear(h_size, hidden_dims[i])
            self.init_weight(h.weight)
            nn.init.zeros_(h.bias)
            layers.append(h)
            layers.append(self.activation())
            if hidden_dropout > 0.0:
                layers.append(nn.Dropout(hidden_dropout))
            h_size = hidden_dims[i]
        
        # output layer
        o = nn.Linear(h_size, self.out_size)
        self.init_weight(o.weight)
        nn.init.zeros_(o.bias)
        layers.append(o)
        
        # fully-connected layers
        self.model = nn.Sequential(*layers)

    def create_input_tensor(self, features: NNFeatures) -> torch.Tensor:
        """Creates the input tensor.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor containing the input features.
        """
        x = torch.empty((1, 0), device=self.device)
        
        for w in features.words:
            if w not in self.feature_map.word_map:
                w = UNK
            elif self.training and self.word_dropout > 0.0 and w != NULL and w!= ROOT:
                p_unk = (self.word_dropout /
                         (self.word_dropout
                          + self.feature_map.wordfreq_map[self.feature_map.word_map[w]]))
                if p_unk > random():
                    w = UNK
            word_index = torch.tensor([self.feature_map.word_map[w]], device=self.device)
            x = torch.cat((x, self.word_embeds(word_index)), 1)
        
        for p in features.pos:
            if p not in self.feature_map.pos_map:
                p = UNK
            pos_index = torch.tensor([self.feature_map.pos_map[p]], device=self.device)
            x = torch.cat((x, self.pos_embeds(pos_index)), 1)
        
        if features.distance >= self.feature_map.deprel_map[MAX_DISTANCE]:
            features.distance = 0
        dis_index = torch.tensor([features.distance], device=self.device)
        x = torch.cat((x, self.dis_embeds(dis_index)), 1)
        
        return x

    def forward(self, features: NNFeatures) -> torch.Tensor:
        """Runs a forward pass.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor of the result.
        """
        return self.model(self.create_input_tensor(features))


class TFeedforward(nn.Module):
    """This is a feedforward neural network for transition-based dependency parsing.
    
    Attributes:
        feature_map: An NNFeatureMap object for feature and label mapping.
        device: The device that the model will run on.
        in_size: An int with the input dimension.
        out_size: An int with the output dimension.
    """
    
    def __init__(self,
                 feature_map: NNFeatureMap,
                 fasttext_file: str | None = None,
                 num_features: int = 18,
                 word_dim: int = 64,
                 pos_dim: int = 32,
                 deprel_dim: int = 32,
                 hidden_dims: Sequence[int] = (256,),
                 hidden_dropout: float = 0.5,
                 word_dropout: float = 0.25,
                 init_weight: str = 'kaiminguniform',
                 activation: str = 'relu',
                 gpu: str | None = None) -> None:
        """Inits a feedforward neural network for transition-based dependency parsing.
        
        Args:
            feature_map: An NNFeatureMap object with the feature-to-index maps.
            fasttext_file: An optional string path to the pretrained fastText binary (.bin) file.
                Default: None.
            num_features: An int for the number of features.
            (Arc-Standard: 18, Arc-Hybrid: 21, Arc-Eager: 22)
                Default: 18.
            word_dim: An int for the dimension of the word embedding.
                Default: 64.
                Will be overridden to the size of the pretrained embedding if given.
            pos_dim: An int for the dimension of the POS embedding.
                Default: 32.
            dis_dim: An int for the dimension of the distance embedding.
                Default: 32.
            hidden_dims: A Sequence of ints for the dimensions of each hidden layer.
                Default: (256,).
                A hidden layer of the given size will be added for every int in the Sequence.
            hidden_dropout: A float for the dropout probability between hidden layers.
                Default: 0.5.
            word_dropout: A float for the word dropout ⍺.
                Default: 0.25.
            init_weight: An INIT_WEIGHT string for a function that initializes tensor weights.
                Default: 'kaiminguniform'
            activation: An ACTIVATION_MODULE string for a non-linear activation module.
                Default: 'relu'
            gpu: An optional string to specify which GPU to use.
                Default: None.
        """
        super().__init__()
        self.feature_map = feature_map
        self.word_dropout = word_dropout
        self.init_weight = INIT_WEIGHT[init_weight]
        self.activation = ACTIVATION_MODULE[activation]
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
       
        # dependency relation embedding layer
        self.deprel_embeds = nn.Embedding(len(feature_map.deprel_map), deprel_dim)
        self.init_weight(self.deprel_embeds.weight)
        
        # calculate in/out size and build network layers
        self.in_size = ((num_features * word_dim)
                        + (num_features * pos_dim)
                        + ((num_features - 6) * deprel_dim))
        self.out_size = len(feature_map.label_map)
        layers = []
        
        # add hidden layers + activation function
        i_size = self.in_size
        for i in range(len(hidden_dims)):
            h = nn.Linear(i_size, hidden_dims[i])
            self.init_weight(h.weight)
            nn.init.zeros_(h.bias)
            layers.append(h)
            layers.append(self.activation())
            if hidden_dropout > 0.0:
                layers.append(nn.Dropout(hidden_dropout))
            i_size = hidden_dims[i]
        
        # output layer
        o = nn.Linear(i_size, self.out_size)
        self.init_weight(o.weight)
        nn.init.zeros_(o.bias)
        layers.append(o)
        
        # fully-connected layers
        self.model = nn.Sequential(*layers)

    def create_input_tensor(self, features: NNFeatures) -> torch.Tensor:
        """Creates the input tensor.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor containing the input features.
        """
        x = torch.empty((1, 0), device=self.device)

        for w in features.words:
            if w not in self.feature_map.word_map:
                w = UNK
            elif self.training and self.word_dropout > 0.0 and w != NULL and w!= ROOT:
                p_unk = (self.word_dropout /
                         (self.word_dropout
                          + self.feature_map.wordfreq_map[self.feature_map.word_map[w]]))
                if p_unk > random():
                    w = UNK
            word_index = torch.tensor([self.feature_map.word_map[w]], device=self.device)
            x = torch.cat((x, self.word_embeds(word_index)), 1)
        
        for p in features.pos:
            if p not in self.feature_map.pos_map:
                p = UNK
            pos_index = torch.tensor([self.feature_map.pos_map[p]], device=self.device)
            x = torch.cat((x, self.pos_embeds(pos_index)), 1)
        
        for r in features.deprels:
            if r not in self.feature_map.deprel_map:
                r = UNK
            deprel_index = torch.tensor([self.feature_map.deprel_map[r]], device=self.device)
            x = torch.cat((x, self.deprel_embeds(deprel_index)), 1)
        
        return x
    
    def forward(self, features: NNFeatures) -> torch.Tensor:
        """Runs a forward pass.
        
        Args:
            features: An NNFeatures object with features of the current state.
        
        Returns:
            A tensor of the result.
        """
        return self.model(self.create_input_tensor(features))