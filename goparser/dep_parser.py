"""This module contains abstract base classes for implementing a dependency parser."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.optim as optim

from bilstm import BiLSTM, GBiLSTM
from feedforward import GFeedforward, TFeedforward
from perceptron import MulticlassPerceptron
from p_features import PFeatureMap
from utils import DevelopmentSet, Treebank, get_attachment_scores

from graph.graph_feats import NNGFeatureExtractor

from transition.trans_feats import NNTFeatureExtractor

from transition.algo.trans_algo import TransitionAlgorithm


class NNDependencyParser(ABC):
    """This is an abstract base class for implementing a dependency parser for a neural network.
    
    Attributes:
        model: A PyTorch Module object containing the neural network architecture.
        algo: A graph decoder function that returns a size n NumPy array for arcs
        given a size n x n NumPy array for scores or a TransitionAlgorithm object
        containing the transition algorithm.
        feature_extractor: A GFeatureExtractor or TFeatureExtractor function.
        learning_rate: A float for the learning rate.
        epochs: An int of the number of epochs to run during training.
            Default: 30.
        labeled_metric: A bool to set the evaluation metric to labeled attachment score.
            Default: True.
        optimizer: A PyTorch optimizer.
            Default: Adam.
    """
    
    __slots__ = ('model',
                 'algo',
                 'feature_extractor',
                 'learning_rate',
                 'epochs',
                 'labeled_metric',
                 'optimizer')

    def __init__(self,
                 model: BiLSTM | GBiLSTM | GFeedforward | TFeedforward,
                 algo: Callable[[np.ndarray], np.ndarray] | TransitionAlgorithm,
                 feature_extractor: NNGFeatureExtractor | NNTFeatureExtractor,
                 learning_rate: float,
                 epochs: int = 30,
                 labeled_metric: bool = True) -> None:
        """Inits a dependency parser for a neural network.
        
        Args:
            model: A PyTorch Module object containing the neural network architecture.
            algo: A graph decoder function that returns a size n NumPy array for arcs
            given a size n x n NumPy array for scores or a TransitionAlgorithm object
            containing the transition algorithm.
            feature_extractor: A feature extractor function.
            learning_rate: A float for the learning rate in the neural network optimizer.
            epochs: An int of the number of epochs to run during training.
                Default: 30.
            labeled_metric: A bool to set the evaluation metric to labeled attachment score.
                Default: True.
        """
        self.model = model.to(model.device)
        self.algo = algo
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.labeled_metric = labeled_metric
        self.optimizer = optim.Adam
    
    @abstractmethod
    def predict(self, treebank: Treebank, state_dict: str | None = None) -> None:
        """Predicts the dependency tree given a Treebank.
        
        Args:
            treebank: A Treebank list containing a treebank.
            state_dict: An optional string path to a saved PyTorch state_dict.
        """
        pass
    
    @abstractmethod
    def train(self,
              train_treebank: Treebank,
              dev_set: DevelopmentSet,
              state_dict: str) -> None:
        """Trains a model on a treebank.
        
        Args:
            train_treebank: A Treebank list containing the training treebank.
            dev_set: A DevelopmentSet tuple containing the development sets.
            state_dict: A string path to save the state_dict or load a preexisting one.
        """
        pass

    def validate(self,
                 dev_set: DevelopmentSet,
                 highest_dev_score: float,
                 state_dict: str) -> float:
        """Validates the model on a development set and saves the best scoring model.
        
        Args:
            dev_set: A DevelopmentSet tuple containing the development sets.
            highest_dev_score: A float containing the highest development score so far.
            state_dict: A string path to save the state_dict.
        
        Returns:
            A float containing the highest development score so far.
        """
        dev_pred = deepcopy(dev_set.blind)
        self.predict(dev_pred)
        print('Development Set:')
        dev_scores = get_attachment_scores(dev_pred, dev_set.gold)
        dev_score = dev_scores.las if self.labeled_metric else dev_scores.uas
        # only save weights with highest dev score
        if dev_score > highest_dev_score:
            print(f'Saving state_dict to \'{state_dict}\'')
            torch.save(self.model.state_dict(), state_dict)
            highest_dev_score = dev_score
        return highest_dev_score


class PDependencyParser(ABC):
    """This is an abstract base class for implementing a dependency parser for a perceptron.
    
    Attributes:
        model: A MultiClassPerceptron object model.
        algo: A graph decoder function that returns a size n NumPy array for arcs
        given a size n x n NumPy array for scores or a TransitionAlgorithm object
        containing the transition algorithm.
        feature_map: A PFeatureMap object for extracting and mapping features.
        epochs: An int of the number of epochs to run during training.
            Default: 30.
        labeled_metric: A bool to set the evaluation metric to labeled attachment score.
            Default: True.
    """
    
    __slots__ = ('model',
                 'trans',
                 'feature_map',
                 'epochs',
                 'labeled_metric')
    
    def __init__(self,
                 model: MulticlassPerceptron,
                 algo: Callable[[np.ndarray], np.ndarray] | TransitionAlgorithm,
                 feature_map: PFeatureMap,
                 epochs: int = 30,
                 labeled_metric: bool = True) -> None:
        """Inits a transition-based parser.
        
        Args:
            model: A MultiClassPerceptron object model.
            algo: A graph decoder function that returns a size n NumPy array for arcs
            given a size n x n NumPy array for scores or a TransitionAlgorithm object
            containing the transition algorithm.
            feature_map: A PFeatureMap object for extracting and mapping features.
            epochs: An int of the number of epochs to run during training.
                Default: 30.
            labeled_metric: A bool to set the evaluation metric to labeled attachment score.
                Default: True.
        """
        self.model = model
        self.algo = algo
        self.feature_map = feature_map
        self.epochs = epochs
        self.labeled_metric = labeled_metric
    
    @abstractmethod
    def predict(self, treebank: Treebank, weights_file: str | None = None) -> None:
        """Predicts the dependency tree given a Treebank.
        
        Args:
            treebank: A Treebank list containing a treebank.
            weights_file: An optional string path to a saved weights file.
        """
        pass
    
    @abstractmethod
    def train(self,
              train_treebank: Treebank,
              dev_set: DevelopmentSet,
              weights_file: str) -> None:
        """Trains a model on a treebank.
        
        Args:
            train_treebank: A Treebank list containing the training treebank.
            dev_set: A DevelopmentSet tuple containing the development sets.
            weights_file: A string path to save the weights file.
        """
        pass

    def validate(self,
                 dev_set: DevelopmentSet,
                 highest_dev_score: float,
                 weights_file: str) -> float:
        """Validates the model on a development set and saves the best scoring model.
        
        Args:
            dev_set: A DevelopmentSet tuple containing the development sets.
            highest_dev_score: A float containing the highest development score so far.
            weights_file: A string path to save the weights file.
        
        Returns:
            A float containing the highest development score so far.
        """
        weights = self.model.get_weights()
        dev_pred = deepcopy(dev_set.blind)
        self.predict(dev_pred)
        print('Development Set:')
        dev_scores = get_attachment_scores(dev_pred, dev_set.gold)
        dev_score = dev_scores.las if self.labeled_metric else dev_scores.uas
        # only save weights with highest dev score
        if dev_score > highest_dev_score:
            print(f'Saving weights to \'{weights_file}\'')
            weights.export_weights(weights_file)
            highest_dev_score = dev_score
        return highest_dev_score