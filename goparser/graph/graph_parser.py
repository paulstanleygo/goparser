"""This module contains graph-based parser implementations."""
from collections.abc import Callable
from random import shuffle
from typing import NamedTuple

import numpy as np
import torch

from bilstm import BiLSTM, GBiLSTM
from dep_parser import NNDependencyParser
from feedforward import GFeedforward
from utils import DevelopmentSet, Sentence, stopwatch, Treebank

from graph.graph_feats import NNGFeatureExtractor


def get_arc_index(h: int, m: int, n: int) -> int:
    """Returns the index for the arc-only scores tensor.

    Args:
        h: An int token ID the head.
        m: An int token ID of the modifier.
        n: An int of the sentence length.
    
    Returns:
        An int index for the scores tensor.
    """
    if m > h:
        return (h * n) + m - 1 - (h * 2)
    else:
        return (h * n) + m - (h * 2)


def get_arc_label_index(h: int, m: int, n: int, lbl: int, lbl_n: int) -> int:
    """Returns the index for the arc and label scores tensor.

    Args:
        h: An int token ID the head.
        m: An int token ID of the modifier.
        n: An int of the sentence length.
        lbl: An int of the label index.
        lbl_n: An int of the number of labels.
    
    Returns:
        An int index for the scores tensor.
    """
    h_m = (h * n) + m - (h * 2)
    if m > h:
        h_m -= 1
    return (h_m * lbl_n) + lbl


class ArcGraphScores(NamedTuple):
    """A namedtuple for storing the predicted arc scores and dependency relation labels.
    
    Fields:
        arc_scores: An n x n NumPy float array containing the predicted arc scores.
        scores_tensor: A PyTorch float tensor containing the predicted arc and label scores.
            Can be set to None when predicting.
    """

    arc_scores: np.ndarray
    scores_tensor: torch.Tensor | None


class ArcLabelGraphScores(NamedTuple):
    """A namedtuple for storing the predicted arc scores and dependency relation labels.
    
    Fields:
        arc_scores: An n x n NumPy float array containing the predicted arc scores.
        labels: A dict with the head and modifier index string as the key and
        int index of the label.
        scores_tensor: A PyTorch float tensor containing the predicted arc and label scores.
            Can be set to None when predicting.
    """

    arc_scores: np.ndarray
    labels: dict[str, int]
    scores_tensor: torch.Tensor | None


class NNSeparateGraphParser(NNDependencyParser):
    """This class implements a graph-based parser that predicts arcs and labels separately
    using a neural network.

    Attributes:
        loss_augment: A float for the loss augmented inference value.
            Default: 1.0.
    """

    __slots__ = ('loss_augment',)

    def __init__(self,
                 model: GBiLSTM,
                 graph_algo: Callable[[np.ndarray], np.ndarray],
                 feature_extractor: NNGFeatureExtractor,
                 learning_rate: float = 1e-3,
                 epochs: int = 30,
                 labeled_metric: bool = True,
                 loss_augment: float = 1.0) -> None:
        """Inits a graph-based parser with a BiLSTM neural network.
        
        Args:
            model: A PyTorch Module object containing the neural network architecture.
            graph_algo: A graph decoder function that returns a size n NumPy array for arcs
            given a size n x n NumPy array for scores.
            feature_extractor: A GFeatureExtractor function.
            learning_rate: A float for the learning rate.
                Default: 1e-2.
            epochs: An int of the number of epochs to run during training.
                Default: 30.
            labeled_metric: A bool to set the evaluation metric to labeled attachment score.
                Default: True.
            loss_augment: A float for the loss augmented inference value.
                Default: 1.0.
        """
        super().__init__(model,
                         graph_algo,
                         feature_extractor,
                         learning_rate,
                         epochs,
                         labeled_metric)
        self.loss_augment = loss_augment
    
    def __get_scores(self, sentence: Sentence) -> ArcGraphScores:
        """Returns a ArcGraphScores namedtuple given a Sentence.
        
        Args:
            sentence: A Sentence list of Tokens.
        
        Returns:
            A GraphScores namedtuple containing the arc scores.

            • Training mode will return 'arc_scores' and 'scores_tensor'.
            
            • Prediction mode will return 'arc_scores' and None for 'scores_tensor'.
        """
        n = len(sentence)
        arc_scores = np.full((n, n), -np.inf)
        scores_tensor = torch.empty((1, 0),
                                    device=self.model.device,
                                    requires_grad=True) if self.model.training else None
        for h in range(n):
            for m in range(1, n):
                if h == m:
                    continue
                output = self.model(self.feature_extractor(sentence, h, m), labels=False)
                arc_scores[h, m] = output.item()
                if self.model.training:
                    scores_tensor = torch.cat((scores_tensor, output), 1)
        return arc_scores, scores_tensor
    
    def predict(self, treebank: Treebank, state_dict: str | None = None) -> None:
        """Predicts the dependency tree given a Treebank.
        
        Args:
            treebank: A Treebank list containing a treebank.
            state_dict: An optional string path to a saved PyTorch state_dict.
        """
        if state_dict is not None:
            print('Loading state_dict: ' + state_dict)
            self.model.load_state_dict(torch.load(state_dict, self.model.device))
        with torch.no_grad():
            self.model.eval()
            for sentence in treebank:
                self.model.init_state()
                self.model.init_bilstm_vectors(sentence)

                arc_scores, _ = self.__get_scores(sentence)
                arcs = self.algo(arc_scores)
                for token in sentence[1:]:
                    token.head = arcs[token.id]
                    lbl_output = self.model(
                        self.feature_extractor(sentence, token.head, token.id),
                        labels=True
                    )
                    lbl_idx = torch.argmax(lbl_output).item()
                    token.deprel = self.model.feature_map.index_map[lbl_idx]
    
    @stopwatch
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
        optimizer = self.optimizer(self.model.parameters(), self.learning_rate)
        highest_dev_score = 0.0
        
        for e in range(self.epochs):
            print(f'Epoch {e + 1}')
            total_loss = 0.0
            num_arcs = 0
            self.model.train()
            shuffle(train_treebank)
            
            for sentence in train_treebank:
                optimizer.zero_grad()
                self.model.init_state()
                self.model.init_bilstm_vectors(sentence)
                
                arc_scores, scores_tensor = self.__get_scores(sentence)
                if self.loss_augment > 0.0:
                    arc_scores += self.loss_augment
                    for token in sentence[1:]:
                        arc_scores[token.head, token.id] -= self.loss_augment
                
                y = [t.head for t in sentence]
                ŷ = self.algo(arc_scores)
                loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
                
                for m in range(1, len(sentence)):
                    # arc hinge loss
                    if ŷ[m] != y[m]:
                        y_arc = get_arc_index(y[m], m, len(sentence))
                        ŷ_arc = get_arc_index(ŷ[m], m, len(sentence))
                        loss = loss + scores_tensor[0, ŷ_arc] - scores_tensor[0, y_arc]
                    # label hinge loss
                    lbl_output = self.model(
                        self.feature_extractor(sentence, y[m], m),
                        labels=True
                    )
                    gold_lbl = self.model.feature_map.label_map[sentence[m].deprel]
                    wrong_lbl = max(
                        (score, i) for i, score in enumerate(lbl_output[0]) if i != gold_lbl
                    )[1]
                    if lbl_output[0, gold_lbl] < lbl_output[0, wrong_lbl] + 1.0:
                        loss = loss + lbl_output[0, wrong_lbl] - lbl_output[0, gold_lbl]
                
                if loss > 0.0:
                    self.model.detach_state()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                num_arcs += len(sentence) - 1
            
            print('Training Set Loss: %.2f%%' % (total_loss / num_arcs))
            highest_dev_score = self.validate(dev_set, highest_dev_score, state_dict)
            print('*FINISH*')


class NNTogetherGraphParser(NNDependencyParser):
    """This class implements a graph-based parser that predicts arcs and labels together
    using a neural network.
    
    Attributes:
        loss_augment: A float for the loss augmented inference value.
        1.0 is recommended for BiLSTM.
            Default: 0.0.
    """

    __slots__ = ('loss_augment',)

    def __init__(self,
                 model: BiLSTM | GFeedforward,
                 graph_algo: Callable[[np.ndarray], np.ndarray],
                 feature_extractor: NNGFeatureExtractor,
                 learning_rate: float = 1e-3,
                 epochs: int = 30,
                 labeled_metric: bool = True,
                 loss_augment: float = 1.0) -> None:
        """Inits a graph-based parser that predicts arcs and labels together.
        
        Args:
            model: A PyTorch Module object containing the neural network architecture.
            graph_algo: A graph decoder function that returns a size n NumPy array for arcs
            given a size n x n NumPy array for scores.
            feature_extractor: A GFeatureExtractor function.
            learning_rate: A float for the learning rate.
                Default: 1e-3.
            epochs: An int of the number of epochs to run during training.
                Default: 30.
            labeled_metric: A bool to set the evaluation metric to labeled attachment score.
                Default: True.
            loss_augment: A float for the loss augmented inference value.
                Default: 1.0.
        """
        super().__init__(model,
                         graph_algo,
                         feature_extractor,
                         learning_rate,
                         epochs,
                         labeled_metric)
        self.loss_augment = loss_augment
    
    def __get_scores(self, sentence: Sentence) -> ArcLabelGraphScores:
        """Returns an ArcLabelGraphScores namedtuple given a Sentence.
        
        Args:
            sentence: A Sentence list of Tokens.
        Returns:
            A GraphScores namedtuple containing the arc scores.

            • Training mode will return arc_scores, scores_tensor, and None for labels.
            
            • Prediction mode will return arc_scores, labels, and None for scores_tensor.
        """
        n = len(sentence)
        arc_scores = np.full((n, n), -np.inf)
        labels = dict[str, int]()
        scores_tensor = torch.empty((1, 0),
                                    device=self.model.device,
                                    requires_grad=True) if self.model.training else None
        for h in range(n):
            for m in range(1, n):
                if h == m:
                    continue
                output = self.model(self.feature_extractor(sentence, h, m))
                argmax = torch.argmax(output).item()
                arc_scores[h, m] = output[0, argmax].item()
                labels[str(h) + str(m)] = argmax
                if self.model.training:
                    scores_tensor = torch.cat((scores_tensor, output), 1)
        return ArcLabelGraphScores(arc_scores, labels, scores_tensor)
    
    def predict(self, treebank: Treebank, state_dict: str | None = None) -> None:
        """Predicts the dependency tree given a Treebank.
        
        Args:
            treebank: A Treebank list containing a treebank.
            state_dict: An optional string path to a saved PyTorch state_dict.
        """
        if state_dict is not None:
            print('Loading state_dict: ' + state_dict)
            self.model.load_state_dict(torch.load(state_dict, self.model.device))
        with torch.no_grad():
            self.model.eval()
            for sentence in treebank:
                if isinstance(self.model, BiLSTM):
                    self.model.init_state()
                    self.model.init_bilstm_vectors(sentence)
                
                arc_scores, labels, _ = self.__get_scores(sentence)
                arcs = self.algo(arc_scores)
                for token in sentence[1:]:
                    token.head = arcs[token.id]
                    token.deprel = self.model.feature_map.index_map[
                        labels[str(token.head) + str(token.id)]
                    ]
    
    @stopwatch
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
        optimizer = self.optimizer(self.model.parameters(), self.learning_rate)
        highest_dev_score = 0.0
        
        for e in range(self.epochs):
            print(f'Epoch {e + 1}')
            total_loss = 0.0
            num_arcs = 0
            self.model.train()
            shuffle(train_treebank)
            
            for sentence in train_treebank:
                optimizer.zero_grad()
                if isinstance(self.model, BiLSTM):
                    self.model.init_state()
                    self.model.init_bilstm_vectors(sentence)
                
                arc_scores, labels, scores_tensor = self.__get_scores(sentence)
                if self.loss_augment > 0.0:
                    arc_scores += self.loss_augment
                    scores_tensor += self.loss_augment
                    for token in sentence[1:]:
                        arc_scores[token.head, token.id] -= self.loss_augment
                
                y = [t.head for t in sentence]
                ŷ = self.algo(arc_scores)
                loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
                
                for m in range(1, len(sentence)):
                    # arc and label hinge loss
                    y_lbl = self.model.feature_map.label_map[sentence[m].deprel]
                    ŷ_lbl = labels[str(ŷ[m]) + str(m)]
                    if ŷ[m] != y[m] or ŷ_lbl != y_lbl:
                        y_arc = get_arc_label_index(
                            y[m],
                            m,
                            len(sentence),
                            y_lbl,
                            len(self.model.feature_map.label_map)
                        )
                        ŷ_arc = get_arc_label_index(
                            ŷ[m],
                            m,
                            len(sentence),
                            ŷ_lbl,
                            len(self.model.feature_map.label_map)
                        )
                        loss = loss + scores_tensor[0, ŷ_arc] - scores_tensor[0, y_arc]
                
                if loss > 0.0:
                    if isinstance(self.model, BiLSTM):
                        self.model.detach_state()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                num_arcs += len(sentence) - 1
            
            print('Training Set Loss: %.2f%%' % (total_loss / num_arcs))
            highest_dev_score = self.validate(dev_set, highest_dev_score, state_dict)
            print('*FINISH*')