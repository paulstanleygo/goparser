"""This module contains abstract base classes for implementing a transition-based
algorithm."""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from bilstm import BiLSTM
from feedforward import TFeedforward
from nn_features import NNFeatures
from perceptron import MulticlassPerceptron, Weights
from p_features import PFeatureMap, PFeatures
from utils import Sentence

from transition.trans_feats import NNTFeatureExtractor
from transition.trans_utils import Configuration, DependencyTree, OracleOutput, \
    get_gold_tree


class TransitionAlgorithm(ABC):
    """This is an abstract base class for implementing a transition-based algorithm.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
    """
    
    __slots__ = ('cross_entropy_loss',
                 '_loss_accum',
                 '__num_loss_accum',
                 '_i')
    
    def __init__(self,
                 cross_entropy_loss: bool = True,
                 num_loss_accum: int = 50) -> None:
        """Inits a transition algorithm.
        
        Args:
            cross_entropy_loss: A bool to set the loss function to cross entropy loss.
            Otherwise, hinge loss will be used.
                Default: True.
            num_loss_accum: The number accumulated losses necessary to run a backward pass.
                Default: 50.
        """
        self.cross_entropy_loss = cross_entropy_loss
        self._loss_accum = list[torch.Tensor]()
        self.__num_loss_accum = num_loss_accum
        self._i = 0
    
    def add_iteration(self) -> None:
        """Add to the iteration count."""
        self._i += 1
    
    def parse(self,
              input_tokens: Sentence,
              features: PFeatureMap | NNTFeatureExtractor,
              model: Weights | TFeedforward | BiLSTM) -> DependencyTree:
        """Parse an input sentence and create a dependency tree.

        Args:
            input_tokens: A list of input tokens.
            features: A PFeatureMap object (Perceptron) or TFeatureExtractor function (NN).
            model: A Weights object or PyTorch Module for predicting transitions.
        
        Returns:
            A defaultdict dependency tree containing sets of Arc objects indexed by the head.
        """
        c = Configuration(input_tokens)

        self._preprocessing(c)
        
        while not self._terminal(c):
            # perceptron
            if isinstance(model, Weights) and isinstance(features, PFeatureMap):
                s = model.predict(features.get_feature_vector(c))
                t, deprel = self._find_first_valid(s, features.index_map, c)
            # neural network
            else:
                s = model(features(c))
                t, deprel = self._find_first_valid(s[0], model.feature_map.index_map, c)
            self._do_transition(c, t, deprel)
        
        self._postprocessing(c, features, model)
        
        return c.tree
    
    @staticmethod
    def _preprocessing(c: Configuration) -> None:
        """Preprocessing steps for the Configuration before parsing."""
        pass
    
    @staticmethod
    @abstractmethod
    def _terminal(c: Configuration) -> bool:
        """Returns True if the Configuration is in the terminal state."""
        pass

    @abstractmethod
    def _find_first_valid(self,
                          scores: np.ndarray | torch.Tensor,
                          index_map: dict[int, str],
                          c: Configuration,
                          e: bool = False) -> tuple[str, str]:
        """Returns the first valid transition.
        
        Args:
            scores: A NumPy array or Pytorch tensor of scores for each transition.
            index_map: A dict mapping feature indices to string.
            c: A Configuration object.
            e: An optional bool which disables the SHIFT transition.
                Default: False.
        
        Returns:
            A string tuple containing the top-scoring valid transition and dependency label.
        """
        pass

    @abstractmethod
    def _do_transition(self, c: Configuration, t: str, deprel: str | None = None) -> None:
        """Given a transition, perform it on the Configuration."""
        pass

    def _postprocessing(self,
                        c: Configuration,
                        features: PFeatureMap | NNTFeatureExtractor,
                        model:  Weights | TFeedforward | BiLSTM) -> None:
        """Postprocessing steps for the Configuration after parsing."""
        pass

    def oracle(self,
               input_tokens: Sentence,
               features: PFeatureMap | NNTFeatureExtractor | None = None,
               model: MulticlassPerceptron | TFeedforward | BiLSTM | None = None,
               optimizer: optim.Optimizer | None = None) -> OracleOutput:
        """Creates a list of transition sequences given an input sentence.

        1. To run online learning for a neural net model, the model (TFeedforward | BiLSTM),
        features (TFeatureExtractor), and optimizer (Optimizer) arguments must be given.

        2. To run online learning for a perceptron model, the model (Perceptron) and
        features (PFeatureMap) arguments must be given.

        3. To map the features for a perceptron, the features (PFeatureMap) argument must
        be given. This will also generate a list of PFeatures instances that the perceptron
        can be directly trained on.

        4. To generate a list of NNFeatures instances for a neural network, the features
        (TFeatureExtractor) argument must be given.

        5. If the model, features, and optimizer arguments are not given, the oracle will
        only generate a list of transition sequences.


        Args:
            input_tokens: A list of input tokens.
            features: An optional PFeatureMap object (Perceptron)
            or TFeatureExtractor function (NN).
            model: An optional Perceptron object or PyTorch Module for training.
            optimizer: An optional PyTorch optimizer.
        
        Returns:
            An OracleOutput namedtuple containing the transition sequence list,
            loss (float), number of transitions (int), and list of instances
            (NNFeatures or PFeatures).
        """
        seq = list[str]()
        sent_loss = 0.0
        num_trans = 0
        instances = list[NNFeatures | PFeatures]()
        c = Configuration(input_tokens)
        gold_tree = get_gold_tree(input_tokens)
        proj_dict = self._get_proj_dict(gold_tree)

        self._preprocessing(c)

        while not self._terminal(c):
            t = self._get_next_transition(c, gold_tree, proj_dict)
            if t is None:
                break
        
            # neural network online learning
            if features is not None and model is not None and optimizer is not None:
                sent_loss += self.__train_nn_model(features(c),
                                                   model,
                                                   optimizer,
                                                   self._loss_accum,
                                                   t)
            # perceptron online learning
            elif features is not None and model is not None:
                model.train(features.get_feature_vector(c),
                            features.label_map.get(t))
            # map features and generate instances for perceptron
            elif isinstance(features, PFeatureMap):
                instances.append(features.get_feature_vector(c))
            # generate instances for neural network
            elif features is not None:
                instances.append(features(c))
            
            self._do_transition(c, t.split()[0])
            seq.append(t)
            num_trans += 1
        
        return OracleOutput(seq, sent_loss, num_trans, instances)
    
    def _get_proj_dict(self, gold_tree: DependencyTree) -> dict[int, int] | None:
        """Returns a dictionary with the token ID as the key and the projective
        order index as the value given a list of projective order token IDs."""
        return None

    @abstractmethod
    def _get_next_transition(self,
                             c: Configuration,
                             gold_tree: DependencyTree,
                             proj_dict: dict[int, int] | None = None) -> str | None:
        """Returns a string of the next valid transition."""
        pass

    def __train_nn_model(self,
                         feats: NNFeatures,
                         model: BiLSTM | TFeedforward,
                         optimizer: optim.Optimizer,
                         loss_accum: list[torch.Tensor],
                         t: str) -> float:
        """Runs a forward pass on the model and returns the loss.
        Also runs a backward pass for cross entropy loss.
        For hinge loss, the backward pass needs to be run with the 'accum_backward' method.
        
        Args:
            feats: An NNFeatures object.
            model: A PyTorch Module for training.
            optimizer: A PyTorch optimizer.
            loss_accum: A list to accumulate loss tensors for BiLSTM.
            t: A string containing the correct transition.
        
        Returns:
            A float of the loss.
        """
        if self.cross_entropy_loss:
            optimizer.zero_grad()
            output = model(feats)
            loss = F.cross_entropy(
                output,
                torch.tensor([model.feature_map.label_map[t]], device=model.device)
            )
            loss.backward()
            optimizer.step()
            return loss.item()
        else:
            if not loss_accum:
                optimizer.zero_grad()
            output = model(feats)
            gold_trans = model.feature_map.label_map[t]
            wrong_trans = max(
                (score, i) for i, score in enumerate(output[0]) if i != gold_trans
            )[1]
            # transition hinge loss
            if output[0, gold_trans] < output[0, wrong_trans] + 1.0:
                loss = output[0, wrong_trans] - output[0, gold_trans]
                loss_accum.append(loss)
                return loss.item()
            return 0.0
    
    def accum_backward(self,
                       model: TFeedforward | BiLSTM,
                       optimizer: optim.Optimizer,
                       force: bool = False) -> None:
        """Runs a backward pass on the accumulated loss for hinge loss.
        
        Args:
            model: A PyTorch Module for predicting transitions.
            optimizer: A PyTorch optimizer.
            force: A bool to force a backward pass even if the number of accumulated
            losses is less than the value set in 'num_loss_accum'.
        """
        if self.cross_entropy_loss:
            return
        if len(self._loss_accum) > self.__num_loss_accum or (force and self._loss_accum):
            if isinstance(model, BiLSTM):
                model.detach_state()
            loss_sum = torch.tensor(0.0, device=model.device, requires_grad=True)
            for l in self._loss_accum:
                loss_sum = loss_sum + l
            loss_sum.backward()
            optimizer.step()
            self._loss_accum = list[torch.Tensor]()