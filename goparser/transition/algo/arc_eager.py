"""This module contains the Arc-Eager transition parser algorithm (Nivre 2003).
Builds a tree eagerly by adding arcs as soon as possible and 
attaches all left dependents first before right dependents.

The algorithm is equivalent to "nivreeager" in MaltParser.

By default, headless tokens are attached to the special ROOT node.
They can also be handled with the Unshift transition (Nivre and Fernández-González 2014).

Can only run projective-only parsing.

Supports Perceptron, Feedforward, and BiLSTM learning.
"""
from bilstm import BiLSTM
from feedforward import TFeedforward
from perceptron import Weights
from p_features import PFeatureMap

from transition.trans_feats import NNTFeatureExtractor
from transition.trans_utils import Arc, Configuration

from transition.algo.eager_algo import EagerAlgorithm


class ArcEager(EagerAlgorithm):
    """This class implements the Arc-Eager transition parser algorithm (Nivre 2003).
    Builds a tree eagerly by adding arcs as soon as possible and 
    attaches all left dependents first before right dependents.

    • The algorithm is equivalent to "nivreeager" in MaltParser.

    • By default, headless tokens are attached to the special ROOT node.
    They can also be handled with the unshift transition
    (Nivre and Fernández-González 2014).

    • Can only run projective-only parsing.

    • Supports Perceptron, Feedforward and BiLSTM learning.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
        unshift: An optional bool to enable the Unshift transition.
            Default: False
        root_label: An optional string to set the root label when headless tokens are
        attached to the special ROOT node.
            Default: None.
    """
    
    __slots__ = ('unshift', 'root_label')
    
    def __init__(self,
                 cross_entropy_loss: bool = True,
                 num_loss_accum: int = 50,
                 unshift: bool = False,
                 root_label: str | None = None) -> None:
        """Inits an Arc-Eager transition algorithm.
        
        Args:
            cross_entropy_loss: A bool to set the loss function to cross entropy loss.
            Otherwise, hinge loss will be used.
                Default: True.
            num_loss_accum: The number accumulated losses necessary to run a backward pass.
                Default: 50.
            unshift: An optional bool to enable the Unshift transition.
                Default: False
            root_label: An optional string to set the root label when headless tokens are
            attached to the special ROOT node.
                Default: None.
        """
        super().__init__(cross_entropy_loss, num_loss_accum)
        self.unshift = unshift
        self.root_label = root_label
    
    @staticmethod
    def _preprocessing(c: Configuration) -> None:
        """Preprocessing steps for the Configuration before parsing."""
        # shift ROOT from buffer to stack
        c.stack.append(c.buffer.popleft())
    
    @staticmethod
    def _terminal(c: Configuration) -> bool:
        """Returns True if the Configuration is in the terminal state."""
        return not c.buffer
    
    def _postprocessing(self,
                        c: Configuration,
                        features: PFeatureMap | NNTFeatureExtractor,
                        model:  Weights | TFeedforward | BiLSTM) -> None:
        """Postprocessing steps for the Configuration after parsing."""
        if self.unshift:
            while len(c.stack) > 1:
                if c.buffer:
                    if isinstance(model, Weights) and isinstance(features, PFeatureMap):
                        s = model.predict(features.get_feature_vector(c))
                        t, deprel = self._find_first_valid(s,
                                                           features.index_map,
                                                           c,
                                                           e=True)
                    else:
                        s = model(features(c))
                        t, deprel = self._find_first_valid(s[0],
                                                           model.feature_map.index_map,
                                                           c,
                                                           e=True)
                    self._do_transition(c, t, deprel)
                else:
                    if c.stack[-1] in c.has_head:
                        # reduce
                        c.stack.pop()
                    else:
                        # unshift
                        c.buffer.appendleft(c.stack.pop())
        else:
            # find unattached tokens and attach to the special ROOT node
            if self.root_label is None and c.tree[0]:
                self.root_label = next(iter(c.tree[0])).deprel
            no_head = set(range(1, len(c.input_tokens))) - c.has_head.keys()
            c.tree[0].update({Arc(0, node, self.root_label) for node in no_head})
    
    @staticmethod
    def _can_left_arc(c: Configuration) -> bool:
        """Returns True if a left arc is possible."""
        return len(c.stack) > 1 and c.stack[-1] not in c.has_head
    
    @staticmethod
    def _can_right_arc(c: Configuration) -> bool:
        """Returns True if a right arc is possible."""
        return True

    @staticmethod
    def _can_shift(c: Configuration) -> bool:
        """Returns True if a shift is possible."""
        return True
    
    @staticmethod
    def _can_reduce(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        return c.stack[-1] in c.has_head