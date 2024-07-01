"""This module contains the Arc-Eager transition parser algorithm (Nivre 2003).
Builds a tree eagerly by adding arcs as soon as possible and 
attaches all left dependents first before right dependents.

The ROOT is placed at the end of the buffer in this variation of Arc-Eager
(Ballesteros and Nivre 2013).
This leads to better performance and guarantees a well-formed tree without
postprocessing.

Can only run projective parsing.
"""
from transition.trans_utils import Configuration

from transition.algo.eager_algo import EagerAlgorithm


class ArcEagerEndRoot(EagerAlgorithm):
    """This class implements the Arc-Eager transition parser algorithm (Nivre 2003).
    Builds a tree eagerly by adding arcs as soon as possible and 
    attaches all left dependents first before right dependents.

    • The ROOT is placed at the end of the buffer in this variation of Arc-Eager
    (Ballesteros and Nivre 2013).
    This leads to potentially better performance and guarantees a well-formed tree
    without postprocessing.

    • Can only run projective parsing.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
    """
    
    @staticmethod
    def _preprocessing(c: Configuration) -> None:
        # move ROOT to the end of the buffer
        c.buffer.append(c.buffer.popleft())
        # move first token to the stack
        c.stack.append(c.buffer.popleft())
    
    @staticmethod
    def _terminal(c: Configuration) -> bool:
        """Returns True if the Configuration is in the terminal state."""
        return not c.stack and len(c.buffer) == 1
    
    @staticmethod
    def _can_left_arc(c: Configuration) -> bool:
        """Returns True if a left arc is possible."""
        return len(c.stack) > 0 and c.stack[-1] not in c.has_head
    
    @staticmethod
    def _can_right_arc(c: Configuration) -> bool:
        """Returns True if a right arc is possible."""
        return len(c.stack) > 0 and len(c.buffer) > 1

    @staticmethod
    def _can_shift(c: Configuration) -> bool:
        """Returns True if a shift is possible."""
        return len(c.buffer) > 1
    
    @staticmethod
    def _can_reduce(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        return len(c.stack) > 0 and c.stack[-1] in c.has_head