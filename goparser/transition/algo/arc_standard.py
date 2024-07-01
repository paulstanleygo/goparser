"""This module contains Arc-Standard transition parser functions (Nivre 2004).
Builds a tree from the bottom up.

The algorithm is equivalent to "stack" series in MaltParser where attach operations occur
between the two top tokens of the stack.

Projective-only parsing is the default and is equivalent to "stackproj" in MaltParser.

Non-projective parsing is supported with the SWAP operation and the oracle can swap as soon
as possible as in "stackeager" from Nivre (2009) or postpone swapping as long as possible by
predicting maximal projective components first as in "stacklazy" from Nivre, Kuhlmann and
Hall (2009).

Supports Perceptron, Feedforward, and BiLSTM learning.
"""
from transition.trans_utils import Arc, Configuration, DependencyTree, NonProjective, \
    has_all_children, set_leftmost_children

from transition.algo.swap_algo import SwapAlgorithm


class ArcStandard(SwapAlgorithm):
    """This class implements the Arc-Standard transition parser algorithm (Nivre 2004).
    Builds a tree from the bottom up.

    • The algorithm is equivalent to "stack" series in MaltParser where attach operations
    occur between the two top tokens of the stack.

    • Projective-only parsing is the default and is equivalent to "stackproj" in MaltParser.

    • Non-projective parsing is supported with the SWAP operation and the oracle can swap
    as soon as possible as in "stackeager" from Nivre (2009) or postpone swapping as long
    as possible by predicting maximal projective components first as in "stacklazy" from
    Nivre, Kuhlmann, and Hall (2009).

    • Supports Perceptron, Feedforward and BiLSTM learning.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
        non_proj: A NonProjective enum to set the non-projective parsing setting.
            NONE, EAGER, and LAZY are valid.
                Default: NONE.
    """

    @staticmethod
    def _can_left_arc(c: Configuration) -> bool:
        """Returns True if a left arc is possible."""
        return len(c.stack) > 2

    @staticmethod
    def _can_swap(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        if len(c.stack) > 2:
            return c.stack[-2] < c.stack[-1]
        return False

    @staticmethod
    def _do_left_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a LEFT_ARC transition."""
        stack_top = c.stack[-1]
        stack_next = c.stack[-2]
        if deprel is None:
            deprel = c.input_tokens[stack_next].deprel
        arc = Arc(stack_top, stack_next, deprel)
        c.tree[stack_top].add(arc)
        temp = c.stack.pop()
        c.stack.pop()
        c.stack.append(temp)
        set_leftmost_children(c, arc)

    @staticmethod
    def _do_swap(c: Configuration) -> None:
        """Does a SWAP transition."""
        temp = c.stack.pop()
        c.buffer.appendleft(c.stack.pop())
        c.stack.append(temp)
    
    def _should_left_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a left arc."""
        if not self._can_left_arc(c):
            return False
        stack_top = c.stack[-1]
        if stack_top not in gold_tree:
            return False
        stack_next = c.stack[-2]
        deprel = c.input_tokens[stack_next].deprel
        if Arc(stack_top, stack_next, deprel) in gold_tree[stack_top]:
            return has_all_children(stack_next, c.tree, gold_tree)

    def _should_swap(self,
                     c: Configuration,
                     gold_tree: DependencyTree,
                     proj_dict: dict[int, int] | None) -> bool:
        """Returns True if the next transition should be a swap."""
        if proj_dict is None:
            return False
        if not self._can_swap(c):
            return False
        stack_next = c.stack[-2]
        stack_top = c.stack[-1]
        if proj_dict[stack_top] < proj_dict[stack_next]:
            if self.non_proj == NonProjective.EAGER:
                return True
            # check if first node in buffer is maximal projective component for lazy oracle
            if c.buffer:
                buffer_front = c.buffer[0]
                if buffer_front in gold_tree:
                    if Arc(buffer_front,
                        stack_top,
                        c.input_tokens[stack_top].deprel) in gold_tree[buffer_front]:
                        if has_all_children(stack_top, c.tree, gold_tree):
                            return False
                if stack_top in gold_tree:
                    if Arc(stack_top,
                        buffer_front,
                        c.input_tokens[buffer_front].deprel) in gold_tree[stack_top]:
                        if has_all_children(buffer_front, c.tree, gold_tree):
                            return False
            return True
        return False