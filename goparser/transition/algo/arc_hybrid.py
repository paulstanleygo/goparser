"""This module contains Arc-Hybrid transition parser functions (Kuhlmann et al. 2011).
Builds a tree from the bottom up and attaches all left dependents first before right
dependents.

Projective-only parsing is the default.

Non-projective parsing is supported with the SWAP operation adapted from Nivre (2009) and
the oracle can swap as soon as possible (Nivre 2009) or postpone swapping as long as possible
by predicting maximal projective components first (Nivre, Kuhlmann and Hall 2009).

Supports Feedforward and BiLSTM learning.
"""
from transition.trans_utils import Arc, Configuration, DependencyTree, NonProjective, \
    has_all_children, set_leftmost_children

from transition.algo.swap_algo import SwapAlgorithm


class ArcHybrid(SwapAlgorithm):
    """This class implements the Arc-Hybrid transition parser algorithm
    (Kuhlmann et al. 2011).
    Builds a tree from the bottom up and attaches all left dependents first before right
    dependents.

    • Projective-only parsing is the default.

    • Non-projective parsing is supported with the SWAP operation adapted from
    Nivre (2009) and the oracle can swap as soon as possible (Nivre 2009) or postpone
    swapping as long as possible by predicting maximal projective components first
    (Nivre, Kuhlmann and Hall 2009).

    • Supports Feedforward and BiLSTM learning.

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
        return len(c.stack) > 1 and len(c.buffer) > 0

    @staticmethod
    def _can_swap(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        if c.stack and c.buffer:
            return c.stack[-1] < c.buffer[0]
        return False

    @staticmethod
    def _do_left_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a LEFT_ARC transition."""
        buffer_front = c.buffer[0]
        stack_top = c.stack[-1]
        if deprel is None:
            deprel = c.input_tokens[stack_top].deprel
        arc = Arc(buffer_front, stack_top, deprel)
        c.tree[buffer_front].add(arc)
        c.stack.pop()
        set_leftmost_children(c, arc)

    @staticmethod
    def _do_swap(c: Configuration) -> None:
        """Does a SWAP transition."""
        temp = c.buffer.popleft()
        c.buffer.appendleft(c.stack.pop())
        c.buffer.appendleft(temp)
    
    def _should_left_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a left arc."""
        if not self._can_left_arc(c):
            return False
        buffer_front = c.buffer[0]
        if buffer_front not in gold_tree:
            return False
        stack_top = c.stack[-1]
        deprel = c.input_tokens[stack_top].deprel
        if Arc(buffer_front, stack_top, deprel) in gold_tree[buffer_front]:
            return has_all_children(stack_top, c.tree, gold_tree)
        return False

    def _should_swap(self,
                     c: Configuration,
                     gold_tree: DependencyTree,
                     proj_dict: dict[int, int] | None) -> bool:
        """Returns True if the next transition should be a swap."""
        if proj_dict is None:
            return False
        if not self._can_swap(c):
            return False
        buffer_front = c.buffer[0]
        stack_top = c.stack[-1]
        if proj_dict[buffer_front] < proj_dict[stack_top]:
            if self.non_proj == NonProjective.EAGER:
                return True
            # check if next node in buffer is maximal projective component for lazy oracle
            if len(c.buffer) > 1:
                buffer_next = c.buffer[1]
                if buffer_next in gold_tree:
                    if Arc(buffer_next,
                        buffer_front,
                        c.input_tokens[buffer_front].deprel) in gold_tree[buffer_next]:
                        if has_all_children(buffer_front, c.tree, gold_tree):
                            return False
                if buffer_front in gold_tree:
                    if Arc(buffer_front,
                        buffer_next,
                        c.input_tokens[buffer_next].deprel) in gold_tree[buffer_front]:
                        if has_all_children(buffer_next, c.tree, gold_tree):
                            return False
            return True
        return False