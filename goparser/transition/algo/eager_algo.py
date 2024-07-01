"""This module contains the abstract base class containing shared methods for implementing
Arc-Eager algorithms."""
from abc import abstractmethod

import torch

from transition.algo.trans_algo import TransitionAlgorithm
from transition.trans_utils import LEFT_ARC, RIGHT_ARC, SHIFT, REDUCE, \
    Arc, Configuration, DependencyTree, \
    has_all_children, set_leftmost_children, set_rightmost_children


class EagerAlgorithm(TransitionAlgorithm):
    """This is an abstract base class containing shared methods for implementing
    Arc-Eager algorithms.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
    """
    
    def _find_first_valid(self,
                          scores: torch.Tensor,
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
        for i in sorted([(score, i) for i, score in enumerate(scores)], reverse=True):
            ŷ = index_map[i[1]].split()
            t = ŷ[0]
            deprel = None if len(ŷ) == 1 else ŷ[1]
            if t == LEFT_ARC:
                if self._can_left_arc(c):
                    return t, deprel
            elif t == RIGHT_ARC:
                if self._can_right_arc(c):
                    return t, deprel
            elif t == REDUCE:
                if self._can_reduce(c):
                    return t, deprel
            elif t == SHIFT and not e:
                if self._can_shift(c):
                    return t, deprel
            elif not e:
                raise ValueError(f'invalid transition: {t}')
    
    @staticmethod
    @abstractmethod
    def _can_left_arc(c: Configuration) -> bool:
        """Returns True if a left arc is possible."""
        pass
    
    @staticmethod
    @abstractmethod
    def _can_right_arc(c: Configuration) -> bool:
        """Returns True if a right arc is possible."""
        pass

    @staticmethod
    @abstractmethod
    def _can_shift(c: Configuration) -> bool:
        """Returns True if a shift is possible."""
        pass
    
    @staticmethod
    @abstractmethod
    def _can_reduce(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        pass
    
    def _do_transition(self, c: Configuration, t: str, deprel: str | None = None) -> None:
        """Given a transition, perform it on the Configuration."""
        if t == LEFT_ARC:
            self._do_left_arc(c, deprel)
        elif t == RIGHT_ARC:
            self._do_right_arc(c, deprel)
        elif t == REDUCE:
            c.stack.pop()
        elif t == SHIFT:
            c.stack.append(c.buffer.popleft())
        else:
            raise ValueError(f'invalid transition: {t}')
    
    @staticmethod
    def _do_left_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a LEFT_ARC transition."""
        buffer_front = c.buffer[0]
        stack_top = c.stack[-1]
        if deprel is None:
            deprel = c.input_tokens[stack_top].deprel
        arc = Arc(buffer_front, stack_top, deprel)
        c.tree[buffer_front].add(arc)
        c.has_head[stack_top] = arc
        c.stack.pop()
        set_leftmost_children(c, arc)
    
    @staticmethod
    def _do_right_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a RIGHT_ARC transition."""
        stack_top = c.stack[-1]
        buffer_front = c.buffer[0]
        if deprel is None:
            deprel = c.input_tokens[buffer_front].deprel
        arc = Arc(stack_top, buffer_front, deprel)
        c.tree[stack_top].add(arc)
        c.has_head[buffer_front] = arc
        c.stack.append(c.buffer.popleft())
        set_rightmost_children(c, arc)

    def _get_next_transition(self,
                             c: Configuration,
                             gold_tree: DependencyTree,
                             proj_dict: dict[int, int] | None = None) -> str | None:
        """Returns a string of the next valid transition."""
        if self._should_left_arc(c, gold_tree):
            t = LEFT_ARC + ' ' + c.input_tokens[c.stack[-1]].deprel
        elif self._should_right_arc(c, gold_tree):
            t = RIGHT_ARC + ' ' + c.input_tokens[c.buffer[0]].deprel
        elif self._should_reduce(c, gold_tree):
            t = REDUCE
        elif self._can_shift(c):
            t = SHIFT
        else:
            return None
        return t
    
    def _should_left_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a left arc."""
        if not self._can_left_arc(c):
            return False
        buffer_front = c.buffer[0]
        if buffer_front not in gold_tree:
            return False
        stack_top = c.stack[-1]
        deprel = c.input_tokens[stack_top].deprel
        return Arc(buffer_front, stack_top, deprel) in gold_tree[buffer_front]

    def _should_right_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a right arc."""
        if not self._can_right_arc(c):
            return False
        stack_top = c.stack[-1]
        if stack_top not in gold_tree:
            return False
        buffer_front = c.buffer[0]
        deprel = c.input_tokens[buffer_front].deprel
        return Arc(stack_top, buffer_front, deprel) in gold_tree[stack_top]

    def _should_reduce(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a swap."""
        return self._can_reduce(c) and has_all_children(c.stack[-1], c.tree, gold_tree)