"""This module contains the abstract base class containing shared methods for implementing
Arc-Standard and Arc-Hybrid algorithms with the swap transition."""
from abc import abstractmethod

import torch

from transition.algo.trans_algo import TransitionAlgorithm
from transition.trans_utils import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP, \
    Arc, Configuration, DependencyTree, NonProjective, \
    get_proj_dict, get_proj_order, has_all_children, set_rightmost_children


class SwapAlgorithm(TransitionAlgorithm):
    """This is an abstract base class containing shared methods for implementing
    Arc-Standard and Arc-Hybrid algorithms with the swap transition.

    Attributes:
        cross_entropy_loss: A bool indicating if the loss function is cross entropy loss.
            If False, hinge loss will be used.
    """
    
    __slots__ = ('non_proj',)
    
    def __init__(self,
                 cross_entropy_loss: bool = True,
                 num_loss_accum: int = 50,
                 non_proj: NonProjective = NonProjective.NONE) -> None:
        """Inits a transition algorithm with swap.
        
        Args:
            cross_entropy_loss: A bool to set the loss function to cross entropy loss.
            Otherwise, hinge loss will be used.
                Default: True.
            num_loss_accum: The number accumulated losses necessary to run a backward pass.
                Default: 50.
            non_proj: A NonProjective enum to set the non-projective parsing setting.
            NONE, EAGER, and LAZY are valid.
                Default: NONE.
        """
        super().__init__(cross_entropy_loss, num_loss_accum)
        self.non_proj = non_proj

    @staticmethod
    def _preprocessing(c: Configuration) -> None:
        """Preprocessing steps for the Configuration before parsing."""
        # shift ROOT from buffer to stack
        c.stack.append(c.buffer.popleft())
        # shift next token to the stack
        c.stack.append(c.buffer.popleft())
    
    @staticmethod
    def _terminal(c: Configuration) -> bool:
        """Returns True if the Configuration is in the terminal state."""
        return len(c.stack) == 1 and not c.buffer
    
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
            elif t == SWAP:
                if self._can_swap(c):
                    return t, deprel
            elif t == SHIFT:
                if self._can_shift(c):
                    return t, deprel
            else:
                raise ValueError(f'invalid transition: {t}')
    
    @staticmethod
    @abstractmethod
    def _can_left_arc(c: Configuration) -> bool:
        """Returns True if a left arc is possible."""
        pass
    
    @staticmethod
    def _can_right_arc(c: Configuration) -> bool:
        """Returns True if a right arc is possible."""
        return len(c.stack) > 1

    @staticmethod
    def _can_shift(c: Configuration) -> bool:
        """Returns True if a shift is possible."""
        return len(c.buffer) > 0
    
    @staticmethod
    @abstractmethod
    def _can_swap(c: Configuration) -> bool:
        """Returns True if a swap is possible."""
        pass
    
    def _do_transition(self, c: Configuration, t: str, deprel: str | None = None) -> None:
        """Given a transition, perform it on the Configuration."""
        if t == LEFT_ARC:
            self._do_left_arc(c, deprel)
        elif t == RIGHT_ARC:
            self._do_right_arc(c, deprel)
        elif t == SWAP:
            self._do_swap(c)
        elif t == SHIFT:
            c.stack.append(c.buffer.popleft())
        else:
            raise ValueError(f'invalid transition: {t}')
    
    @staticmethod
    @abstractmethod
    def _do_left_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a LEFT_ARC transition."""
        pass
    
    @staticmethod
    def _do_right_arc(c: Configuration, deprel: str | None = None) -> None:
        """Does a RIGHT_ARC transition."""
        stack_next = c.stack[-2]
        stack_top = c.stack[-1]
        if deprel is None:
            deprel = c.input_tokens[stack_top].deprel
        arc = Arc(stack_next, stack_top, deprel)
        c.tree[stack_next].add(arc)
        c.stack.pop()
        set_rightmost_children(c, arc)
    
    @staticmethod
    @abstractmethod
    def _do_swap(c: Configuration) -> None:
        """Does a SWAP transition."""
        pass
    
    def _get_proj_dict(self, gold_tree: DependencyTree) -> dict[int, int] | None:
        """Returns a dictionary with the token ID as the key and the projective
        order index as the value given a list of projective order token IDs."""
        proj_dict = None
        if self.non_proj != NonProjective.NONE:
            proj_list = get_proj_order(gold_tree)
            if proj_list != sorted(proj_list):
                proj_dict = get_proj_dict(proj_list)
        return proj_dict

    def _get_next_transition(self,
                             c: Configuration,
                             gold_tree: DependencyTree,
                             proj_dict: dict[int, int] | None = None) -> str | None:
        """Returns a string of the next valid transition."""
        if self._should_left_arc(c, gold_tree):
            t = LEFT_ARC + ' ' + c.input_tokens[c.stack[-2]].deprel
        elif self._should_right_arc(c, gold_tree):
            t = RIGHT_ARC + ' ' + c.input_tokens[c.stack[-1]].deprel
        elif self._should_swap(c, gold_tree, proj_dict):
            t = SWAP
        elif self._can_shift(c):
            t = SHIFT
        else:
            return None
        return t
    
    @abstractmethod
    def _should_left_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a left arc."""
        pass

    def _should_right_arc(self, c: Configuration, gold_tree: DependencyTree) -> bool:
        """Returns True if the next transition should be a right arc."""
        if not self._can_right_arc(c):
            return False
        stack_next = c.stack[-2]
        if stack_next not in gold_tree:
            return False
        stack_top = c.stack[-1]
        deprel = c.input_tokens[stack_top].deprel
        if Arc(stack_next, stack_top, deprel) in gold_tree[stack_next]:
            return has_all_children(stack_top, c.tree, gold_tree)
        return False

    @abstractmethod
    def _should_swap(self,
                     c: Configuration,
                     gold_tree: DependencyTree,
                     proj_dict: dict[int, int] | None) -> bool:
        """Returns True if the next transition should be a swap."""
        pass