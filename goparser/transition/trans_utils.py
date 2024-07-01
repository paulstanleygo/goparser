"""This module contains utility classes, functions, constants, and type aliases
for transition-based dependency parsing."""
from collections import defaultdict, deque
from enum import auto, Enum
from typing import NamedTuple

from nn_features import NNFeatures
from p_features import PFeatures
from utils import Sentence


LEFT_ARC = 'LA'
"""String constant representing the 'left-arc' transition."""
RIGHT_ARC = 'RA'
"""String constant representing the 'right-arc' transition."""
SHIFT = 'SH'
"""String constant representing the 'shift' transition."""
REDUCE_SWAP = SWAP_REDUCE = REDUCE = SWAP = 'RS'
"""String constant representing the 'reduce' or 'swap' transition."""


class OracleOutput(NamedTuple):
    """A namedtuple for storing the return output of an oracle.
    
    Fields:
        sequence: A list of strings containing the transition sequence.
        sentence_loss: A float containing the loss when training a model.
        num_transitions: An int containing the number of transitions in a sentence.
        instances: A list of NNFeatures or PFeatures instances.
    """

    sequence: list[str]
    sentence_loss: float
    num_transitions: int
    instances: list[NNFeatures | PFeatures]


class NonProjective(Enum):
    """Enum for non-projective options."""

    NONE = auto()
    """Enum constant for projective only parsing."""
    EAGER = auto()
    """Enum constant for non-projective eager oracle."""
    LAZY = auto()
    """Enum constant for non-projective lazy oracle."""


class Arc(NamedTuple):
    """This namedtuple stores a dependency arc.

    Fields:
        head: An integer containing the ID number of the head.
        dependent: An integer containing the ID number of the dependent.
        deprel: An optional string containing the dependency relation of the arc.
    """

    head: int
    dependent: int
    deprel: str | None = None


ArcSet = set[Arc]
"""Type alias ArcSet for a set containing Arc objects."""


DependencyTree = defaultdict[int, ArcSet]
"""Type alias DependencyTree for an adjacency list implementation of a directed graph
for a dependency tree constructed from a defaultdict with the token ID of the head as
the index key and a set of Arc objects as the value."""


class Configuration:
    """This class stores the transition state configuration for transition parsing.

    Attributes:
        input_tokens: A Sentence container of input tokens.
        stack: A list for stack σ containing int token IDs.
        queue: A deque for buffer β containing int token IDs.
        tree: A DependencyTree containing sets of Arc objects indexed by the head.
        has_head: A dict containing an Arc of dependent's head indexed by the dependent.
        leftmost_child: A dict containing the Arc of leftmost child indexed by the head.
        rightmost_child: A dict containing the Arc of rightmost child indexed by the head.
        second_leftmost_child: A dict containing the Arc of second leftmost child
        indexed by the head.
        second_rightmost_child: A dict containing the Arc of second rightmost child
        indexed by the head.
    """
    
    __slots__ = ('input_tokens',
                 'stack',
                 'buffer',
                 'tree',
                 'has_head',
                 'leftmost_child',
                 'rightmost_child',
                 'second_leftmost_child',
                 'second_rightmost_child')

    def __init__(self, input_tokens: Sentence) -> None:
        """Inits a Configuration given a Sentence container of input tokens."""
        self.input_tokens = input_tokens
        self.stack = list[int]()
        self.buffer = deque[int](range(len(input_tokens)))
        self.tree = DependencyTree(ArcSet)
        self.has_head = dict[int, Arc]()
        self.leftmost_child = dict[int, Arc]()
        self.rightmost_child = dict[int, Arc]()
        self.second_leftmost_child = dict[int, Arc]()
        self.second_rightmost_child = dict[int, Arc]()


def get_gold_tree(sentence: Sentence) -> DependencyTree:
    """Returns a DependencyTree defaultdict with the head as the key
    and the value as a set of Arc objects with the dependency arc and label.
    
    Args:
        sentence: A Sentence list of Tokens.
    
    Returns:
        A DependencyTree containing sets of Arc objects indexed by the head.
    """
    gold_tree = DependencyTree(ArcSet)
    for token in sentence:
        gold_tree[token.head].add(Arc(token.head, token.id, token.deprel))
    return gold_tree


def set_arcs(sentence: Sentence, tree: DependencyTree) -> None:
    """Sets the head and relation of the tokens in a sentence.
    
    Args:
        sentence: A Sentence list of Tokens.
        tree: A DependencyTree containing sets of Arc objects indexed by the head.
    """
    for arcs in tree.values():
        for arc in arcs:
            sentence[arc.dependent].head = arc.head
            sentence[arc.dependent].deprel = arc.deprel


def has_all_children(head: int, tree: DependencyTree, gold_tree: DependencyTree) -> bool:
    """Returns True if a head token has all its children.
    
    Args:
        head: The token ID int of the head.
        tree: A DependencyTree containing sets of Arc objects indexed by the head.
        gold_tree: A gold DependencyTree containing sets of Arc objects indexed by the head.
    
    Returns:
        A boolean.
    """
    # has no children
    if head not in gold_tree:
        return True
    return tree[head] == gold_tree[head]


def set_leftmost_children(c: Configuration, arc: Arc) -> None:
    """Sets the leftmost children of a head given a Configuration and Arc.

    Args:
        c: A Configuration object.
        arc: An Arc namedtuple.
    """
    if arc.head in c.leftmost_child:
        if arc.dependent < c.leftmost_child[arc.head].dependent:
            c.second_leftmost_child[arc.head] = c.leftmost_child[arc.head]
            c.leftmost_child[arc.head] = arc
        elif arc.head in c.second_leftmost_child:
            if arc.dependent < c.second_leftmost_child[arc.head].dependent:
                c.second_leftmost_child[arc.head] = arc
        else:
            c.second_leftmost_child[arc.head] = arc
    else:
        c.leftmost_child[arc.head] = arc


def set_rightmost_children(c: Configuration, arc: Arc) -> None:
    """Sets the rightmost children of a head given a Configuration and Arc.

    Args:
        c: A Configuration object.
        arc: An Arc namedtuple.
    """
    if arc.head in c.rightmost_child:
        if arc.dependent > c.rightmost_child[arc.head].dependent:
            c.second_rightmost_child[arc.head] = c.rightmost_child[arc.head]
            c.rightmost_child[arc.head] = arc
        elif arc.head in c.second_rightmost_child:
            if arc.dependent > c.second_rightmost_child[arc.head].dependent:
                c.second_rightmost_child[arc.head] = arc
        else:
            c.second_rightmost_child[arc.head] = arc
    else:
        c.rightmost_child[arc.head] = arc


def get_proj_order(gold_tree: DependencyTree) -> list[int]:
    """Returns a list of token IDs sorted in the projective order of the sentence given
    a gold DependencyTree containing sets of Arc objects indexed by the head.
    
    Args:
        gold_tree: A gold DependencyTree containing sets of Arc objects indexed by the head.
    
    Returns:
        A list of token IDs sorted into projective order.
    """
    proj_list = list[int]()
    _proj_order(gold_tree, proj_list, set[int](), 0)
    return proj_list


def _proj_order(gold_tree: DependencyTree,
                proj_list: list[int],
                proj_set: set[int],
                head: int) -> None:
    """Performs a recursive inorder search of a DependencyTree containing sets of 
    Arc objects indexed by the head and finds the projective order of the sentence.
    
    Args:
        gold_tree: A gold DependencyTree containing sets of Arc objects indexed by the head.
        proj_list: A list storing token IDs in projective order.
        proj_set: A set to keep track of the token IDs added so far.
        head: The token ID of the head.
    """
    if head in gold_tree:
        arcs = sorted(gold_tree[head])
        for arc in arcs:
            if arc.dependent > head:
                if head not in proj_set:
                    proj_list.append(head)
                    proj_set.add(head)
            _proj_order(gold_tree, proj_list, proj_set, arc.dependent)
    if head not in proj_set:
        proj_list.append(head)
        proj_set.add(head)


def get_proj_dict(proj_list: list[int]) -> dict[int, int]:
    """Returns a dictionary with the token ID as the key and the projective
    order index as the value given a list of projective order token IDs.
    
    Args:
        proj_list: A list of token IDs sorted into projective order.
    
    Returns:
        A dictionary with token ID as key and the projective order index as value.
    """
    return {tok: i for i, tok in enumerate(proj_list)}