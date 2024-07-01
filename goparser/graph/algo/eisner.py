'''This module contains Eisner's algorithm for decoding the maximum spanning tree of
a graph.
'''
import numpy as np


OPEN = 0
"""Index constant 0 for open structure."""
CLOSE = 1
"""Index constant 1 for close structure."""
RIGHT = 0
"""Index constant 0 for right head."""
LEFT = 1
"""Index constant 1 for left head."""


def eisner(scores: np.ndarray) -> np.ndarray:
    """Finds the maximum spanning tree of a graph with Eisner's algorithm.
    
    Args:
        scores: An n x n NumPy float or int array containing scores for every possible arc.
        The row index should be the head ID and the column index should be the dependent ID.
    
    Returns:
        A size n NumPy int array indexed by the dependent ID containing the ID of its head.
    """
    m, n = np.shape(scores)
    if m != n:
        raise ValueError('\'scores\' must be an n x n array')
    
    # index: row, column, structure, head
    table = np.zeros((n, n, 2, 2), dtype=scores.dtype)
    backpointers = np.full((n, n, 2, 2), -1)
    
    for m in range(1, n):
        for s in range(n - m):
            t = s + m

            open = (table[s, s:t, CLOSE, LEFT]
                    + table[(s + 1):(t + 1), t, CLOSE, RIGHT])

            open_right = open + scores[t, s]
            o_r_argmax = np.argmax(open_right)
            table[s, t, OPEN, RIGHT] = open_right[o_r_argmax]
            backpointers[s, t, OPEN, RIGHT] = s + o_r_argmax
            
            open_left = open + scores[s, t]
            o_l_argmax = np.argmax(open_left)
            table[s, t, OPEN, LEFT] = open_left[o_l_argmax]
            backpointers[s, t, OPEN, LEFT] = s + o_l_argmax

            close_right = (table[s, s:t, CLOSE, RIGHT]
                           + table[s:t, t, OPEN, RIGHT])
            c_r_argmax = np.argmax(close_right)
            table[s, t, CLOSE, RIGHT] = close_right[c_r_argmax]
            backpointers[s, t, CLOSE, RIGHT] = s + c_r_argmax
            
            close_left = (table[s, (s + 1):(t + 1), OPEN, LEFT]
                          + table[(s + 1):(t + 1), t, CLOSE, LEFT])
            c_l_argmax = np.argmax(close_left)
            table[s, t, CLOSE, LEFT] = close_left[c_l_argmax]
            backpointers[s, t, CLOSE, LEFT] = s + 1 + c_l_argmax

    arcs = np.full(n, -1)
    _backtrack(backpointers, 0, n - 1, CLOSE, LEFT, arcs)

    return arcs


def _backtrack(backpointers: np.ndarray, s: int, t: int,
               structure: int, head: int, arcs: np.ndarray) -> None:
    """Recursively follows backpointers to get the arcs.
    
    Args:
        backpointers: An n x n NumPy int array containing backpointers.
        s: An int containing the start of the span.
        t: An int containing the end of the span.
        structure: An int containing the index for structure type.
            OPEN = 0, CLOSE = 1.
        head: An int containing the index for head direction.
            RIGHT = 0, LEFT = 1.
        arcs: A size n NumPy int array indexed by the dependent ID.
    """
    r = backpointers[s, t, structure, head]
    if r == -1:
        return
    if structure == OPEN:
        if head == RIGHT:
            arcs[s] = t
        else:
            arcs[t] = s
        _backtrack(backpointers, s, r, CLOSE, LEFT, arcs)
        _backtrack(backpointers, r + 1, t, CLOSE, RIGHT, arcs)
    else:
        if head == RIGHT:
            _backtrack(backpointers, s, r, CLOSE, RIGHT, arcs)
            _backtrack(backpointers, r, t, OPEN, RIGHT, arcs)
        else:
            _backtrack(backpointers, s, r, OPEN, LEFT, arcs)
            _backtrack(backpointers, r, t, CLOSE, LEFT, arcs)