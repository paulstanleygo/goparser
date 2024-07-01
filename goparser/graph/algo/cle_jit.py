'''This module contains the Chu-Liu-Edmonds algorithm for decoding the maximum spanning
tree of a graph. The functions are just-in-time compiled using Numba for faster performance.
'''
from datetime import timedelta
from time import perf_counter
from numba import njit
from numba.core import types
from numba.typed import Dict, List
import numpy as np


@njit
def chu_liu_edmonds(scores: np.ndarray) -> np.ndarray:
    """Finds the maximum spanning tree of a graph with the Chu-Liu-Edmonds algorithm.
    
    Args:
        scores: An n x n NumPy float or int array containing scores for every possible arc.
        The row index should be the head ID and the column index should be the dependent ID.
    
    Returns:
        A size n NumPy int array indexed by the dependent ID containing the ID of its head.
    """
    m, n = np.shape(scores)
    if m != n:
        raise ValueError('\'scores\' must be an n x n array')
    
    score_array = np.copy(scores)

    current_nodes = np.full(n, True)
    
    previous_leaving = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        previous_leaving[i] = np.arange(n)
    
    previous_entering = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        previous_entering[i] = np.full(n, i)
    
    representatives = List()
    for i in range(n):
        d = Dict.empty(
            key_type=types.int64,
            value_type= types.boolean
        )
        d[i] = True
        representatives.append(d)
    
    arcs = np.full(n, -1)

    _cle(score_array, current_nodes,
         previous_leaving, previous_entering,
         representatives, arcs)
    
    return arcs


@njit
def _cle(score_array: np.ndarray, current_nodes: np.ndarray,
         previous_leaving: np.ndarray, previous_entering: np.ndarray,
         representatives: List[Dict[int, int]], arcs: np.ndarray) -> None:
    """Recursively runs the Chu-Liu-Edmonds algorithm.
    
    Args:
        score_array: An n x n NumPy float or int array containing the arc scores.
        current_nodes: An n x 1 NumPy boolean array indicating if a node is current.
        previous_leaving: An n x n NumPy int array containing the previous leaving edges.
        previous_entering: An n x n NumPy int array containing the previous entering edges.
        representatives: A list of dictionaries with int keys as the representative nodes.
        arcs: A size n NumPy int array of arcs indexed by the dependent ID.
    """
    n = current_nodes.size

    # select best-scoring heads greedily
    heads = np.full(n, -1)
    heads[1:][current_nodes[1:]] = 0
    for i in range(1, n):
        if not current_nodes[i]:
            continue
        best_score = score_array[0, i]
        for j in range(n):
            if j == i or not current_nodes[j]:
                continue
            if score_array[j, i] > best_score:
                best_score = score_array[j, i]
                heads[i] = j
    
    # find a cycle; if none, return MST
    cycle = _find_one_cycle(heads, current_nodes)
    if not cycle:
        for i in range(1, n):
            if not current_nodes[i]:
                continue
            head = previous_entering[heads[i], i]
            dependent = previous_leaving[heads[i], i]
            arcs[dependent] = head
        return
    
    cycle_nodes = List(cycle.keys())
    representative = cycle_nodes[0]

    # contract the cycle
    cycle_weight = 0.0
    for j in cycle_nodes:
        cycle_weight += score_array[heads[j], j]
    
    for i in range(n):
        if i in cycle or not current_nodes[i]:
            continue

        leaving_weight = -np.inf
        entering_weight = -np.inf

        for j in cycle_nodes:
            score = cycle_weight + score_array[i, j] - score_array[heads[j], j]
            if score > leaving_weight:
                leaving_weight = score
                leaving_edge = j
            
            if score_array[j, i] > entering_weight:
                entering_weight = score_array[j, i]
                entering_edge = j
            
        score_array[i, representative] = leaving_weight
        previous_leaving[i, representative] = previous_leaving[i, leaving_edge]
        previous_entering[i, representative] = previous_entering[i, leaving_edge]

        score_array[representative, i] = entering_weight
        previous_entering[representative, i] = previous_entering[entering_edge, i]
        previous_leaving[representative, i] = previous_leaving[entering_edge, i]

    reps_considered = List()
    for i in cycle_nodes:
        rep_con = Dict.empty(
            key_type=types.int64,
            value_type=types.boolean
        )
        keys = List(representatives[i])
        for key in keys:
            rep_con[key] = True
        reps_considered.append(rep_con)
    
    for i in cycle_nodes[1:]:
        current_nodes[i] = False
        for j in representatives[i]:
            representatives[representative][j] = True
    
    # call CLE recursively
    _cle(score_array, current_nodes,
         previous_leaving, previous_entering,
         representatives, arcs)

    # resolve the cycle
    found = False
    for i, rep in enumerate(reps_considered):
        for key in rep:
            if arcs[key] != -1:
                key_node = cycle_nodes[i]
                found = True
                break
        if found:
            break
    
    previous = heads[key_node]
    while previous != key_node:
        dependent = previous_leaving[heads[previous], previous]
        head = previous_entering[heads[previous], previous]
        arcs[dependent] = head
        previous = heads[previous]


@njit
def _find_one_cycle(heads: np.ndarray, current_nodes: np.ndarray) -> Dict[int, int]:
    """Finds a cycle in the graph.
    
    Args:
        heads: An n x 1 NumPy int array containing the head nodes indexed by the dependent ID.
        current_nodes: An n x 1 NumPy boolean array indicating if node is current.
    
    Returns:
        A dictionary containing a cycle indexed by the dependent ID.
    """
    n = current_nodes.size
    found_cycle = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    added = np.full(n, False)
    for i in range(n):
        if found_cycle:
            break
        if added[i] or not current_nodes[i]:
            continue
        added[i] = True
        cycle = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        cycle[i] = 0
        current_node = i
        while True:
            if heads[current_node] == -1:
                added[current_node] = True
                break
            if heads[current_node] in cycle:
                cycle = Dict.empty(
                    key_type=types.int64,
                    value_type=types.int64
                )
                origin = heads[current_node]
                cycle[origin] = heads[origin]
                added[origin] = True
                next_node = heads[origin]
                while next_node != origin:
                    cycle[next_node] = heads[next_node]
                    added[next_node] = True
                    next_node = heads[next_node]
                found_cycle = cycle
                break
            cycle[current_node] = 0
            current_node = heads[current_node]
            if added[current_node] and current_node not in cycle:
                break
            added[current_node] = True
    return found_cycle