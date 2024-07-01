'''This module contains the Chu-Liu-Edmonds algorithm for decoding the maximum spanning
tree of a graph.
'''
import numpy as np


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
    
    score_list = scores.tolist()
    current_nodes = [True] * n
    previous_leaving = [[i for i in range(n)] for _ in range(n)]
    previous_entering = [[i] * n for i in range(n)]
    representatives = [{i} for i in range(n)]
    arcs = [-1] * n
    _cle(score_list, current_nodes,
         previous_leaving, previous_entering,
         representatives, arcs)
    return np.array(arcs)


def _cle(score_list: list[list[float|int]], current_nodes: list[bool],
         previous_leaving: list[list[int]], previous_entering: list[list[int]],
         representatives: list[set[int]], arcs: list[int]) -> None:
    """Recursively runs the Chu-Liu-Edmonds algorithm.
    
    Args:
        score_list: An n x n list containing floats or ints of the arc scores.
        current_nodes: An n x 1 list containing booleans indicating if node is current.
        previous_leaving: An n x n list containing ints of the previous leaving edges.
        previous_entering: An n x n list containing ints of the previous entering edges.
        representatives: A list of sets containing ints of the representative nodes.
        arcs: A list of arcs indexed by the dependent ID containing the ID of its head.
    """
    n = len(current_nodes)

    # select best-scoring heads greedily
    heads = [0 if current_nodes[i] else -1 for i in range(n)]
    heads[0] = -1
    for i in range(1, n):
        if not current_nodes[i]:
            continue
        best_score = score_list[0][i]
        for j in range(n):
            if j == i or not current_nodes[j]:
                continue
            if score_list[j][i] > best_score:
                best_score = score_list[j][i]
                heads[i] = j
    
    # find a cycle; if none, return MST
    cycle = _find_one_cycle(heads, current_nodes)
    if not cycle:
        for i in range(1, n):
            if not current_nodes[i]:
                continue
            head = previous_entering[heads[i]][i]
            dependent = previous_leaving[heads[i]][i]
            arcs[dependent] = head
        return
    
    cycle_nodes = list(cycle.keys())
    representative = cycle_nodes[0]

    # contract the cycle
    cycle_weight = 0.0
    for j in cycle_nodes:
        cycle_weight += score_list[heads[j]][j]
    
    for i in range(n):
        if i in cycle or not current_nodes[i]:
            continue

        leaving_weight = -float("inf")
        entering_weight = -float("inf")

        for j in cycle_nodes:
            score = cycle_weight + score_list[i][j] - score_list[heads[j]][j]
            if score > leaving_weight:
                leaving_weight = score
                leaving_edge = j

            if score_list[j][i] > entering_weight:
                entering_weight = score_list[j][i]
                entering_edge = j
            
        score_list[i][representative] = leaving_weight
        previous_leaving[i][representative] = previous_leaving[i][leaving_edge]
        previous_entering[i][representative] = previous_entering[i][leaving_edge]
        
        score_list[representative][i] = entering_weight
        previous_entering[representative][i] = previous_entering[entering_edge][i]
        previous_leaving[representative][i] = previous_leaving[entering_edge][i]
    
    reps_considered = [list(representatives[i]) for i in cycle_nodes]
    
    for i in cycle_nodes[1:]:
        current_nodes[i] = False
        for j in representatives[i]:
            representatives[representative].add(j)
    
    # call CLE recursively
    _cle(score_list, current_nodes,
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
        dependent = previous_leaving[heads[previous]][previous]
        head = previous_entering[heads[previous]][previous]
        arcs[dependent] = head
        previous = heads[previous]


def _find_one_cycle(heads: list[int], current_nodes: list[bool]) -> dict[int, int]:
    """Finds a cycle in the graph.
    
    Args:
        heads: A list of ints containing the head nodes indexed by the dependent ID.
        current_nodes: An n x 1 list containing booleans indicating if node is current.
    
    Returns:
        A dictionary containing a cycle indexed by the dependent ID.
    """
    n = len(current_nodes)
    found_cycle = dict[int, int]()
    added = [False] * n
    for i in range(n):
        if found_cycle:
            break
        if added[i] or not current_nodes[i]:
            continue
        added[i] = True
        cycle = {i: 0}
        current_node = i
        while True:
            if heads[current_node] == -1:
                added[current_node] = True
                break
            if heads[current_node] in cycle:
                cycle = dict[int, int]()
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