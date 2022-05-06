from typing import List, Set, Tuple
from numpy.typing import NDArray
from collections import namedtuple, defaultdict
import itertools as it
import numpy as np

from tree_split import citation55
from refinement import solve_level

from tsp.core.tsp import N_TSP


def do_split(v, e, edges, k):
    return citation55(v, e, edges, k)


def _find_parent(i, parents):
    while parents[i] != i:
        i = parents[i]
    return i


def cluster_boruvka(nodes: List, k: int):
    c = [nodes]
    v = []
    e = []
    while not v or len(v[-1]) > 1:
        n = len(c[-1])
        v.append(list(map(lambda i: [i], range(n))))

        edges = np.zeros((n, n), dtype=np.float)
        for i in range(n):
            edges[i, i] = np.inf
            for j in range(i + 1, n):
                value = np.sqrt(np.sum(np.square(c[-1][i] - c[-1][j])))
                edges[i, j] = value
        i_lower = np.tril_indices(n, -1)
        edges[i_lower] = edges.T[i_lower]

        minimum_edges = {i : (np.inf, None, None) for i in range(n)}
        for c1, c2 in it.combinations(range(n), 2):
            for edge in it.product(v[-1][c1], v[-1][c2]):
                if edges[edge] < minimum_edges[c1][0]:
                    minimum_edges[c1] = edges[edge], edge, c2
                if edges[edge] < minimum_edges[c2][0]:
                    minimum_edges[c2] = edges[edge], edge, c1

        parents = list(range(n))
        edge_tracker = defaultdict(set)
        for c1, (_, edge, c2) in sorted(minimum_edges.items(), key=lambda t: t[1][0]):
            c1_parent = _find_parent(c1, parents)
            c2_parent = _find_parent(c2, parents)
            if c1_parent != c2_parent:
                parents[c2_parent] = c1_parent
                edge_tracker[c1_parent].add(edge)

        parents = [_find_parent(i, parents) for i in parents]
        vertex_tracker = defaultdict(list)
        centroid_tracker = defaultdict(list)
        for i, p in enumerate(parents):
            vertex_tracker[p].append(i)
            centroid_tracker[p].append(c[-1][i])
            if i != p:
                edge_tracker[p].update(edge_tracker[i])
                del edge_tracker[i]

        c.append([])
        v[-1] = []
        e.append([])
        for p in set(parents):
            if len(vertex_tracker[p]) > k:
                split_v, split_e = do_split(
                    vertex_tracker[p],
                    list(edge_tracker[p]),
                    edges,
                    k
                )
                split_c = []
                for vertices in split_v:
                    split_c.append(np.mean([c[-2][i] for i in vertices], axis=0))
                c[-1].extend(split_c)
                v[-1].extend(split_v)
                e[-1].extend(split_e)
            else:
                c[-1].append(np.mean(centroid_tracker[p], axis=0))
                v[-1].append(vertex_tracker[p])
                e[-1].append(list(edge_tracker[p]))

    return c, v, e


def pyramid_solve(tsp: N_TSP, k: int = 6) -> NDArray:
    nodes = list(map(lambda a: np.array(a, dtype=np.float64), tsp.cities))
    c, v, e = cluster_boruvka(nodes, k)
    result = []
    assert len(v[-1]) == 1
    solve_level(c, v, len(v) - 1, 0, result)
    return np.array(result)
