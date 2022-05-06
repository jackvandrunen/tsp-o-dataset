from typing import List, Tuple
from numpy.typing import NDArray
from collections import defaultdict
import itertools as it
import numpy as np

from tree_split import citation55

from tsp.extra.visgraph import calculate_visgraph, shortest_path, Point, Graph, _distance
from tsp.core.tsp import distance


def vg_midpoint(a: Point, b: Point, graph: Graph, exclude: List[Point] = None) -> Point:
    path = shortest_path(a, b, graph, exclude)
    distance_cumsum = []
    total_distance = 0.0
    p1 = path[0]
    for p2 in path[1:]:
        total_distance += _distance(p1, p2)
        distance_cumsum.append(total_distance)
        p1 = p2
    for i, d in enumerate(distance_cumsum):
        if d > total_distance / 2.0:
            break
    if i > 0:
        norm = d - distance_cumsum[i - 1]
        dist_to_midpoint = (total_distance / 2.0) - distance_cumsum[i - 1]
    else:
        norm = d
        dist_to_midpoint = total_distance / 2.0
    v = -path[i][0] + path[i+1][0], -path[i][1] + path[i+1][1]
    u = v[0] * dist_to_midpoint / norm, v[1] * dist_to_midpoint / norm
    p = path[i]
    return int(p[0] + u[0]), int(p[1] + u[1])


def do_split(v, e, edges, k):
    return citation55(v, e, edges, k)


def _find_parent(i, parents):
    while parents[i] != i:
        i = parents[i]
    return i


def cluster_boruvka(nodes: List, obstacles: List, bound: Tuple[int, int] = None):
    k = 2
    c = [nodes]
    v = []
    e = []
    while not v or len(v[-1]) > 1:
        n = len(c[-1])
        v.append(list(map(lambda i: [i], range(n))))

        graph = calculate_visgraph(c[-1], obstacles, bound)

        edges = np.zeros((n, n), dtype=np.float)
        for i in range(n):
            edges[i, i] = np.inf
            for j in range(i + 1, n):
                value = distance(shortest_path(c[-1][i], c[-1][j], graph))
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
                    if len(vertices) == 2:
                        split_c.append(vg_midpoint(*[c[-2][i] for i in vertices], graph))
                    else:
                        assert len(vertices) == 1
                        split_c.append(c[-2][vertices[0]])
                c[-1].extend(split_c)
                v[-1].extend(split_v)
                e[-1].extend(split_e)
            else:
                if len(centroid_tracker[p]) == 2:
                    c[-1].append(vg_midpoint(*centroid_tracker[p], graph))
                else:
                    assert len(centroid_tracker[p]) == 1
                    c[-1].append(centroid_tracker[p][0])
                v[-1].append(vertex_tracker[p])
                e[-1].append(list(edge_tracker[p]))

    return c, v, e
