from numpy.typing import NDArray
import numpy as np

import itertools as it

from tsp.core.tsp import N_TSP
from cluster import cluster_boruvka
from refinement import cheapest_insertion


def solve_level(c, v, level, subcluster, prev_centroid=None, prev_centroids=[], next_centroid=None):
    centroids = c[level]
    nodes = v[level][subcluster] + prev_centroids
    return cheapest_insertion(centroids, nodes, prev_centroid, next_centroid)


def pyramid_solve(tsp: N_TSP, k: int = 6, s: int = 1) -> NDArray:
    nodes = list(map(lambda a: np.array(a, dtype=np.float64), tsp.cities))
    c, v, e = cluster_boruvka(nodes, k)
    level = len(v) - 1
    result = solve_level(c, v, level, 0)
    while level > 0:
        level -= 1
        new_result = []
        for i, subcluster in enumerate(result):
            if new_result:
                prev_tour = new_result[-s:]
                if len(prev_tour) > 1:
                    new_result = new_result[:-(len(prev_tour)-1)]
                if i + 1 == len(result):
                    new_result.extend(solve_level(c, v, level, subcluster, c[level][prev_tour[0]], prev_tour[1:], c[level][new_result[0]]))
                else:
                    new_result.extend(solve_level(c, v, level, subcluster, c[level][prev_tour[0]], prev_tour[1:], c[level + 1][result[(i + 1) % len(result)]]))
            else:
                new_result.extend(solve_level(c, v, level, subcluster, prev_centroid=c[level + 1][result[i - 1]], next_centroid=c[level + 1][result[(i + 1) % len(result)]]))
        result = new_result
    assert len(result) == len(nodes)
    return np.array(result)
