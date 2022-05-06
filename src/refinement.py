import itertools as it
import numpy as np

from tsp.core.tsp import distance as calculate_distance
from tsp.extra.visgraph import shortest_path, calculate_visgraph


def cheapest_insertion(centroids, nodes, prev_centroid, next_centroid):
    min_distance = float('inf')
    result = None
    for partial_tour in it.permutations(nodes):
        if prev_centroid is not None:
            partial_tour_centroids = [prev_centroid] + [centroids[i] for i in partial_tour] + [next_centroid]
        else:
            partial_tour_centroids = [centroids[i] for i in partial_tour]
            partial_tour_centroids.append(partial_tour_centroids[0])  # make closed tour!
        distance = 0.
        for i in range(1, len(partial_tour_centroids)):
            distance += np.sqrt(np.sum(np.square(partial_tour_centroids[i] - partial_tour_centroids[i - 1])))
        if distance < min_distance:
            min_distance = distance
            result = partial_tour
    return result


def cheapest_insertion_obs(centroids, nodes, prev_centroid, next_centroid, tsp):
    if prev_centroid is None:
        graph = calculate_visgraph([centroids[i] for i in nodes], tsp.obstacles, (tsp.w, tsp.h))
    else:
        graph = calculate_visgraph([centroids[i] for i in nodes] + [prev_centroid, next_centroid], tsp.obstacles, (tsp.w, tsp.h))
    min_distance = float('inf')
    result = None
    for partial_tour in it.permutations(nodes):
        if prev_centroid is not None:
            partial_tour_centroids = [prev_centroid] + [centroids[i] for i in partial_tour] + [next_centroid]
        else:
            partial_tour_centroids = [centroids[i] for i in partial_tour]
            partial_tour_centroids.append(partial_tour_centroids[0])  # make closed tour!
        distance = 0.
        for i in range(1, len(partial_tour_centroids)):
            distance += calculate_distance(shortest_path(partial_tour_centroids[i], partial_tour_centroids[i - 1], graph))
        if distance < min_distance:
            min_distance = distance
            result = partial_tour
    return result


def solve_level(c, v, level, subcluster, tour, prev_centroid=None, next_centroid=None):
    centroids = c[level]
    nodes = v[level][subcluster]
    # prev_centroid = c[0][tour[-1]] if tour else None
    partial_tour = cheapest_insertion(centroids, nodes, prev_centroid, next_centroid)
    if level == 0:
        tour.extend(partial_tour)
    else:
        for i, node in enumerate(partial_tour):
            if tour:
                next_prev_centroid = c[0][tour[-1]]
            else:
                next_prev_centroid = c[level][partial_tour[-1]]
            if i + 1 == len(partial_tour):
                solve_level(c, v, level - 1, node, tour, next_prev_centroid, next_centroid if next_centroid is not None else c[0][tour[0]])
            else:
                solve_level(c, v, level - 1, node, tour, next_prev_centroid, c[level][partial_tour[i + 1]])
