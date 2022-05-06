import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import itertools


if __name__ == '__main__':
    np.random.seed(122)
    xy = np.random.uniform(0,10, (20,2))
    D = distance_matrix(xy, xy)
    nsit = [(18,3), (3,8), (8,12), (12,15), (15,19), (19,7), (7,14), (14,6), (6,1), (1,5), (5,2),
        (2,4), (4,0), (0,13), (13,16), (16,9), (9,11), (11,17), (17,10), (10,18)]
    sit = [(18,3), (3,8), (8,12), (12,15), (15,19), (19,7), (7,14), (14,6),
        (4,0), (0,13), (13,16), (16,9), (9,11), (11,17), (17,10), (10,18), (6,2), (2,5), (5,1), (1,4)]


    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
    ax1.scatter(xy[:, 0], xy[:, 1])
    for i in range(xy.shape[0]):
        ax1.text(xy[i, 0], xy[i, 1], i)
    for i,j in nsit:
        ax1.plot([xy[i, 0], xy[j, 0]],[xy[i, 1], xy[j, 1]], color="orange")
    ax1.set_title("Good Tour")

    ax2.scatter(xy[:, 0], xy[:, 1])
    for i in range(xy.shape[0]):
        ax2.text(xy[i, 0], xy[i, 1], i)
    for i,j in sit:
        ax2.plot([xy[i, 0], xy[j, 0]],[xy[i, 1], xy[j, 1]], color="orange")
    ax2.set_title("Self Intersecting Tour")

    fig.show()



def is_self_intersecting(xy, edges, print_intersection=False):
    """
    input: xy (20 x 2 ndarray) if 20 city problem
           edges [(i,j), (k,l), ...], list of edges in tour
    output: True or False
    
    The idea is to find intersection of two edges and see if it's in the domain of both of them. 
   
    """
    eps = 1e-12
    for e1, e2 in itertools.combinations(edges, 2):
        dct = {}
        for k, e in enumerate([e1, e2]):
            x0, y0 = xy[e[0], 0], xy[e[0], 1]
            x1, y1 = xy[e[1], 0], xy[e[1], 1]
            if x0 != x1:
                dct[f"a{k}"] = (y1 - y0)/(x1 - x0)
                dct[f"b{k}"] = y1 - dct[f"a{k}"]*x1
            else:
                dct[f"a{k}"] = "Vertical"
                dct[f"y0{k}"] = min([y0, y1])
                dct[f"y1{k}"] = max([y0, y1])
                            
        d = []
        d += [sorted([xy[p, 0] for p in e]) for e in [e1,e2]]
        #print(f"domains = {d}")
        
        if dct["a0"] != "Vertical" and dct["a1"] != "Vertical":        
            if dct["a0"] == dct["a1"]:
                # parallel won't intersect
                continue
            else:
                X = (dct["b1"] - dct["b0"])/(dct["a0"] - dct["a1"])
                #print(f"{e1} -> {e2}, X = {X}")
                if d[0][0]+eps < X < d[0][1]- eps and d[1][0]+eps < X < d[1][1]-eps:
                    if print_intersection:
                        print("Domain = ", d)
                        Y = dct["a0"]*X + dct["b0"]
                        print(f"{e1} & {e2} intersect at ({np.round(X,3)}, {np.round(Y,3)})")
                    return True
                
        elif dct["a0"] != "Vertical" and dct["a1"] == "Vertical":
            #print(domains)
            X = d[1][0]
            Y = dct["a0"]*X + dct["b0"]
            if dct['y01'] + eps <= Y <= dct['y11'] - eps and d[0][0] < X < d[0][1]:
                if print_intersection: 
                    print(f"{e1} & {e2} intersect at ({np.round(X,3)}, {np.round(Y,3)})")
                return True
        
        elif dct["a1"] != "Vertical" and dct["a0"] == "Vertical":
            #print(domains)
            X = d[0][0]
            Y = dct["a1"]*X + dct["b1"]
            if dct['y00'] + eps <= Y <= dct['y10'] - eps and d[1][0] < X < d[1][1]:
                if print_intersection: 
                    print(f"{e1} & {e2} intersect at ({np.round(X,3)}, {np.round(Y,3)})")
                return True
        
        else:
            # both vertical - then no intersection if min 1 greater than max other, else intersection
            if dct['y00'] < dct['y01'] < dct['y10'] or dct['y00'] < dct['y11'] < dct['y10'] or dct['y01'] < dct['y00'] < dct['y11'] or dct['y01'] < dct['y10'] < dct['y11']:
                return True
        
    return False


def check_self_intersecting(tsp, tour):
    edges = []
    for i, n in enumerate(tour):
        edges.append((tour[i - 1], n))
    return is_self_intersecting(tsp.cities, edges)


if __name__ == '__main__':
    print(is_self_intersecting(xy, sit, print_intersection=True))
    print(is_self_intersecting(xy, nsit, print_intersection=True))
