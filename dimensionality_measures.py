# adapted from https://github.com/Cecca/role-of-dimensionality/blob/fb94a8b1e52c7f71c9fcc5024b5592d68a8c6aac/additional-scripts/compute-lid.py

import numpy as np
import pandas as pd
import utils


def compute_lid(distances, k, scale="log"):
    assert len(distances) >= k
    w = distances[min(len(distances) - 1, k)]
    half_w = 0.5 * w

    distances = distances[:k+1]
    distances = distances[distances > 1e-5]

    small = distances[distances < half_w]
    large = distances[distances >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

    assert s != 0
    if scale == "log":
        return np.log(-valid / s)
    else:
        return -valid / s


def compute_rc(distances, k, scale="log"):
    avg_dist = distances.mean()
    if scale == "log":
        return np.log( avg_dist / distances[k] )
    else:
        return avg_dist / distances[k]


def compute_expansion(distances, k, scale="log"):
    if scale == "log":
        return np.log( distances[2*k] / distances[k] )
    else:
        return distances[2*k] / distances[k]

def compute(query, dataset, metric, k, distance_metric="euclidean"):
    assert k >= 1
    assert len(query) == dataset.shape[1], f"Query shape {len(query)}, dataset dimensions {dataset.shape[1]}"
    distances = utils.compute_distances(query, None, distance_metric, dataset)
    assert (distances >= 0).all()
    if metric == "lid":
        return compute_lid(distances, k, scale="linear")
    elif metric == "loglid":
        return compute_lid(distances, k, scale="log")
    elif metric == "logrc":
        return compute_rc(distances, k, scale="log")
    elif metric == "rc":
        return compute_rc(distances, k, scale="linear")
    elif metric == "logexpansion":
        return compute_expansion(distances, k, scale="log")
    elif metric == "expansion":
        return compute_expansion(distances, k, scale="linear")
    else:
        raise Exception(f"Unknown metric {metric}")

def compute_metrics(query, dataset, k, scale="log", distance_metric="euclidean"):
    assert query.shape[0] == dataset.shape[1], "data and query are expected to have the same dimension"

    distances = utils.compute_distances(query, k, distance_metric, dataset)
    lid = compute_lid(distances, k, scale)
    rc = compute_rc(distances, k, scale)
    expansion = compute_expansion(distances, k, scale)

    return lid, rc, expansion


if __name__ == "__main__":
    import tqdm
    import sys
    import bench
    dataname = sys.argv[1]
    dataset, queries, distances, distance_metric = bench.load_dataset(dataname)

    k = 10

    print("i,lid,rc,expansion")
    for i, q in tqdm.tqdm(enumerate(queries), total=queries.shape[0]):
        lid, rc, expansion = compute_metrics(q, dataset, k, distance_metric=distance_metric, scale="linear")
        print(f"{i},{lid},{rc},{expansion}")


