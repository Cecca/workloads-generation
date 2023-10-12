# adapted from https://github.com/Cecca/role-of-dimensionality/blob/fb94a8b1e52c7f71c9fcc5024b5592d68a8c6aac/additional-scripts/compute-lid.py

import numpy as np
import pandas as pd


def compute_lid(distances, k, scale="log"):
    w = distances[min(len(distances) - 1, k)]
    half_w = 0.5 * w

    distances = distances[:k+1]
    distances = distances[distances > 1e-5]

    small = distances[distances < half_w]
    large = distances[distances >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

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
    assert len(query) == dataset.shape[1], f"Query shape {len(query)}, dataset dimensions {dataset.shape[1]}"
    if distance_metric == "euclidean":
        distances = np.linalg.norm(query - dataset, axis=1)
    else:
        distances = 1 - np.dot(dataset, query)
    np.ndarray.sort(distances)
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

    if distance_metric == "euclidean":
        distances = np.linalg.norm(query - dataset, axis=1)
    elif distance_metric == "angular":
        distances = 1 - np.dot(dataset, query)
    else:
        raise Exception("unknown distance metric")
    np.ndarray.sort(distances)

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


