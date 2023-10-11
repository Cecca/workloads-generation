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


def compute_metrics(query, dataset, k, scale="log"):
    assert query.shape[0] == dataset.shape[1], "data and query are expected to have the same dimension"

    distances = np.linalg.norm(query - dataset, axis=1)
    np.ndarray.sort(distances)

    lid = compute_lid(distances, k, scale)
    rc = compute_rc(distances, k, scale)
    expansion = compute_expansion(distances, k, scale)

    return lid, rc, expansion


