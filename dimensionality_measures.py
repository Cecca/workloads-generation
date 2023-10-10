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


def run_benchmark(k=10, runs=10):
    import os
    import h5py
    import faiss
    import time
    from tqdm import tqdm
    import requests

    output_file = "ans.csv"
    if os.path.isfile(output_file):
        return pd.read_csv(output_file)
    data_file = ".fashion-mnist-784-euclidean.hdf5"
    data_file = ".glove-100-angular.hdf5"
    url = f"http://ann-benchmarks.com/{data_file[1:]}"
    if not os.path.isfile(data_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(data_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    with h5py.File(data_file) as hfp:
        dataset = hfp['train'][:]
        queries = hfp['test'][:10]
        distances = hfp['distances'][:]
        distance_metric = hfp.attrs['distance']

    print("Building index for metric", distance_metric)
    if distance_metric == "euclidean":
        index = faiss.IndexHNSWFlat(dataset.shape[1], 16)
        index.add(dataset)
    elif distance_metric == "angular":
        index = faiss.IndexHNSWFlat(dataset.shape[1], 16)
        dataset /= np.linalg.norm(dataset)
        dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        index.add(dataset)
    else:
        raise Exception(f"Unsupported distance metric {distance_metric}")
    print("Index built")

    def compute_recall(dataset_distances, run_distances, count, epsilon=1e-3):
        t = dataset_distances[count - 1] + epsilon
        actual = 0
        for d in run_distances[:count]:
            if d <= t:
                actual += 1
        return float(actual) / float(count)


    with open(output_file, "w") as fp:
        print("i,lid,rc,expansion,time,recall", file=fp)
        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:]
            lid, rc, expansion = compute_metrics(query, dataset, k)

            qq = np.array([query]) # just to comply with faiss API
            # index.hnsw.efSearch = k * 2
            run_dists, _ = index.search(qq, k)
            run_dists = np.sqrt(run_dists[0]) # Faiss returns a matrix of squared distances
            rec = compute_recall(distances[i,:], run_dists, k)
            start = time.time()
            for _ in range(runs):
                index.search(qq, k)
            end = time.time()

            estimate = (end - start) / runs
            print(f"{i}, {lid}, {rc}, {expansion}, {estimate}, {rec}", file=fp)

    return pd.read_csv(output_file)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import itertools

    bench = run_benchmark(k = 20, runs = 3)

    print(bench.drop("i", axis=1).corr('spearman'))
    print("minimum recall", bench['recall'].min())

    for x, y in itertools.combinations(["expansion", "lid", "rc", "recall", "time"], 2):
        plt.figure()
        sns.scatterplot(
            data = bench,
            y    = y,
            x    = x
        )
        plt.savefig(f"imgs/scatter-{x}-{y}.png")


