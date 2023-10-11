import numpy as np
import pandas as pd
import os
import h5py
import faiss
import time
from tqdm import tqdm
import requests
import dimensionality_measures as dm

DATASETS = {
    "fashion-mnist": ".fashion-mnist-784-euclidean.hdf5",
    "glove-100": ".glove-100-angular.hdf5"
}

def load_dataset(name, nqueries = None):
    data_file = DATASETS[name]
    url = f"http://ann-benchmarks.com/{data_file[1:]}"
    if not os.path.isfile(data_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(data_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    with h5py.File(data_file) as hfp:
        dataset = hfp['train'][:]
        if nqueries is not None:
            queries = hfp['test'][:nqueries]
        else:
            queries = hfp['test'][:]
        distances = hfp['distances'][:]
        distance_metric = hfp.attrs['distance']
    return dataset, queries, distances, distance_metric


def compute_recall(dataset_distances, run_distances, count, epsilon=1e-3):
    t = dataset_distances[count - 1] + epsilon
    actual = 0
    for d in run_distances[:count]:
        if d <= t:
            actual += 1
    return float(actual) / float(count)


def run_benchmark(dataset_name, k=10, target_recall=0.9, nqueries=None):
    output_file = "ans.csv"
    if os.path.isfile(output_file):
        return pd.read_csv(output_file)
    
    dataset, queries, distances, distance_metric = load_dataset(
        dataset_name, nqueries=nqueries)

    print("Building index for metric", distance_metric)
    n_list = 32
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    if distance_metric == "angular":
        dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
    index.train(dataset)
    index.add(dataset)
    print("Index built")

    with open(output_file, "w") as fp:
        print("i,lid,rc,expansion,distcomp,recall,time", file=fp)
        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:]
            lid, rc, expansion = dm.compute_metrics(query, dataset, k)
            qq = np.array([query]) # just to comply with faiss API
            distcomp = None
            elapsed = None
            for nprobe in range(1,1000):
                faiss.cvar.indexIVF_stats.reset()
                index.nprobe = nprobe
                tstart = time.time()
                run_dists = index.search(qq, k)[0][0]
                tend = time.time()
                elapsed = tend - tstart
                if distance_metric == "angular":
                    # The index returns squared euclidean distances, 
                    # which we turn to angular distances in the following
                    run_dists = 1 - (2 - run_dists) / 2
                else:
                    assert False, "fix this branch"
                rec = compute_recall(distances[i,:], run_dists, k)
                if rec >= target_recall:
                    distcomp = faiss.cvar.indexIVF_stats.ndis + + faiss.cvar.indexIVF_stats.nq * n_list
                    break

            assert distcomp is not None

            print(f"{i}, {lid}, {rc}, {expansion}, {distcomp}, {rec}, {elapsed}", file=fp)

    return pd.read_csv(output_file)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools

    bench = run_benchmark("glove-100", k=100, nqueries=None, target_recall=0.95)
    corrs = bench.drop('i', axis=1).corr('spearman')

    plt.figure()
    sns.heatmap(corrs, annot=True, fmt=".2f", linewidth=0.5, square=True)
    plt.tight_layout()
    plt.savefig("imgs/correlations.png")

    xs = ["lid", "rc", "expansion"]
    ys = ["recall", "distcomp", "time"]

    bins = 40

    for x, y in itertools.product(xs, ys):
        pdata = bench.copy()
        plt.figure()
        sns.scatterplot(
            data = pdata,
            y    = y,
            x    = x
        )
        plt.tight_layout()
        plt.savefig(f"imgs/scatter-{x}-{y}.png")

        pdata['bin'] = pd.cut(pdata[x], bins, labels=None)
        plt.figure(figsize=(4, 12))
        sns.barplot(
            data = pdata,
            x    = y,
            y    = 'bin'
        )
        plt.tight_layout()
        plt.savefig(f"imgs/barplot-{x}-{y}.png")
