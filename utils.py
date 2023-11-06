import numpy as np


DATASETS = {
    "fashion-mnist": ".fashion-mnist-784-euclidean.hdf5",
    "glove-100": ".glove-100-angular.hdf5",
    "glove-25": ".glove-25-angular.hdf5"
}


def compute_distances(query, k, metric, faiss_index=None, dataset=None):
    assert faiss_index is not None or dataset is not None, "you should provide either an index or the data itself"
    if faiss_index is not None:
        qq = np.array([query]) # just to comply with faiss API
        dists = faiss_index.search(qq, k)[0][0]
        if metric == "angular":
            # The index returns squared euclidean distances, 
            # which we turn to angular distances in the following
            return 1 - (2 - dists) / 2
        elif metric == "euclidean" :
            # The index returns the _squared_ euclidean distances
            return np.sqrt(dists)
        else:
            raise RuntimeError("unknown distance" + metric)
    elif dataset is not None:
        if metric == "angular":
            dists = 1 - np.dot(dataset, query) 
        elif metric == "euclidean":
            dists = np.linalg.norm(query - dataset, axis=1)
        else:
            raise RuntimeError("unknown distance" + metric)
        dists = np.partition(dists, k)[:k]
        return np.sort(dists)



def compute_recall(ground_distances, run_distances, count, epsilon=1e-3):
    t = ground_distances[count - 1] + epsilon
    actual = 0
    for d in run_distances[:count]:
        if d <= t:
            actual += 1
    return float(actual) / float(count)


def load_dataset(name, nqueries = None):
    import requests
    import h5py
    import os

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

    if distance_metric == "angular":
        dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
    return dataset, queries, distances, distance_metric


if __name__ == "__main__":
    import faiss

    dataset_name = "fashion-mnist"
    dataset, queries, distances, distance_metric = load_dataset(
        dataset_name, nqueries=100)

    k = 10

    # Check that the brute force distance computation is correct
    for i, q in enumerate(queries):
        dists = compute_distances(q, k, distance_metric, dataset=dataset)
        ground = distances[i,:k]
        assert np.all( np.abs(dists - ground) < 0.001 )
    print("Exact search all OK")

    # Check that the approximate distance computation gives a reasonable recall
    n_list = 32
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    index.nprobe = 32
    for i, q in enumerate(queries):
        dists = compute_distances(q, k, distance_metric, faiss_index=index)
        ground = distances[i,:k]
        rec = compute_recall(ground, dists, k)
        assert rec >= 0.95
    print("Approximate search all OK")

