import h5py
import os
import numpy as np
import requests
from utils import compute_distances
import sys

dataset_path="/data/qwang/datasets/"
sald_noise_path="/mnt/hddhelp/ts_benchmarks/datasets/sald/"
glove_noise_path = "/mnt/hddhelp/ts_benchmarks/datasets/annbench/glove100/"

DATASETS = {
"astro": f"{dataset_path}astro-256-100m.bin",
"deep1b": f"{dataset_path}deep1b-96-100m.bin",
"f10": f"{dataset_path}f10-256-100m.bin",
"f5": f"{dataset_path}f5-256-100m.bin",
"rw": f"{dataset_path}rw-256-100m.bin",
"seismic": f"{dataset_path}seismic-256-100m.bin",
"sald": f"{dataset_path}sald-128-100m.bin",
"fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
"glove-100": "glove-100-angular.hdf5",
"glove-25": "glove-25-angular.hdf5",
"glove-200": "glove-200-angular.hdf5",
"mnist": "mnist-784-euclidean.hdf5",
"sift": "sift-128-euclidean.hdf5",
"glove-100-bin": f"{glove_noise_path}glove-100-1183514-angular.bin"
}

WORKLOADS = {
"astro": f"{dataset_path}astro-256-1k.bin", 
"deep1b": f"{dataset_path}deep1b-96-1k.bin", 
"f10": f"{dataset_path}f10-256-1k.bin", 
"f5": f"{dataset_path}f5-256-1k.bin", 
"rw": f"{dataset_path}rw-256-1k.bin", 
"seismic": f"{dataset_path}seismic-256-1k.bin",
"sald": f"{dataset_path}sald-128-1k.bin",
"sald-noise-50": f"{sald_noise_path}sald-128-1k-hard50p.bin",
"sald-noise-30": f"{sald_noise_path}sald-128-1k-hard30p.bin",
"sald-noise-10": f"{sald_noise_path}sald-128-1k-hard10p.bin",
"fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
"glove-100": "glove-100-angular.hdf5",
"glove-25": "glove-25-angular.hdf5",
"glove-25-lid-25": "/tmp/glove-25-lid-25.hdf5",
"glove-25-lid-68": "/tmp/glove-25-lid-68.hdf5",
"glove-200": "glove-200-angular.hdf5",
"mnist": "mnist-784-euclidean.hdf5",
"sift": "sift-128-euclidean.hdf5",
"sift-lid20": "sift-lid20.hdf5",
"sift-lid50": "sift-lid50.hdf5",
"sift-lid100": "sift-lid100.hdf5",
"sift-lid400": "sift-lid400.hdf5",
"sift-rc2": "sift-rc2.hdf5",
"sift-rc1.4": "sift-rc1.4.hdf5",
"sift-rc1.1": "sift-rc1.1.hdf5",
"sift-rc1.04": "sift-rc1.04.hdf5",
"sift-rc1.02": "sift-rc1.02.hdf5",
"sift-rc3": "sift-rc3.hdf5",
"glove-noise-0": f"{glove_noise_path}glove-100-10k-hard0p-angular.bin",
"glove-noise-10": f"{glove_noise_path}glove-100-10k-hard10p-angular.bin",
"glove-noise-30": f"{glove_noise_path}glove-100-10k-hard30p-angular.bin",
"glove-noise-50": f"{glove_noise_path}glove-100-10k-hard50p-angular.bin"
}


def download(url, local):
    if not os.path.isfile(local):
        print("Downloading from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)


def _download_ann_benchmarks(data_path):
    """Downloads a dataset from ann-benchmarks.com"""
    url = f"http://ann-benchmarks.com/{data_path}"
    if not os.path.isfile(data_path):
        print("Downloading from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(data_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)


def read_hdf5(filename, what, limit=None):
    with h5py.File(filename) as hfp:
        distance_metric = hfp.attrs.get('distance', None)
        if limit is not None:
            data = hfp[what][:limit]
        else:
            data = hfp[what][:]
    if distance_metric == "angular" and what in ["train", "test"]:
        data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    return data, distance_metric


def read_from_txt(filename):
    data = np.loadtxt(filename)
    return data


def read_from_bin(filename, sids, sdim):
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape(sids,sdim)


def read_multiformat(name, what, data_limit=None):
    if not os.path.isfile(name):
        if what == "train":
            path = DATASETS[name]
        elif what == "test":
            path = WORKLOADS[name]
        else:
            raise Exception("`what` can only be one of `train` or `test`")
    else:
        path = name

    if path.endswith('.txt'):
        data = read_from_txt(path)
        if data_limit is not None:
            data = data[:data_limit, :]
        distance_metric = "euclidean"
    elif path.endswith('.hdf5'):
        if not os.path.isfile(path):
            _download_ann_benchmarks(path)
        data, distance_metric = read_hdf5(path, what, data_limit)
    elif path.endswith(".bin"):
        data_samples, data_features = parse_filename(path)

        if data_limit is None:
            data_limit = data_samples

        data = np.fromfile(path, dtype='float32', count=data_features*data_limit).reshape(data_limit, data_features)
        distance_metric = "euclidean"
    else:
        print("Invalid file extension. Supported formats: .txt, .hdf5, .bin")
        sys.exit()

    if distance_metric == "angular":
        data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

    return data, distance_metric


def read_data(dataset_name, queryset_name, data_limit=None, query_limit=None):
    data_path = DATASETS[dataset_name]
    query_path = WORKLOADS[queryset_name]

    data_samples, data_features = 0, 0

    # Read dataset and query based on file extension
    if data_path.endswith('.txt') and query_path.endswith('.txt'):
        dataset = read_from_txt(data_path)
        queries = read_from_txt(query_path)
        distance_metric = "euclidean"
        distances = compute_distances(queries, 100, distance_metric, dataset)
    elif data_path.endswith('.hdf5'):
        _download_ann_benchmarks(data_path)
        _download_ann_benchmarks(query_path)
        dataset, distance_metric = read_hdf5(data_path, "train", data_limit)
        queries, _ = read_hdf5(query_path, "test", query_limit)
        if data_limit is not None or data_path != query_path:
            # We have to recompute the distances, because the `distances`
            # matrix stored in the hdf5 file is relative to the _entire_
            # dataset, not parts of it
            print("WARNING: Computing ground truth distances on the fly, because we are using the `data_limit` parameter")
            if distance_metric == "angular":
                dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
                queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
            distances = compute_distances(queries, 100, distance_metric, dataset) # if we have k>100 ?
        else:
            print("data_path", data_path)
            print("query_path", query_path)
            distances, _ = read_hdf5(data_path, "distances", query_limit)
    elif data_path.endswith('.bin'):
        data_samples, data_features = parse_filename(data_path)
        query_samples, query_features = parse_filename(query_path)

        # assert data_limit is not None
        # assert query_limit is not None
        assert data_features == query_features

        if data_limit is None:
            data_limit = data_samples
        if query_limit is None:
            query_limit = query_samples

        dataset = np.fromfile(data_path, dtype='float32', count=data_features*data_limit).reshape(data_limit, data_features)
        queries = np.fromfile(query_path, dtype='float32', count=query_features*query_limit).reshape(query_limit, query_features)
        
        distance_metric = "euclidean"
        # distances = compute_distances(queries, 100, distance_metric, dataset)
        distances = None
    else:
        print("Invalid file extension. Supported formats: .txt, .hdf5, .bin")
        sys.exit()

    if distance_metric == "angular":
        dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]

    return dataset, queries, distances, distance_metric

def str_to_digits(sids_str):
    num_map = {'K':1000, 'M':1000000, 'B':1000000000}
    sids = 0
    if sids_str.isdigit():
        sids = int(sids_str) 
    else:
        sids = int(float(sids_str[:-1]) * num_map.get(sids_str[-1].upper(), 1))
    return sids

def parse_filename(filepath):   
    # parse sdim and sids /path/to/file/deep1b-96-1k.bin
    file = filepath.rsplit("/",1)[1].split('.')[0]
    file_arr = file.split('-')
    samples = str_to_digits(file_arr[2])
    features = int(file_arr[1])

    return samples, features


# Test a few things
if __name__ == "__main__":
    # Here we check that the distance-computing function makes sense.
    import faiss
    from utils import compute_recall
    import time

    dataset_name = "glove-100"
    dataset, queries, distances, distance_metric = read_data(
        dataset_name, dataset_name, data_limit=None, query_limit=16)

    k = 10

    # Check that the brute force distance computation is correct
    start = time.time()
    dists = compute_distances(queries, k, distance_metric, dataset)
    end = time.time()
    ground = distances[:len(dists), :k]
    assert np.allclose(dists, ground)
    print("Exact search all OK:", end - start, "seconds")

    # Check FAISS exact index
    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)
    start = time.time()
    dists = compute_distances(queries, k, distance_metric, index)
    end = time.time()
    ground = distances[:len(dists), :k]
    assert np.allclose(dists, ground)
    print("FAISS exact search all OK:", end - start, "seconds")
    
    # Check that the approximate distance computation gives a reasonable recall
    n_list = 32
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    index.nprobe = 32
    start = time.time()
    dists = compute_distances(queries, k, distance_metric, index)
    ground = distances[:len(dists), :k]
    recalls = np.array([
        compute_recall(g, d, k)
        for g, d in zip(ground, dists)
    ])
    assert np.all(recalls >= 0.95)
    end = time.time()
    print("Approximate search all OK:", end - start, "seconds")

