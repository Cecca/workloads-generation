import h5py
import os
import numpy as np

dataset_path="/data/qwang/datasets/"
noise_q_path="/mnt/hddhelp/ts_benchmarks/datasets/workloads_karima/sald/"

DATASETS = {
"astro": f"{dataset_path}astro-256-100m.bin",
"deep1b": f"{dataset_path}deep1b-96-100m.bin",
"f10": f"{dataset_path}f10-256-100m.bin",
"f5": f"{dataset_path}f5-256-100m.bin",
"rw": f"{dataset_path}rw-256-100m.bin",
"seismic": f"{dataset_path}seismic-256-100m.bin",
"sald": f"{dataset_path}sald-128-100m.bin",
"fashion-mnist": ".fashion-mnist-784-euclidean.hdf5",
"glove-100": ".glove-100-angular.hdf5",
"glove-25": ".glove-25-angular.hdf5",
"glove-200": "glove-200-angular.hdf5",
"mnist": "mnist-784-euclidean.hdf5",
"sift": "sift-128-euclidean.hdf5"
}

WORKLOADS = {
"astro": f"{dataset_path}astro-256-1k.bin", 
"deep1b": f"{dataset_path}deep1b-96-1k.bin", 
"f10": f"{dataset_path}f10-256-1k.bin", 
"f5": f"{dataset_path}f5-256-1k.bin", 
"rw": f"{dataset_path}rw-256-1k.bin", 
"seismic": f"{dataset_path}seismic-256-1k.bin",
"sald": f"{dataset_path}sald-128-1k.bin",
"sald-noise-50": f"{noise_q_path}sald-128-10k-hard50p-znorm.bin",
"sald-noise-30": f"{noise_q_path}sald-128-10k-hard30p-znorm.bin",
"sald-noise-10": f"{noise_q_path}sald-128-10k-hard10p-znorm.bin",
"sald-noise-1": f"{noise_q_path}sald-128-10k-hard1p-znorm.bin",
"fashion-mnist": ".fashion-mnist-784-euclidean.hdf5",
"glove-100": ".glove-100-angular.hdf5",
"glove-25": ".glove-25-angular.hdf5",
"glove-200": "glove-200-angular.hdf5",
"mnist": "mnist-784-euclidean.hdf5",
"sift": "sift-128-euclidean.hdf5"
}

def read_from_hdf5(filename):
        with h5py.File(filename) as hfp:
            # dataset = hfp['train'][:data_limit]
            # queries = hfp['test'][:query_limit]
            dataset = hfp['train'][:]
            queries = hfp['test'][:]

            distances = hfp['distances'][:]
            distance_metric = hfp.attrs['distance']
        return dataset, queries, distances, distance_metric

def read_from_txt(filename):
    data = np.loadtxt(filename)
    return data

def read_from_bin(filename, sids, sdim):
    data = np.fromfile(filename, dtype=np.float32)

    return data.reshape(sids,sdim)

def read_data(dataset_name, queryset_name,data_limit, query_limit):
    data_path = DATASETS[dataset_name]
    query_path = WORKLOADS[queryset_name]

    data_samples, data_features = 0, 0

    # Read dataset and query based on file extension
    if data_path.endswith('.txt') and query_path.endswith('.txt'):
        dataset = read_from_txt(data_path)
        queries = read_from_txt(query_path)
        distances = get_distances(dataset)
        distance_metric = "euclidean"
    elif data_path.endswith('.hdf5'): 
        url = "http://ann-benchmarks.com/" + data_path
        if not os.path.isfile(data_path):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(data_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        dataset, queries, distances, distance_metric = read_from_hdf5(data_path)
    elif data_path.endswith('.bin'):
        data_samples, data_features = parse_filename(data_path)
        query_samples, query_features = parse_filename(query_path)

        assert data_features == query_features

        dataset = np.fromfile(data_path, dtype='float32', count=data_features*data_limit).reshape(data_limit, data_features)
        queries = np.fromfile(query_path, dtype='float32', count=query_features*query_limit).reshape(query_limit, query_features)
        
        distances = get_distances(dataset)
        distance_metric = "euclidean"
    else:
        print("Invalid file extension. Supported formats: .txt, .hdf5, .bin")
        sys.exit

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

def get_distances(dataset,distance_metric="euclidean"):
    #TODO
    distances = None
    return distances