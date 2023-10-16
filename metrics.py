import sys
import os
import numpy as np
import h5py
from tqdm import tqdm
import requests
from functools import partial
import csv
import faiss
import time

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

data_limit = 1000000
query_limit = 10000

def read_from_hdf5(filename):
        with h5py.File(filename) as hfp:
            # dataset = hfp['train'][:data_limit]
            # queries = hfp['test'][:query_limit]
            dataset = hfp['train'][:]
            queries = hfp['test'][:]
        return dataset, queries

def read_from_txt(filename):
    data = np.loadtxt(filename)
    return data

def read_from_bin(filename, sids, sdim):
    data = np.fromfile(filename, dtype=np.float32)

    return data.reshape(sids,sdim)

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


def compute_epsilon_hardness(distances, epsilon):
    min_dist = distances[0]
    epsilon_dist = (1 + epsilon) * min_dist
    epsilon_nn = distances[distances <= epsilon_dist]
    epsilon_hardness = len(epsilon_nn) / len(distances)
    return epsilon_hardness

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

def compute_metrics(query, dataset, epsilons, k, scale="log", distance_metric="euclidean"):
    assert query.shape[0] == dataset.shape[1], "data and query are expected to have the same dimension"

    # distances = np.linalg.norm(dataset - query, axis=1)  # Euclidean distance
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

    epsilons_hardness = [compute_epsilon_hardness(distances, e) for e in epsilons]

    return lid, rc, expansion, epsilons_hardness


def read_sys_argv_list(start_index=4):
    if len(sys.argv) >= start_index + 1:
        return sys.argv[start_index:]
    else:
        return None

def build_index(dataset):
    print("Building index")
    n_list = 32
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    # if distance_metric == "angular":
    #     dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
    #     queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
    index.train(dataset)
    index.add(dataset)
    print("Index built")

    return index

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python filename.py dataset_file query_file k [optional:epsilon as list with spaces]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    queryset_name = sys.argv[2]
    k_value = int(sys.argv[3])
    epsilons = []
    opt_params = read_sys_argv_list()
    if opt_params is not None:
        epsilons = [float(x) for x in opt_params]
    else: 
        epsilons.append(1)

    print (epsilons)
    data_samples, data_features = 0, 0

    epsilons_str = '_'.join(str(e) for e in epsilons)

    data_path = DATASETS[dataset_name]
    query_path = WORKLOADS[queryset_name]

    # Read dataset and query based on file extension
    if data_path.endswith('.txt') and query_path.endswith('.txt'):
        dataset = read_from_txt(data_path)
        queries = read_from_txt(query_path)
    elif data_path.endswith('.hdf5'): 
        url = "http://ann-benchmarks.com/" + data_path
        if not os.path.isfile(data_path):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(data_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        dataset, queries = read_from_hdf5(data_path)
    elif data_path.endswith('.bin'):
        data_samples, data_features = parse_filename(data_path)
        query_samples, query_features = parse_filename(query_path)

        assert data_features == query_features

        dataset = np.fromfile(data_path, dtype='float32', count=data_features*data_limit).reshape(data_limit, data_features)
        queries = np.fromfile(query_path, dtype='float32', count=query_features*query_limit).reshape(query_limit, query_features)
        
    else:
        print("Invalid file extension. Supported formats: .txt, .hdf5, .bin")
        sys.exit

    output_file = f"res_{dataset_name}_{queryset_name}_{k_value}_{epsilons_str}"
    
    # index = build_index(dataset)

    with open(output_file+'.csv', "w", newline="") as fp:
        writer = csv.writer(fp)

        header = ["i", "lid_"+str(k_value), "rc_"+str(k_value), f"exp_{2*k_value}|{k_value}"]
        header.extend(["eps_" + str(e) for e in epsilons])
        writer.writerow(header)


        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:]  
            lid, rc, expansion, epsilons_hard = compute_metrics(query, dataset, epsilons, k_value)
            
            # qq = np.array([query]) # just to comply with faiss API
            # distcomp = None
            # elapsed = None
            # for nprobe in range(1,1000):
            #     faiss.cvar.indexIVF_stats.reset()
            #     index.nprobe = nprobe
            #     tstart = time.time()
            #     run_dists = index.search(qq, k)[0][0]
            #     tend = time.time()
            #     elapsed = tend - tstart
            #     # if distance_metric == "angular":
            #     #     # The index returns squared euclidean distances, 
            #     #     # which we turn to angular distances in the following
            #     #     run_dists = 1 - (2 - run_dists) / 2
            #     # else:
            #     #     assert False, "fix this branch"
            #     rec = compute_recall(distances[i,:], run_dists, k)
            #     if rec >= target_recall:
            #         distcomp = faiss.cvar.indexIVF_stats.ndis + + faiss.cvar.indexIVF_stats.nq * n_list
            #         break

            # assert distcomp is not None

            row = [i, lid, rc, expansion]
            # row = [i, lid, rc, expansion, distcomp, rec, elapsed]
            row.extend(epsilons_hard)

            writer.writerow(row)