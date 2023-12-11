import sys
import argparse

import numpy as np
from tqdm import tqdm
import csv
import faiss
import time
import read_data as rd
from utils import *

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

def compute_metrics(distances, epsilons, k, scale="log"):
   
    lid = compute_lid(distances, k-1, scale)
    rc = compute_rc(distances, k-1, scale)
    expansion = compute_expansion(distances, k-1, scale)

    epsilons_hardness = [compute_epsilon_hardness(distances, e) for e in epsilons]

    return lid, rc, expansion, epsilons_hardness


def read_sys_argv_list(start_index=4):
    if len(sys.argv) >= start_index + 1:
        return sys.argv[start_index:]
    else:
        return None

def get_epsilons(queries, dataset, distance_metric):
    # max_dist_arr = [compute_distances(qq, None, distance_metric, dataset)[-1] for qq in queries]
    # # print(max_dist_arr)
    # mean_max_dist = sum(max_dist_arr)/len(max_dist_arr)

    max_e_arr = []
    for qq in queries:
        dist = compute_distances(qq, None, distance_metric, dataset)
        max_e = dist[-1]/dist[0]-1
        max_e_arr.append(max_e)
    mean_max_e = sum(max_e_arr)/len(max_e_arr)
    print(mean_max_e)

    # return [mean_max_dist*r for r in [0.001, 0.01, 0.05]]
    return [mean_max_e*r for r in[0.25, 0.5, 0.75]]

def build_index(dataset, n_list, distance_metric):
    print("Building index")
    
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    print("Index built")

    return index


def partition_by(candidates, fun):
    # first do an exponential search
    upper = 0
    lower = 0
    cur_res = None
    while upper < len(candidates):
        res = fun(candidates[upper])
        if res is not None:
            cur_res = res
            break
        lower = upper
        upper = upper * 2 if upper > 0 else 1

    # now we know that the predicate is satisfied between prev_ids (where it 
    # is not satisfied) and cur_idx (where it is satisfied). So we do a binary search between the two
    while lower < upper:
        mid = (lower + upper) // 2
        mid_res = fun(candidates[mid])
        if mid_res is not None:
            cur_res = mid_res
            upper = mid
        else:
            lower = mid + 1

    return cur_res


class EmpiricalDifficultyIVF(object):
    """
    Stores (and possibly caches on a file) a FAISS-IVF index to evaluate the difficulty
    of queries, using the number of computed distances as a proxy for the difficulty.
    """

    def __init__(self, dataset, recall, exact_index, distance_metric):
        import hashlib
        import os
        import logging
        from threading import Lock

        self.n_list = int(np.ceil(np.sqrt(dataset.shape[0])))

        # we cache the index to a finle, whose name depends on the contents
        # of the dataset and on the n_list parameter
        sha = hashlib.new("sha256")
        sha.update(dataset.tobytes())
        sha = sha.hexdigest()
        fname = f".index-cache/faiss-ivf-{self.n_list}-{sha}.bin"

        if os.path.isfile(fname):
            logging.info("reading index from file")
            index = faiss.read_index(faiss.FileIOReader(fname))
        else:
            logging.info("Computing index")
            if not os.path.isdir(".index-cache"):
                os.mkdir(".index-cache")
            quantizer = faiss.IndexFlatL2(dataset.shape[1])
            index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
            index.train(dataset)
            index.add(dataset)
            faiss.write_index(index, faiss.FileIOWriter(fname))

        self.index = index
        self.exact_index = exact_index
        self.lock = Lock()
        self.recall = recall
        self.distance_metric = distance_metric
            
            
    def evaluate(self, x, k):
        """Evaluates the empirical difficulty of the given point `x` for the given `k`.
        Returns the number of distance computations, scaled by the number of datasets."""
        distances = compute_distances(x, None, self.distance_metric, self.exact_index)[0, :]

        def tester(nprobe):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with self.lock:
                faiss.cvar.indexIVF_stats.reset()
                self.index.nprobe = nprobe
                run_dists = compute_distances(x, k, self.distance_metric, self.index)[0]
                distcomp = (
                    faiss.cvar.indexIVF_stats.ndis
                    + faiss.cvar.indexIVF_stats.nq * self.n_list
                )

            rec = compute_recall(distances, run_dists, k)
            if rec >= self.recall:
                return distcomp / self.index.ntotal
            else:
                return None

        dist_frac = partition_by(list(range(1, self.n_list)), tester)
        if dist_frac is not None:
            return dist_frac
        else:
            raise Exception("Could not get the desired recall, even visiting the entire dataset")
        


def metrics_csv(dataset_path, queries_path, output_path, k, target_recall=0.99, additional_header=[], additional_row=[]):
    assert len( additional_header ) == len(additional_row)
    dataset, distance_metric = rd.read_hdf5(dataset_path, "train")
    n = dataset.shape[0]
    queries, _ = rd.read_hdf5(queries_path, "test")

    n_list = int(np.ceil(np.sqrt(n)))
    index = build_index(dataset, n_list, distance_metric)

    with open(output_path, "w", newline="") as fp:
        writer = csv.writer(fp)

        header = ["i", "lid_"+str(k), "rc_"+str(k), f"exp_{2*k}|{k}"]
        # header.extend(["eps_" + f'{e:.2f}' for e in epsilons])
        header.extend(["distcomp", "recall", "elapsed"])
        header.extend(additional_header)
        writer.writerow(header)

        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:].astype(np.float32)
            q_distances = compute_distances(query, None, distance_metric, dataset)[0]
            # lid, rc, expansion, epsilons_hard = compute_metrics(q_distances, epsilons, k)
            lid = compute_lid(q_distances, k, "linear")
            rc = compute_rc(q_distances, k, "linear")
            expansion = compute_expansion(q_distances, k, "linear")
            
            row = [i, lid, rc, expansion]
            # row.extend(epsilons_hard)

            # qq = np.array([query]) # just to comply with faiss API
            distcomp = None
            elapsed = None
            for nprobe in range(1, n_list):
                faiss.cvar.indexIVF_stats.reset()
                index.nprobe = nprobe
                tstart = time.time()
                run_dists = compute_distances(query, k, distance_metric, index)[0]
                tend = time.time()
                elapsed = tend - tstart
                q_dists = q_distances

                rec = compute_recall(q_dists, run_dists, k)
                if rec >= target_recall:
                    distcomp = faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * n_list
                    break

            assert distcomp is not None
            row.extend([distcomp / n, rec, elapsed])
            row.extend(additional_row)

            writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to the dataset file')
    parser.add_argument('--query', required=True, help='Path to the query file')
    parser.add_argument('--k', type=int, required=True, help='Number of nearest neighbors to find')
    parser.add_argument('--epsilon', type=float, nargs='+', help='List of epsilon values to use')
    parser.add_argument('--data_limit', type=int,  help='Maximum number of data points to load from the dataset file')
    parser.add_argument('--query_limit', type=int,  help='Maximum number of query points to load from the query file')
    parser.add_argument('--sample', type=float, default=0.05, help='Fraction of query set used to calculate epsilons')

    args = parser.parse_args()

    # Extract the named parameters
    dataset_name = args.dataset
    queryset_name = args.query
    k_value = args.k
    epsilon = args.epsilon
    data_limit = args.data_limit
    query_limit = args.query_limit
    sample = args.sample

    # Check if the required parameters are present
    if dataset_name is None or queryset_name is None or k_value is None:
        print("Usage: python metrics.py dataset_file query_file k [optional:epsilon as list with spaces] [optional:data_limit] [optional:query_limit]")
        sys.exit(1)
    
    dataset, queries, distances, distance_metric = rd.read_data(dataset_name, queryset_name, data_limit, query_limit)
    print("Loaded dataset with {} points, and queryset with {} queries".format(dataset.shape, queries.shape))

    # epsilons = None
    #
    # if epsilon is not None:
    #     epsilons = [float(x) for x in opt_params]
    # else:
    #     epsilons = get_epsilons(queries[:int(sample*len(queries))+1], dataset, distance_metric)
    #
    # print(epsilons)
    # epsilons_str = '_'.join(f'{e:.2f}' for e in epsilons)
    # print (f'e-values (based on {sample} query sample): {epsilons_str}')

    epsilons = None

    if epsilon is not None:
        epsilons = [float(x) for x in opt_params]
    else: 
        epsilons = get_epsilons(queries[:int(sample*len(queries))+1], dataset, distance_metric)

    epsilons_str = '_'.join(f'{e:.2f}' for e in epsilons)
    print (f'e-values (based on {sample} query sample): {epsilons_str}')

 
    #output_file = f"res_{dataset_name}_{queryset_name}_{k_value}"
    output_file = f"res_{dataset_name}_{queryset_name}_{k_value}_{epsilons_str}"
    
    target_recall = 0.95
    n_list = 32
    index = build_index(dataset, n_list, distance_metric)

    flag = True
    with open(output_file+'.csv', "w", newline="") as fp:
        writer = csv.writer(fp)

        header = ["i", "lid_"+str(k_value), "rc_"+str(k_value), f"exp_{2*k_value}|{k_value}"]
        header.extend(["eps_" + f'{e:.2f}' for e in epsilons])
        header.extend(['distcomp','rec', 'elapsed'])
        writer.writerow(header)

        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:].astype(np.float32)
            #q_distances = compute_distances(query, None, distance_metric, dataset)[0]
            # lid, rc, expansion, epsilons_hard = compute_metrics(q_distances, epsilons, k_value)
            # lid = compute_lid(q_distances, k_value, "linear")
            # rc = compute_rc(q_distances, k_value, "linear")
            # expansion = compute_expansion(q_distances, k_value, "linear")
        
            q_distances = compute_distances(query, None, distance_metric, dataset)
            lid, rc, expansion, epsilons_hard = compute_metrics(q_distances, epsilons, k_value)
            
            row = [i, lid, rc, expansion]
            row.extend(epsilons_hard)

            # qq = np.array([query]) # just to comply with faiss API
            distcomp = None
            elapsed = None
            for nprobe in range(1,1000):
                faiss.cvar.indexIVF_stats.reset()
                index.nprobe = nprobe
                tstart = time.time()
                #run_dists = compute_distances(query, k_value, distance_metric, index)[0]
                run_dists = compute_distances(query, k_value, distance_metric, index)
                tend = time.time()
                elapsed = tend - tstart
                if distances is not None:
                    q_dists =  distances[i,:]
                else:
                    q_dists = q_distances[:100]
                #debug    
                if flag:
                    print(f"run_dist: {run_dists} \n q_dist {q_dists}")
                    flag = False

                rec = compute_recall(q_dists, run_dists, k_value)
                if rec >= target_recall:
                    distcomp = faiss.cvar.indexIVF_stats.ndis + + faiss.cvar.indexIVF_stats.nq * n_list
                    break

            assert distcomp is not None
            row.extend([distcomp, rec, elapsed])

            writer.writerow(row)
