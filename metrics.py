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

def get_epsilons(queries, dataset, distance_metric):
    max_dist_arr [compute_distances(qq, None, distance_metric, dataset) for qq in queries]
    mean_max_dist = sum(max_dist_arr)/len(max_dist_arr)

    return [mean_max_dist*r for r in [0.001, 0.01, 0.05]]

def build_index(dataset, n_list, distance_metric):
    print("Building index")
    
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    print("Index built")

    return index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to the dataset file')
    parser.add_argument('--query', required=True, help='Path to the query file')
    parser.add_argument('--k', type=int, required=True, help='Number of nearest neighbors to find')
    parser.add_argument('--epsilon', type=float, nargs='+', help='List of epsilon values to use')
    parser.add_argument('--data_limit', type=int, default=1000000, help='Maximum number of data points to load from the dataset file')
    parser.add_argument('--query_limit', type=int, default=10000, help='Maximum number of query points to load from the query file')
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

    epsilons = None

    if epsilon is not None:
        epsilons = [float(x) for x in opt_params]
    else: 
        epsilons = get_epsilons(queries[:int(sample*len(queries))+1], dataset, distance_metric)

    epsilons_str = '_'.join(f'{e:.2f}' for e in epsilons)
    print (f'e-values (based on {sample} query sample): {epsilons_str}')

 
    output_file = f"res_{dataset_name}_{queryset_name}_{k_value}_{epsilons_str}"
    
    target_recall = 0.95
    n_list = 32
    # index = build_index(dataset, n_list, distance_metric)
    print("Building index")
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    print("Index built")

    flag = True
    with open(output_file+'.csv', "w", newline="") as fp:
        writer = csv.writer(fp)

        header = ["i", "lid_"+str(k_value), "rc_"+str(k_value), f"exp_{2*k_value}|{k_value}"]
        header.extend(["eps_" + f'{e:.2f}' for e in epsilons])
        writer.writerow(header)

        nqueries = queries.shape[0]
        for i in tqdm(range(nqueries)):
            query = queries[i,:]  
            q_distances = compute_distances(query, 100, distance_metric, dataset)
            lid, rc, expansion, epsilons_hard = compute_metrics(q_distances, epsilons, k_value)
            
            row = [i, lid, rc, expansion]
            row.extend(epsilons_hard)

            qq = np.array([query]) # just to comply with faiss API
            distcomp = None
            elapsed = None
            for nprobe in range(1,1000):
                faiss.cvar.indexIVF_stats.reset()
                index.nprobe = nprobe
                tstart = time.time()
                run_dists = compute_distances(query, k_value, distance_metric, index)
                tend = time.time()
                elapsed = tend - tstart
                if distances is not None:
                    q_dists =  distances[i,:]
                else:
                    q_dists = q_distances
                #debug    
                if flag:
                    print(f"run_dist: {run_dists} \n q_dist {q_dists}")
                    flag = False

                rec = compute_recall(q_dists, run_dists, k_value)
                # print(rec)
                if rec >= target_recall:
                    distcomp = faiss.cvar.indexIVF_stats.ndis + + faiss.cvar.indexIVF_stats.nq * n_list
                    break

            assert distcomp is not None
            row.extend([distcomp, rec, elapsed])

            writer.writerow(row)
