import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import csv
import faiss
import time
import read_data as rd
from threading import Lock
from cache import MEM
from utils import *
import indices
from indices import FAISS_LOCK


def compute_epsilon_hardness(distances, epsilon, k=1):
    min_dist = distances[k - 1]
    epsilon_dist = (1 + epsilon) * min_dist
    epsilon_nn = distances[distances <= epsilon_dist]
    epsilon_hardness = len(epsilon_nn) / len(distances)
    return epsilon_hardness


def compute_lid(distances, k, scale="log"):
    w = distances[min(len(distances) - 1, k)]
    half_w = 0.5 * w

    distances = distances[: k + 1]
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
        return np.log(avg_dist / distances[k])
    else:
        return avg_dist / distances[k]


def compute_expansion(distances, k, scale="log"):
    if scale == "log":
        return np.log(distances[2 * k] / distances[k])
    else:
        return distances[2 * k] / distances[k]


# def compute_metrics(distances, epsilons, k, scale="log"):

#     lid = compute_lid(distances, k-1, scale)
#     rc = compute_rc(distances, k-1, scale)
#     expansion = compute_expansion(distances, k-1, scale)

#     epsilons_hardness = [compute_epsilon_hardness(distances, e) for e in epsilons]

#     return lid, rc, expansion, epsilons_hardness


def read_sys_argv_list(start_index=4):
    if len(sys.argv) >= start_index + 1:
        return sys.argv[start_index:]
    else:
        return None


def get_epsilons(queries, dataset, distance_metric, threads=None):
    print("Compute epsilons for a given dataset and workload")
    sample = len(queries)
    max_e_arr = []

    # for qq in queries:
    for i in tqdm(range(sample)):
        dist = compute_distances(queries[i], None, distance_metric, dataset)[0]
        max_e = dist[-1] / dist[0] - 1
        max_e_arr.append(max_e)

    mean_max_e = sum(max_e_arr) / len(max_e_arr)
    print(f"mean_max_epsilon: {mean_max_e}")

    quantiles = [0.25, 0.5, 0.75]
    return [mean_max_e * r for r in quantiles], quantiles


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
    upper = min(upper, len(candidates))

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
        self.n_list = int(np.ceil(np.sqrt(dataset.shape[0])))
        self.index = indices.build_faiss_ivf(dataset, self.n_list)
        self.exact_index = exact_index
        self.recall = recall
        self.distance_metric = distance_metric

    def evaluate(self, x, k, distances=None):
        """Evaluates the empirical difficulty of the given point `x` for the given `k`.
        Returns the number of distance computations, scaled by the number of datasets.
        Optionally uses distances computed elsewhere.
        """
        start = time.time()
        if distances is None:
            distances = compute_distances(
                x, None, self.distance_metric, self.exact_index
            )[0, :]
        assert distances.shape[0] == self.index.ntotal
        elapsed_bf = time.time() - start

        def tester(nprobe):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with FAISS_LOCK:
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

        start = time.time()
        dist_frac = partition_by(list(range(1, self.n_list)), tester)

        if dist_frac is not None:
            return dist_frac
        else:
            raise Exception(
                "Could not get the desired recall, even visiting the entire dataset"
            )


class EmpiricalDifficultyHNSW(object):
    """
    Stores (and possibly caches on a file) a FAISS-HNSW index to evaluate the difficulty
    of queries, using the number of computed distances as a proxy for the difficulty.
    """

    def __init__(self, dataset, recall, exact_index, distance_metric):
        self.index = indices.build_faiss_hnsw(dataset)
        self.exact_index = exact_index
        self.recall = recall
        self.distance_metric = distance_metric

    def evaluate(self, x, k, distances=None):
        """Evaluates the empirical difficulty of the given point `x` for the given `k`.
        Returns the number of distance computations, scaled by the number of datasets.
        Optionally uses distances computed elsewhere.
        """
        start = time.time()
        if distances is None:
            distances = compute_distances(
                x, None, self.distance_metric, self.exact_index
            )[0, :]
        assert distances.shape[0] == self.index.ntotal
        elapsed_bf = time.time() - start

        def tester(efsearch):
            # we need to lock the execution because the statistics collection is
            # not thread safe, in that it uses global variables.
            with FAISS_LOCK:
                faiss.cvar.hnsw_stats.reset()
                self.index.hnsw.efSearch = efsearch
                run_dists = compute_distances(x, k, self.distance_metric, self.index)[0]
                stats = faiss.cvar.hnsw_stats
                distcomp = stats.n1 + stats.n2 + stats.n3 + stats.ndis

            rec = compute_recall(distances, run_dists, k)
            if rec >= self.recall:
                return distcomp / self.index.ntotal
            else:
                return None

        start = time.time()
        dist_frac = partition_by(list(range(1, self.index.ntotal)), tester)

        if dist_frac is not None:
            return dist_frac
        else:
            raise Exception(
                "Could not get the desired recall, even visiting the entire dataset"
            )


def metrics_csv(
    dataset_path,
    queries_path,
    output_path,
    k,
    target_recall=0.99,
    additional_header=[],
    additional_row=[],
    threads=None,
    sample=0.05,
):
    assert len(additional_header) == len(additional_row)
    dataset, distance_metric = rd.read_multiformat(dataset_path, "train")
    queries, _ = rd.read_multiformat(queries_path, "test")

    exact_index = faiss.IndexFlatL2(dataset.shape[1])
    exact_index.add(dataset)
    ivf_difficulty = EmpiricalDifficultyIVF(
        dataset,
        recall=target_recall,
        distance_metric=distance_metric,
        exact_index=exact_index,
    )

    # epsilons, eps_quantiles = get_epsilons(
    #     queries[: int(sample * len(queries)) + 1], exact_index, distance_metric
    # )

    epsilons = [.05, .1, .5, 1] # fixed epsilons 
    epsilons_str = "_".join(f"{e:.2f}" for e in epsilons)
    print(f"e-values (based on {sample} query sample): {epsilons_str}")

    def compute_row(i):
        """Computes the metrics of a single query, i.e. a single row of
        the output csv file"""
        query = queries[i, :].astype(np.float32)
        q_distances = compute_distances(query, None, distance_metric, exact_index)[0]
        lid = compute_lid(q_distances, k, "linear")
        rc = compute_rc(q_distances, k, "linear")
        expansion = compute_expansion(q_distances, k, "linear")
        empirical_difficulty = ivf_difficulty.evaluate(query, k, q_distances)
        epsilons_hard = [compute_epsilon_hardness(q_distances, e) for e in epsilons]

        row = [i, lid, rc, expansion, empirical_difficulty]
        row.extend(additional_row)
        row.extend(epsilons_hard)
        return row

    if threads is None:
        import os

        threads = os.cpu_count()
    nqueries = queries.shape[0]
    with ThreadPoolExecutor(threads) as pool:
        tasks = [pool.submit(compute_row, i) for i in range(nqueries)]
        rows = []
        for task in tqdm(as_completed(tasks), total=len(tasks)):
            row = task.result()
            rows.append(row)
        rows.sort()

    with open(output_path, "w", newline="") as fp:
        writer = csv.writer(fp)

        header = ["i", "lid_" + str(k), "rc_" + str(k), f"exp_{2*k}|{k}"]

        header.extend(["distcomp"])
        header.extend(additional_header)
        # header.extend([f"eps_q{e}" for e in eps_quantiles])
        header.extend([f"eps_{e}" for e in epsilons])
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)
