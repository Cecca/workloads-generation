import pandas as pd
from snakemake.utils import Paramspace
from itertools import product
import pickle
import hashlib
from collections import OrderedDict


class WorkloadPatterns:
    def __init__(self, configs, workloads_dict):
        self._configs = configs
        self._workloads_dict = workloads_dict
        self._wildcard_pattern = (
            "dataset~{dataset}/workload_key~{workload_key}/{workload_file}"
        )
        self._patterns = [self.wildcard_pattern.format(**c) for c in self._configs]

    def description_for(self, key):
        config = self.config_for(key)
        workload_type = config["workload_type"]
        if workload_type == "synthetic-simulated-annealing":
            metric = config["difficulty"]
            low = config["target_low"]
            high = config["target_high"]
            return f"Annealing({metric}, {low:.2f}, {high:.2f})"
        elif workload_type == "synthetic-gaussian-noise":
            scale = config["scale"]
            return f"GaussianNoise({scale})"
        elif workload_type == "file-based":
            fname = config["queries_filename"]
            return f"File({fname})"
        else:
            raise KeyError(f"unknown workload type {workload_type}")

    def config_for(self, key):
        return self._workloads_dict[key]

    @property
    def wildcard_pattern(self):
        return self._wildcard_pattern

    @property
    def instance_patterns(self):
        return self._patterns

    @property
    def workloads_dict(self):
        return self._workloads_dict


def _file_based_workloads():
    """List of configurations using already existing files containing queries."""

    configs = []
    workloads_dict = dict()

    workload_type = "file-based"
    dataset_query_pairs = [
        (
            "fashion_mnist-euclidean-784-60K",
            "queries_fashion_mnist-euclidean-784-10000",
        ),
        ("glove-angular-104-1183514", "queries_glove-angular-104-10000"),
        ("nytimes-angular-256-289761", "queries_nytimes-angular-256-9991"),
        ("sald-128-100m", "sald-128-1k"),
        # TODO: add synthetic ?
        ("astro-256-100m", "astro-256-1k"),
        ("deep1b-96-100m", "deep1b-96-1k"),
        ("seismic-256-100m", "seismic-256-1k"),
    ]
    k_values = [10]
    for (dataset, queryset), k in product(dataset_query_pairs, k_values):
        conf = {
            "workload_type": workload_type,
            "dataset": dataset,
            "k": k,
            "queries_filename": queryset,
        }
        key = hashlib.sha256(pickle.dumps(conf)).hexdigest()
        workloads_dict[key] = conf
        configs.append(
            {
                "dataset": dataset,
                "workload_key": key,
                "workload_file": queryset,
            }
        )

    return configs, workloads_dict


def _annealing_workloads():
    import sys
    import os

    # Make the read_data module visible by adding its directory to
    # the search path.
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    import read_data as rd

    configs = []
    workloads_dict = dict()

    datasets = [
        "fashion_mnist-euclidean-784-60K",
        # "glove-angular-32-1183514",
        "glove-angular-104-1183514",
        # "nytimes-angular-256-289761",
        # "sald-128-1000000",
    ]

    # Simulated annealing synthetic queries
    workload_type = "synthetic-simulated-annealing"
    faiss_ivf_difficulties = [(t - 0.01, t + 0.01) for t in [0.05, 0.2]]
    target_difficulty = {
        "faiss_ivf": {
            "fashion_mnist-euclidean-784-60K": faiss_ivf_difficulties,
            "glove-angular-104-1183514": faiss_ivf_difficulties,
            "glove-angular-32-1183514": faiss_ivf_difficulties,
            # "sald-128-1000000": faiss_ivf_difficulties,
            "nytimes-angular-256-289761": faiss_ivf_difficulties,
        },
        "rc": {
            "fashion_mnist-euclidean-784-60K": [(2.1, 1.9), (1.05, 1.03), (1.01, 1.0)],
            "glove-angular-104-1183514": [(2.1, 1.9), (1.8, 1.7), (1.5, 1.3)],
            "nytimes-angular-256-289761": [(2.1, 1.9)],
            "sald-128-1000000": [(100, 1)],
        },
    }
    scales = {
        "fashion_mnist-euclidean-784-60K": [10],
        "glove-angular-32-1183514": [0.1],
        "glove-angular-104-1183514": [0.1],
        "sald-128-1000000": [10],
        "nytimes-angular-256-289761": [0.1],
    }
    initial_temperature = {
        "fashion_mnist-euclidean-784-60K": [10],
        "glove-angular-32-1183514": [1],
        "glove-angular-104-1183514": [1],
        "sald-128-1000000": [10],
        "nytimes-angular-256-289761": [1],
    }

    num_queries = [10]
    k_values = [10]

    for dataset, k, nq in product(datasets, k_values, num_queries):
        for tf in target_difficulty.keys():
            if dataset not in target_difficulty[tf]:
                continue
            for scale, temp, (lower, upper) in product(
                scales[dataset],
                initial_temperature[dataset],
                target_difficulty[tf][dataset],
            ):
                conf = OrderedDict(  # to maintain a consistent hash value
                    {
                        "workload_type": workload_type,
                        "dataset": dataset,
                        "k": k,
                        "num_queries": nq,
                        "difficulty": tf,
                        "target_low": lower,
                        "target_high": upper,
                        "scale": scale,
                        "initial_temperature": temp,
                    }
                )
                key = hashlib.sha256(pickle.dumps(conf)).hexdigest()
                workloads_dict[key] = conf
                dname, _, features, distance_metric = rd.parse_filename(dataset)
                workload_fname = f"{dname}-{distance_metric}-{features}-{nq}.bin"
                configs.append(
                    {
                        "dataset": dataset,
                        "workload_key": key,
                        "workload_file": workload_fname,
                    }
                )

    return configs, workloads_dict


def _gaussian_noise_workloads():
    import sys
    import os

    # Make the read_data module visible by adding its directory to
    # the search path.
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    import read_data as rd

    configs = []
    workloads_dict = dict()

    workload_type = "synthetic-gaussian-noise"

    datasets = [
        "fashion_mnist-euclidean-784-60K",
        # "glove-angular-32-1183514",
        "glove-angular-104-1183514",
        "nytimes-angular-256-289761",
        "sald-128-100m",
        "astro-256-100m",
        "deep1b-96-100m",
        "seismic-256-100m",
    ]
    scales = [0.1, 1.0, 10.0]
    num_queries = [30]
    k_values = [10]

    for dataset, k, nq in product(datasets, k_values, num_queries):
        for scale in scales:
            conf = {
                "workload_type": workload_type,
                "dataset": dataset,
                "k": k,
                "num_queries": nq,
                "scale": scale,
            }
            key = hashlib.sha256(pickle.dumps(conf)).hexdigest()
            workloads_dict[key] = conf
            dname, _, features, distance_metric = rd.parse_filename(dataset)
            workload_fname = f"{dname}-{distance_metric}-{features}-{nq}.bin"
            configs.append(
                {
                    "dataset": dataset,
                    "workload_key": key,
                    "workload_file": workload_fname,
                }
            )

    return configs, workloads_dict


def workloads():
    """Builds the configuration of all workloads that we use in our experiments.

    Returns the WorkloadPatterns instance containing all the workloads we want to run.
    """

    configs = []
    workloads_dict = dict()

    annealing_configs = _annealing_workloads()
    configs.extend(annealing_configs[0])
    workloads_dict.update(annealing_configs[1])

    noise_configs = _gaussian_noise_workloads()
    configs.extend(noise_configs[0])
    workloads_dict.update(noise_configs[1])

    file_configs = _file_based_workloads()
    configs.extend(file_configs[0])
    workloads_dict.update(file_configs[1])

    return WorkloadPatterns(configs, workloads_dict)


if __name__ == "__main__":
    import sys
    import os
    import re
    import pprint

    DATA_DIR = os.environ.get("WORKGEN_DATA_DIR", ".data")
    GENERATED_DIR = os.path.join(DATA_DIR, "generated")

    WORKLOADS = workloads()

    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = ""

    for pat in WORKLOADS.instance_patterns:
        fname = os.path.join(GENERATED_DIR, pat + ".bin")
        wkey = re.findall("workload_key~([a-z0-9]+)", pat)[0]
        if not wkey.startswith(query):
            continue
        if os.path.isfile(fname):
            print(fname)
        else:
            print("MISSING: ", fname)
        conf = WORKLOADS.config_for(wkey)
        pprint.pprint(conf)
