import pandas as pd
from snakemake.utils import Paramspace
from itertools import product
import pickle
import hashlib


def get_data_path(base_dir):
    import os

    def inner(wildcards):
        print(wildcards)
        hdf5_datasets = {
            "fashion-mnist-784-euclidean",
            "glove-100-angular",
            "nytimes-256-angular",
        }
        if wildcards.dataset in hdf5_datasets:
            ext = "hdf5"
        else:
            ext = "bin"
        return os.path.join(base_dir, f"{wildcards.dataset}.{ext}")

    return inner


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


def workloads():
    """Builds the configuration of all workloads that we use in our experiments.

    Returns a pair, where the first element is the paramspace to be used
    in the workflow, and the second is a dictionary where the keys are
    workload identifiers (used in the paths as part of the wildcard)
    and the values are dictionaries holding the workload configuration.
    """
    import sys
    import os

    # Make the read_data module visible by adding its directory to
    # the search path.
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    import read_data as rd

    configs = []
    workloads_dict = dict()

    # TODO: add builtin queries, including the ones that are not generated

    datasets = [
        "fashion_mnist-euclidean-784-60K",
        # "glove-angular-32-1183514",
        # "glove-angular-104-1183514",
        # "nytimes-angular-256-289761",
    ]

    # Simulated annealing synthetic queries
    workload_type = "synthetic-simulated-annealing"
    faiss_ivf_difficulties = [(t - 0.01, t + 0.01) for t in [0.05, 0.4]]
    target_difficulty = {
        "faiss_ivf": {
            "fashion_mnist-euclidean-784-60K": faiss_ivf_difficulties,
            "glove-angular-104-1183514": faiss_ivf_difficulties,
            "glove-angular-32-1183514": faiss_ivf_difficulties,
            "sald-128-1000000": faiss_ivf_difficulties,
            "nytimes-angular-256-289761": faiss_ivf_difficulties,
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

    num_queries = [3]
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
                conf = {
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

    # Gaussian noise workloads
    workload_type = "synthetic-gaussian-noise"
    scales = [0.1, 1.0, 10.0]
    num_queries = [300]
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

    return WorkloadPatterns(configs, workloads_dict)


def setup_param_space():
    datasets = [
        "fashion-mnist-784-euclidean",
        "glove-100-angular",
        "sald-128-1000000",
        "nytimes-256-angular",
    ]
    queries = {
        "sald": ["sald", "sald-noise-10", "sald-noise-30", "sald-noise-50"],
        "glove-100-bin": [
            "glove-noise-0",
            "glove-noise-10",
            "glove-noise-30",
            "glove-noise-50",
        ],
    }
    target_difficulty = {
        "faiss_ivf": {
            "fashion-mnist-784-euclidean": [
                (t - 0.01, t + 0.01) for t in [0.05, 0.1, 0.3]
            ],
            "glove-100-angular": [(t - 0.01, t + 0.01) for t in [0.05, 0.1, 0.3]],
            "sald-128-1000000": [(t - 0.01, t + 0.01) for t in [0.05, 0.1, 0.3]],
            "nytimes-256-angular": [(t - 0.01, t + 0.01) for t in [0.05, 0.1, 0.3]],
        },
        "rc": {
            "fashion-mnist-784-euclidean": [],
            "glove-100-angular": [],
            "sald-128-1000000": [],
            "nytimes-256-angular": [],
        },
    }
    num_queries = {
        "fashion-mnist-784-euclidean": [5],
        "glove-100-angular": [5],
        "sald-128-1000000": [5],
        "nytimes-256-angular": [5],
    }
    scale = {
        "fashion-mnist-784-euclidean": [10],
        "glove-100-angular": [0.1],
        "sald-128-1000000": [10],
        "nytimes-256-angular": [0.1],
    }
    initial_temperature = {
        "fashion-mnist-784-euclidean": [10],
        "glove-100-angular": [1],
        "sald-128-1000000": [10],
        "nytimes-256-angular": [1],
    }
    ks = [10]

    configs = []

    for dataset in datasets:
        for nq in num_queries[dataset]:
            for k in ks:
                for tf in target_difficulty.keys():
                    if dataset in target_difficulty[tf]:
                        for lower, upper in target_difficulty[tf][dataset]:
                            for s in scale[dataset]:
                                for temp in initial_temperature[dataset]:
                                    configs.append(
                                        {
                                            "dataset": dataset,
                                            "k": k,
                                            "num_queries": nq,
                                            "difficulty": tf,
                                            "target_lower": lower,
                                            "target_upper": upper,
                                            "scale": s,
                                            "initial_temperature": temp,
                                        }
                                    )
    return Paramspace(pd.DataFrame(configs))


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
        if os.path.isfile(fname) and wkey.startswith(query):
            print(fname)
            conf = WORKLOADS.config_for(wkey)
            pprint.pprint(conf)
