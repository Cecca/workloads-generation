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
            difficulty = config["target_class"]
            return f"Annealing({metric}, {difficulty})"
        elif workload_type == "synthetic-sgd":
            metric = config["difficulty"]
            difficulty = config["target_class"]
            return f"SGD({metric}, {difficulty})"
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

    def config_from_path(self, path):
        """Extracts the workload key from the given path, and decodes it into the corresponding dictionary"""
        import re
        match = re.findall("workload_key~([a-z0-9]+)/", path)
        if len(match) == 0:
            raise ValueError("the path does not contain a workload key")
        return self.config_for(match[0])

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
            "fashion_mnist-angular-784-60K",
            "queries_fashion_mnist-angular-784-10000",
        ),
        ("glove-angular-104-1183514", "queries_glove-angular-104-10000"),
        ("nytimes-angular-256-289761", "queries_nytimes-angular-256-9991"),
        ("sald-128-100m", "sald-128-1k"),
        # TODO: add synthetic ?
        ("rw-256-100m", "rw-256-1k"),
        ("astro-256-100m", "astro-256-1k"),
        ("deep1b-96-100m", "deep1b-96-1k"),
        ("seismic-256-100m", "seismic-256-1k"),
    ]
    k_values = [10, 1]
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


def _sgd_workloads():
    import sys
    import os

    # Make the read_data module visible by adding its directory to
    # the search path.
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    import read_data as rd

    configs = []
    workloads_dict = dict()

    datasets = [
        "fashion_mnist-angular-784-60K",
        "glove-angular-104-1183514",
        "nytimes-angular-256-289761",
        "sald-128-100m",
        "astro-256-100m",
        "deep1b-96-100m",
        "seismic-256-100m",
        "rw-256-100m",
    ]

    # Stochastic gradient descent
    workload_type = "synthetic-sgd"
    target_difficulties = ["easy", "medium", "hard", "hard+"]
    target_metrics = ["rc"]
    num_queries = [10]
    k_values = [10, 1]

    for dataset, k, nq, target_difficulty, target_metric in product(
        datasets, k_values, num_queries, target_difficulties, target_metrics
    ):
        assert target_metric == "rc", "only RC is supported"
        conf = OrderedDict(  # to maintain a consistent hash value
            {
                "workload_type": workload_type,
                "dataset": dataset,
                "k": k,
                "num_queries": nq,
                "difficulty": target_metric,
                "target_class": target_difficulty,
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
        "fashion_mnist-angular-784-60K",
        # "glove-angular-32-1183514",
        # "glove-angular-104-1183514",
        # "nytimes-angular-256-289761",
        #"sald-128-1000000",
        # "fashion_mnist-angular-784-60K",
        "glove-angular-104-1183514",
        "nytimes-angular-256-289761",
        "sald-128-100m",
        "astro-256-100m",
        "deep1b-96-100m",
        "seismic-256-100m",
        "rw-256-100m",
    ]

    # Simulated annealing synthetic queries
    workload_type = "synthetic-simulated-annealing"
    target_difficulties = ["easy", "medium", "hard"]
    target_metrics = ["rc"]
    num_queries = [10]
    k_values = [10, 1]

    for dataset, k, nq, target_difficulty, target_metric in product(
        datasets, k_values, num_queries, target_difficulties, target_metrics
    ):
        conf = OrderedDict(  # to maintain a consistent hash value
            {
                "workload_type": workload_type,
                "dataset": dataset,
                "k": k,
                "num_queries": nq,
                "difficulty": target_metric,
                "target_class": target_difficulty,
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
        "fashion_mnist-angular-784-60K",
        "glove-angular-104-1183514",
        "nytimes-angular-256-289761",
        "sald-128-100m",
        "astro-256-100m",
        "deep1b-96-100m",
        "seismic-256-100m",
        "rw-256-100m",
    ]
    # scales = [0.1, 1.0, 10.0]
    scales = ["easy", "medium", "hard"]
    num_queries = [100]
    k_values = [10, 1]

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
                OrderedDict(
                    {
                        "dataset": dataset,
                        "workload_key": key,
                        "workload_file": workload_fname,
                    }
                )
            )

    return configs, workloads_dict


def workloads():
    """Builds the configuration of all workloads that we use in our experiments.

    Returns the WorkloadPatterns instance containing all the workloads we want to run.
    """

    configs = []
    workloads_dict = dict()

    sgd_configs = _annealing_workloads()
    configs.extend(sgd_configs[0])
    workloads_dict.update(sgd_configs[1])

    sgd_configs = _sgd_workloads()
    configs.extend(sgd_configs[0])
    workloads_dict.update(sgd_configs[1])

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
