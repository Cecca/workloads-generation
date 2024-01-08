import pandas as pd
from snakemake.utils import Paramspace


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


def setup_param_space():
    datasets = [
        # "fashion-mnist-784-euclidean",
        # "glove-100-angular",
        "sald-128-1000000",
        # "nytimes-256-angular",
    ]
    queries = {
        "sald": [
        "sald",
        "sald-noise-10",
        "sald-noise-30",
        "sald-noise-50"
        ],
        "glove-100-bin": [
        "glove-noise-0",
        "glove-noise-10",
        "glove-noise-30",
        "glove-noise-50"
        ]
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
    print(setup_param_space())
