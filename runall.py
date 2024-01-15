from joblib import parallel
import pandas as pd
import itertools
import logging
import os
from pprint import pprint

import metrics
import generate
import read_data
from caching import MEM

logging.basicConfig(level=logging.INFO)

DATA_DIR = os.environ.get(
    "WORKGEN_DATA_DIR", ".data"
)  # on nefeli: export WORKGEN_DATA_DIR=/mnt/hddhelp/workgen_data/
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
RES_DIR = os.environ.get(
    "WORKGEN_RES_DIR", "results"
)  # on nefeli: export WORKGEN_RES_DIR=/mnt/hddhelp/workgen_results/

DATASETS = [
    "fashion-mnist-784-euclidean",
    "glove-100-angular",
    "nytimes-256-angular",
    "sald-128-1000000",
]


def get_data_path(dataset_name):
    hdf5_datasets = {
        "fashion-mnist-784-euclidean",
        "glove-100-angular",
        "nytimes-256-angular",
    }
    if dataset_name in hdf5_datasets:
        ext = "hdf5"
    else:
        ext = "bin"
    return os.path.join(DATA_DIR, f"{dataset_name}.{ext}")


@MEM.cache
def build_synthetic_workloads(datasets, k_values=[10], num_queries=[5]):
    """
    Generate synthetic workloads using simulated annealing.
    Further parameters are specified in the body of this function.

    Calls to the annealing process are cached using joblib, as wrapped
    in our module `caching`.

    Parameters
    ----------
    datasets : List[str]
        the datasets for which to generate the workloads
    k_values : int
        the k for the queries
    num_queries : int
        the number of queries to generate for each workload

    Returns
    -------
    A list of dictionaries with a key-value pair for each configuration, and associated to the
    key `"queries"` the 2-dimensional numpy array of the queries.
    """
    target_difficulty = {
        "faiss_ivf": {(t - 0.01, t + 0.01) for t in [0.1, 0.2, 0.3]},
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

    threads = os.cpu_count()

    results = []

    for dataset_name, k, n_queries, difficulty_type in itertools.product(
        datasets, k_values, num_queries, target_difficulty.keys()
    ):
        for (lower, upper), s, temp in itertools.product(
            target_difficulty[difficulty_type],
            scale[dataset_name],
            initial_temperature[dataset_name],
        ):
            res = {
                "dataset": dataset_name,
                "k": k,
                "num_queries": n_queries,
                "difficulty": difficulty_type,
                "target_lower": lower,
                "target_upper": upper,
                "scale": s,
                "initial_temperature": temp,
            }
            pprint(res)
            dataset, distance_metric = read_data.read_multiformat(
                get_data_path(dataset_name), "train"
            )
            print("loaded dataset with shape", dataset.shape)

            res["queries"] = generate.generate_queries_annealing(
                dataset,
                distance_metric,
                k,
                difficulty_type,
                lower,
                upper,
                n_queries,
                scale=s,
                max_steps=10000,
                initial_temperature=temp,
                seed=1234,
                threads=threads,
            )
            results.append(res)

    return results


@MEM.cache
def compute_metrics(workloads):
    alldf = []
    for workload in workloads:
        dataset, distance_metric = read_data.read_multiformat(
            get_data_path(workload["dataset"]), "train"
        )
        df = metrics.metrics_dataframe(
            dataset,
            workload["queries"],
            distance_metric,
            workload["k"],
            additional_header=[
                "dataset",
                "k",
                "difficulty_type",
                "target_lower",
                "target_upper",
            ],
            additional_row=[
                workload["dataset"],
                workload["k"],
                workload["difficulty"],
                workload["target_lower"],
                workload["target_upper"],
            ],
        )
        alldf.append(df)
    return pd.concat(alldf, ignore_index=True)


def plot_metrics(metrics, output):
    import seaborn.objects as so
    import numpy as np

    datasets = metrics["dataset"].drop_duplicates()

    def normalize(column, how="minmax"):
        if not np.issubdtype(column.dtype, np.number) or column.name == "distcomp":
            return column
        elif how == "minmax":
            return (column - column.min()) / (column.max() - column.min())
        elif how == "none":
            return column
        else:
            return (column - column.mean()) / column.std()

    normalized = []

    for dataset in datasets:
        dmetrics = metrics[metrics["dataset"] == dataset].copy()
        dmetrics["target"] = dmetrics[["target_lower", "target_upper"]].apply(
            lambda row: f"{row[0]}--{row[1]}", axis=1
        )
        dmetrics.reset_index(inplace=True)
        dmetrics = dmetrics.apply(normalize)
        normalized.append(dmetrics)

    metrics = pd.concat(normalized)

    numerics = [
        "distcomp",
        "lid_10",
        "rc_10",
        "exp_20|10",
        "eps_q25",
        "eps_q50",
        "eps_q75",
    ]
    melted = pd.melt(
        metrics, ["index", "dataset", "difficulty_type", "target"], numerics
    )

    (
        so.Plot(melted, y="value", x="variable", color="target", group="index")
        .facet(row="dataset", col="difficulty_type")
        .add(so.Dot())
        .add(so.Line())
        .layout(size=(10, 10))
        .save(output)
    )


def main():
    synthetic = build_synthetic_workloads(DATASETS)
    metrics = compute_metrics(synthetic)
    plot_metrics(metrics, output="results/plots/synthetic-difficulty.png")


main()
