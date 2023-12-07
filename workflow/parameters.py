import pandas as pd
from snakemake.utils import Paramspace

def setup_param_space():
    datasets = [
        "fashion-mnist-784-euclidean",
        "glove-100-angular",
    ]
    target_difficulty = {
        "faiss_ivf": {
            "fashion-mnist-784-euclidean": [
                (0.01, 1.0),
                (0.1, 1.0),
                (0.2, 1.0),
                (0.3, 1.0),
                (0.4, 1.0),
            ],
            "glove-100-angular": [
                (0.01, 1.0),
                (0.1, 1.0),
                (0.2, 1.0),
                (0.3, 1.0),
                (0.4, 1.0),
            ]
        },
        "rc": {
            "fashion-mnist-784-euclidean": [
                # (2.05, 1.95),
                # (1.55, 1.45),
                # (1.25, 1.15)
                # (1.05, 1.01),
            ],
            "glove-100-angular": [
                # (2.05, 1.95)
                #(1.55, 1.45)
            ]
        }
    }
    num_queries = {
        "fashion-mnist-784-euclidean": [5],
        "glove-100-angular": [5]
    }
    scale = {
        "fashion-mnist-784-euclidean": [10],
        "glove-100-angular": [0.1] 
    }
    initial_temperature = {
        "fashion-mnist-784-euclidean": [10],
        "glove-100-angular": [1]
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
                                    configs.append({
                                        "dataset": dataset,
                                        "k": k,
                                        "num_queries": nq,
                                        "difficulty": tf,
                                        "target_lower": lower,
                                        "target_upper": upper,
                                        "scale": s,
                                        "initial_temperature": temp
                                    })
    return Paramspace(pd.DataFrame(configs))



if __name__ == "__main__":
    print(setup_param_space())


