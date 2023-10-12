"""
This module collects approaches to generate workloads for a given dataset.
"""

import numpy as np
import dimensionality_measures as dm


def gen_random_walk(dataset, k, metric, target, probes=4, scale=1.0, max_steps=100, seed=1234, startquery=None):
    dim = dataset.shape[1]
    def gen_query_angular(base, gen):
        if base is None:
            query = gen.normal(size=dim)
        else:
            rotation = gen.normal(scale=scale, size=(dim, dim))
            query = np.dot(base, rotation)
        query /= np.linalg.norm(query)
        query_metric = dm.compute(query, dataset, metric, k, distance_metric="angular")
        # if base is not None:
        #     dist = np.linalg.norm(query - base)
        #     print(f"   distance from previous {dist}, metric {query_metric}")
        return query_metric, query


    gen = np.random.default_rng(seed)
    if startquery is None:
        query_metric, query = gen_query_angular(None, gen)
    else:
        query_metric = dm.compute(startquery, dataset, metric, k, distance_metric="angular")
        query = startquery
    print("initial query", query)

    for step in range(max_steps):
        print("step", step, "metric", query_metric)
        candidates = [gen_query_angular(query, gen) for _ in range(probes)]
        candidates = sorted(candidates, reverse=False)
        for candidate_metric, candidate in candidates:
            if candidate_metric >= target: # TODO: use the right direction of the comparison
                return candidate_metric, candidate

        # if we did not return, pick the candidate with the 
        # best metric, if it is better than the current best query
        candidate_metric, candidate = candidates[-1]
        if candidate_metric > query_metric:
            query = candidate
            query_metric = candidate_metric
    print("Failed to find candidate query")


if __name__ == "__main__":
    import bench
    dataset, queries, distances, distance_metric = bench.load_dataset("glove-25")
    k=10

    qm, q = gen_random_walk(
        dataset,
        k,
        "lid",
        target=40,
        probes=2**4,
        scale=1,
        max_steps=200,
        seed=None,
        startquery=None#queries[9533]
    )
    print("Generated query\n", q, "\nwith metric", qm)

