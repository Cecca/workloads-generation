"""
This module collects approaches to generate workloads for a given dataset.
"""

import numpy as np
import dimensionality_measures as dm


def gen_random_walk(dataset, k, metric, target, probes=4, scale=1.0,
                    max_steps=100, seed=1234, startquery=None, distance_metric="angular"):
    dim = dataset.shape[1]

    def gen_query_angular(base, gen):
        if base is None:
            query = gen.normal(size=dim)
        else:
            rotation = gen.normal(scale=scale, size=(dim, dim))
            query = np.dot(base, rotation)
        query /= np.linalg.norm(query)
        query_metric = dm.compute(query, dataset, metric, k, distance_metric="angular")
        return query_metric, query

    def gen_query_euclidean(base, gen):
        if base is None:
            query = gen.normal(size=dim)
        else:
            offset = gen.normal(scale=scale, size=(dim, dim))
            query = base + offset
        query_metric = dm.compute(query, dataset, metric, k, distance_metric="euclidean")
        return query_metric, query

    # comparator that sets the direction of harder queries, for the given metric
    harder_than_cmp = {
        "lid": "__gt__",
        "loglid": "__gt__",
        "rc": "__lt__",
        "logrc": "__lt__",
        "expansion": "__lt__",
        "logexpansion": "__lt__"
    }
    should_reverse = {
        "lid": False,
        "loglid": False,
        "rc": True,
        "logrc": True,
        "expansion": True,
        "logexpansion": True
    }
    def harder_than(m, ref):
        return getattr(m, harder_than_cmp[metric])(ref)

    gen = np.random.default_rng(seed)
    if startquery is None:
        query_metric, query = gen_query_angular(None, gen)
    else:
        query_metric = dm.compute(startquery, dataset, metric, k, distance_metric=distance_metric)
        query = startquery
    print("initial query", query)

    for step in range(max_steps):
        print("step", step, "metric", query_metric)
        candidates = [gen_query_angular(query, gen) for _ in range(probes)]
        candidates = sorted(candidates, reverse=should_reverse[metric])
        for candidate_metric, candidate in candidates:
            if harder_than(candidate_metric, target):
                return candidate_metric, candidate

        # if we did not return, pick the candidate with the 
        # best metric, if it is better than the current best query
        candidate_metric, candidate = candidates[-1]
        if harder_than(candidate_metric, query_metric):
            query = candidate
            query_metric = candidate_metric
    print("Failed to find candidate query")


if __name__ == "__main__":
    import bench
    dataset, queries, distances, distance_metric = bench.load_dataset("glove-25")
    k=10

    # loglid is the one that seems to be working the best, but why?
    qm, q = gen_random_walk(
        dataset,
        k,
        "loglid",
        target= 40,
        probes=2**6,
        scale=1,
        max_steps=200,
        seed=None,
        startquery=None #queries[9533]
    )
    print("Generated query\n", q, "\nwith metric", qm)

