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
        assert query.shape[0] == dataset.shape[1]
        query_metric = dm.compute(query, dataset, metric, k, distance_metric="angular")
        return query_metric, query

    def gen_query_euclidean(base, gen):
        if base is None:
            mins = np.min(dataset, axis=0)
            maxs = np.max(dataset, axis=0)
            assert mins.shape[0] == dataset.shape[1]
            # offset = np.array([gen.normal(scale=(maxs[i] - mins[i])/3) for i in range(dataset.shape[1])])
            # query = np.mean(dataset, axis=0) + offset #gen.normal(size=dim, scale=scale)
            query = gen.uniform(mins, maxs)
        else:
            offset = gen.normal(scale=scale, size=dim)
            query = base + offset
        assert query.shape[0] == dataset.shape[1]
        query_metric = dm.compute(query, dataset, metric, k, distance_metric="euclidean")
        return query_metric, query

    genfunc_dict = {
        "angular": gen_query_angular,
        "euclidean": gen_query_euclidean
    }
    genfunc = genfunc_dict[distance_metric]

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
        query_metric, query = genfunc(None, gen)
    else:
        query_metric = dm.compute(startquery, dataset, metric, k, distance_metric=distance_metric)
        query = startquery

    path = [query]

    for step in range(max_steps):
        print("step", step, metric, "=", query_metric)
        candidates = [genfunc(query, gen) for _ in range(probes)]
        candidates = sorted(candidates, reverse=should_reverse[metric])
        for candidate_metric, candidate in candidates:
            if harder_than(candidate_metric, target):
                path.append(candidate)
                path = np.stack(path)
                assert path.shape[1] == dataset.shape[1]
                return candidate_metric, candidate, path

        # if we did not return, pick the candidate with the 
        # best metric, if it is better than the current best query
        candidate_metric, candidate = candidates[-1]
        if harder_than(candidate_metric, query_metric):
            query = candidate
            query_metric = candidate_metric
            path.append(query)
    print("Failed to find candidate query")


if __name__ == "__main__":
    import bench
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    # from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

    dataset, queries, distances, distance_metric = bench.load_dataset("fashion-mnist")
    # dataset = PCA(2).fit_transform(dataset)
    k=10

    pca = PCA(2)
    proj = pca.fit_transform(dataset)

    # loglid is the one that seems to be working the best, but why?
    qm, q, path = gen_random_walk(
        dataset,
        k,
        "lid",
        distance_metric=distance_metric,
        target=200, # possibly, the LID should be related to the actual dimensionality of the dataset
        probes=2**4,
        scale=10,
        max_steps=200,
        seed=None,
        startquery=None #queries[9533]
    )
    print("Generated query with metric", qm)
    q = pca.transform(q.reshape(1, -1))
    print(q)
    path = pca.transform(path)
    print(path)

    plt.figure(figsize=(10,10))
    plt.scatter(proj[:,0], proj[:,1], s=1)
    plt.plot(path[:,0], path[:,1], c="orange")
    plt.scatter(path[:,0], path[:,1], c="orange")
    plt.scatter(q[:,0], q[:,1], s=10, c="red")
    plt.savefig("generated.png")

    plt.figure(figsize=(10,10))
    plt.scatter(proj[:,0], proj[:,1], s=1)
    plt.plot(path[:,0], path[:,1], c="orange")
    plt.scatter(path[:,0], path[:,1], c="orange")
    plt.scatter(q[:,0], q[:,1], s=10, c="red")
    scale = 4
    xmin = path[:,0].min()
    ymin = path[:,1].min()
    xmax = path[:,0].max()
    ymax = path[:,1].max()
    xext = abs(xmax - xmin)
    yext = abs(ymax - ymin)
    plt.xlim(xmin - scale*xext, xmax + scale*xext)
    plt.ylim(ymin - scale*yext, ymax + scale*yext)
    print(plt.xlim())
    print(plt.ylim())
    plt.tight_layout()
    plt.savefig("zoomed.png")


