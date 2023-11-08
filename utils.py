import numpy as np


DATASETS = {
    "fashion-mnist": ".fashion-mnist-784-euclidean.hdf5",
    "glove-100": ".glove-100-angular.hdf5",
    "glove-25": ".glove-25-angular.hdf5"
}


def compute_distances(query, k, metric, data_or_index):
    """
    Compute the k smallest distances from the given query in the
    specified metric space. The last argument can be either:

      - a dataset, in which case the distance is brute forced, 
      - or a FAISS index to be used to answer the query

    The function applies the appropriate adjustments so that the output is:

      - the Euclidean distance (not squared) is `metric="euclidean"`
      - the angular distance (i.e. 1 - dot(q,d), with q being the query 
        and d a data point) if `metric="angular"`
    """
    if len(query.shape) == 2:
        # We were given multiple queries
        return np.array([
            compute_distances(q, k, metric, data_or_index)
            for q in query
        ])
    if hasattr(data_or_index, 'shape'):
        dataset = data_or_index
        if metric == "angular":
            dists = 1 - np.dot(dataset, query) 
        elif metric == "euclidean":
            dists = np.linalg.norm(query - dataset, axis=1)
        else:
            raise RuntimeError("unknown distance" + metric)
        dists = np.partition(dists, k)[:k]
        return np.sort(dists)
    else:
        faiss_index = data_or_index
        qq = np.array([query]) # just to comply with faiss API
        dists = faiss_index.search(qq, k)[0][0]
        if metric == "angular":
            # The index returns squared euclidean distances, 
            # which we turn to angular distances in the following
            return 1 - (2 - dists) / 2
        elif metric == "euclidean" :
            # The index returns the _squared_ euclidean distances
            return np.sqrt(dists)
        else:
            raise RuntimeError("unknown distance" + metric)


def compute_recall(ground_distances, run_distances, count, epsilon=1e-3):
    """
    Compute the recall against the given ground truth, for `count` 
    number of neighbors.
    """
    t = ground_distances[count - 1] + epsilon
    actual = 0
    for d in run_distances[:count]:
        if d <= t:
            actual += 1
    return float(actual) / float(count)


