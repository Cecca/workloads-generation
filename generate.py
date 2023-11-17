"""
This module collects approaches to generate workloads for a given dataset.
"""

import numpy as np
import dimensionality_measures as dm
import read_data
import faiss
import utils
import time


class RandomWalk(object):
    def __init__(self, dataset, k, metric, target, probes=4, scale=1.0,
                 max_steps=100, seed=1234, startquery=None, distance_metric="angular"):
        self.dataset = dataset
        self.index = faiss.IndexFlatL2(dataset.shape[1])
        self.index.add(dataset)
        self.k = k
        self.metric = metric
        self.target = target
        self.probes = probes
        self.scale = scale
        self.cnt_steps = 0
        self.max_steps = max_steps
        self.seed = seed
        self.distance_metric = distance_metric
        self.dim = dataset.shape[1]
        # whether the direction of "harder" queries is increasing, metric-wise
        if "lid" == metric:
            self.compute_metric = lambda dists, k: dm.compute_lid(dists, k, "linear")
        elif "rc" == metric:
            self.compute_metric = lambda dists, k: dm.compute_rc(dists, k, "linear")
        elif "expansion" == metric:
            self.compute_metric = lambda dists, k: dm.compute_expansion(dists, k, "linear")
        else:
            raise Exception("Unknown metric %s" % metric)
        self.direction_increasing = metric in ["lid", "loglid"]
        self.gen = np.random.default_rng(seed)
        self.path = []
        self.tested = []
        self.candidate = startquery if startquery is not None else self.generate_random_point().astype(np.float32)
        candidate_distances = utils.compute_distances(self.candidate, None, self.distance_metric, self.index)[0,:]
        self.candidate_metric = self.compute_metric(candidate_distances, self.k)

    def harder_than(self, candidate_metric, target):
        if self.direction_increasing:
            return candidate_metric > target
        else:
            return candidate_metric < target

    def generate_random_point(self):
        raise NotImplementedError()

    def generate_candidate(self, base):
        raise NotImplementedError()

    def move_next(self):
        print("Step", self.cnt_steps, "metric", self.candidate_metric)
        self.cnt_steps += 1
        start = time.time()
        candidates = [
            self.generate_candidate(self.candidate).astype(np.float32) 
            for _ in range(self.probes)
        ]
        end = time.time()
        print("time to generate candidates", end - start, "seconds")
        self.tested.extend(candidates)
        candidates = np.array(candidates)
        start = time.time()
        candidate_distances = utils.compute_distances(candidates, None, self.distance_metric, self.index)
        end = time.time()
        print("Time to compute distances", end - start, "seconds")
        candidates = [
            # (dm.compute(c, self.dataset, self.metric, self.k, distance_metric=self.distance_metric), c)
            ( self.compute_metric(dists, self.k) , c)
            for c, dists in zip(candidates, candidate_distances)
        ]
        candidates = sorted(candidates, reverse=self.direction_increasing)

        # pick the candidate with the best metric, if it is better than the
        # current best candidate
        new_metric, new_candidate = candidates[0]
        if self.harder_than(new_metric, self.candidate_metric):
            self.candidate_metric = new_metric
            self.candidate = new_candidate
            self.path.append(new_candidate)

    def is_done(self):
        return self.harder_than(self.candidate_metric, self.target) or self.cnt_steps >= self.max_steps

    
    def run(self):
        while not self.is_done():
            self.move_next()
        return self.candidate_metric, self.candidate

    def get_path(self):
        return np.stack(self.path)

    def get_tested(self):
        return np.stack(self.tested)


class RandomWalkAngular(RandomWalk):
    def __init__(self, dataset, k, metric, target, probes=4, scale=1, max_steps=100, seed=1234, startquery=None):
        super().__init__(dataset, k, metric, target, probes, scale, max_steps, seed, startquery, distance_metric="angular")

    def generate_random_point(self):
        query = self.gen.normal(size=self.dim)
        query /= np.linalg.norm(query)
        return query

    def generate_candidate(self, base):
        coord = self.gen.integers(self.dim)
        next_coord = (coord+1) % self.dim
        rotation = np.identity(self.dim)
        angle = self.gen.normal(scale=self.scale)
        rotation[coord, coord] = np.cos(angle)
        rotation[next_coord, next_coord] = np.cos(angle)
        rotation[coord, next_coord] = -np.sin(angle)
        rotation[next_coord, coord] = np.sin(angle)
        query = np.dot(base, rotation)
        query /= np.linalg.norm(query)
        return query


class RandomWalkEuclidean(RandomWalk):
    def __init__(self, dataset, k, metric, target, probes=4, scale=1, max_steps=100, seed=1234, startquery=None):
        super().__init__(dataset, k, metric, target, probes, scale, max_steps, seed, startquery, distance_metric="euclidean")

    def generate_random_point(self):
        mins = np.min(self.dataset, axis=0)
        maxs = np.max(self.dataset, axis=0)
        assert mins.shape[0] == self.dataset.shape[1]
        query = self.gen.uniform(mins, maxs)
        return query

    def generate_candidate(self, base):
        offset = self.gen.normal(scale=self.scale, size=self.dim)
        query = base + offset
        return query


def plot_path(dataset, q, path, tested):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(2)
    proj = pca.fit_transform(dataset)
    q = pca.transform(q.reshape(1, -1))
    path = pca.transform(path)
    tested = pca.transform(tested)

    plt.figure(figsize=(10,10))
    plt.scatter(proj[:,0], proj[:,1], s=1, zorder=0)
    plt.scatter(q[:,0], q[:,1], s=20, c="red", zorder=10)
    plt.scatter(path[:,0], path[:,1], c="orange", zorder=3)
    plt.scatter(tested[:,0], tested[:,1], c="yellow", s=2, zorder=2)
    plt.plot(path[:,0], path[:,1], c="orange", zorder=1)
    plt.savefig("generated.png")

    plt.figure(figsize=(10,10))
    plt.scatter(proj[:,0], proj[:,1], s=1, zorder=0)
    plt.scatter(q[:,0], q[:,1], s=20, c="red", zorder=10, edgecolors="black")
    plt.scatter(path[:,0], path[:,1], c="orange", zorder=3, edgecolors="black")
    plt.scatter(tested[:,0], tested[:,1], c="yellow", zorder=2, edgecolors="black")
    plt.plot(path[:,0], path[:,1], c="orange", zorder=1)
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


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to the dataset file')
    parser.add_argument('--query', required=True, help='Path to the query file')
    parser.add_argument('--k', type=int, required=True, help='Number of nearest neighbors to find')
    parser.add_argument('--metric', type=str, required=True, help='Difficult metric')
    parser.add_argument('--target', type=float, required=True, help='Target difficulty')
    parser.add_argument('--probes', type=int, required=True, help='Number of probes to use')
    parser.add_argument('--scale', type=float, required=False, default=10.0, help='Noise scale')
    parser.add_argument('--max-steps', type=int, required=False, default=100, help='Number of random walk steps (maximum)')
    parser.add_argument('--data_limit', type=int,  help='Maximum number of data points to load from the dataset file')
    parser.add_argument('--query_limit', type=int,  help='Maximum number of query points to load from the query file')
    parser.add_argument('--start-from-datapoint', type=int,  help='Datapoint from which to make the search start')
    parser.add_argument('--num-queries', type=int,  help='Number of queries to generate')
    parser.add_argument('--queries-output', type=str,  help='Path to queries output file')


    args = parser.parse_args()

    dataset_name = args.dataset
    queryset_name = args.query
    dataset, _queries, _distances, distance_metric = read_data.read_data(dataset_name, queryset_name)
    print("Dataset with shape", dataset.shape)
    k = args.k
    metric = args.metric
    target = args.target
    probes = args.probes
    scale = args.scale
    max_steps = args.max_steps
    if args.start_from_datapoint is not None:
        starting = dataset[args.start_from_datapoint,:]
    else:
        starting = None

    if distance_metric == "angular":
        walker_class = RandomWalkAngular
    elif distance_metric == "euclidean":
        walker_class = RandomWalkEuclidean
    else:
        raise NotImplementedError("Distance metric not implemented")

    if args.num_queries is None:
        walker = walker_class(
            dataset,
            k,
            metric,
            target=target,
            probes=probes,
            scale=scale,
            max_steps=max_steps,
            startquery = starting
        )

        qm, q = walker.run()
        print("Generated query with metric", qm)
        plot_path(dataset, q, walker.get_path(), walker.get_tested())
    else:
        assert args.queries_output is not None
        if os.path.isfile(args.queries_output):
            exit(f"File {args.queries_output} already exists, provide another one")
        gen = np.random.default_rng(1234)
        starting_ids = gen.choice(np.arange(dataset.shape[0]), size=args.num_queries, replace=False)
        queries = []
        for idx in starting_ids:
            print("Starting from point", idx)
            starting = dataset[idx,:]
            walker = walker_class(
                dataset,
                k,
                metric,
                target=target,
                probes=probes,
                scale=scale,
                max_steps=max_steps,
                startquery = starting
            )
            qm, q = walker.run()
            print("Generated query with metric", qm)
            queries.append(q)
        queries = np.stack(queries)
        write_queries_hdf5(queries, args.queries_output)



def write_queries_hdf5(queries, path):
    import h5py
    with h5py.File(path, "w") as hfp:
        hfp['test'] = queries


if __name__ == "__main__":
    main()
