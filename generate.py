"""
This module collects approaches to generate workloads for a given dataset.
"""

import numpy as np
import dimensionality_measures as dm
import read_data


class RandomWalk(object):
    def __init__(self, dataset, k, metric, target, probes=4, scale=1.0,
                 max_steps=100, seed=1234, startquery=None, distance_metric="angular"):
        self.dataset = dataset
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
        self.direction_increasing = metric in ["lid", "loglid"]
        self.gen = np.random.default_rng(seed)
        self.path = []
        self.candidate = startquery if startquery is not None else self.generate_random_point()
        self.candidate_metric = dm.compute(self.candidate, self.dataset, self.metric, self.k, distance_metric=distance_metric)

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
        candidates = [self.generate_candidate(self.candidate) for _ in range(self.probes)]
        candidates = [
            (dm.compute(c, self.dataset, self.metric, self.k, distance_metric=self.distance_metric), c)
            for c in candidates
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


def plot_path(dataset, q, path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(2)
    proj = pca.fit_transform(dataset)
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


def main():
    import argparse

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


    args = parser.parse_args()

    dataset_name = args.dataset
    queryset_name = args.query
    dataset, queries, _distances, distance_metric = read_data.read_data(dataset_name, queryset_name)
    print("Dataset with shape", dataset.shape)
    k = args.k
    metric = args.metric
    target = args.target
    probes = args.probes
    scale = args.scale
    max_steps = args.max_steps

    if distance_metric == "angular":
        walker = RandomWalkAngular(
            dataset,
            k,
            metric,
            target=target,
            probes=probes,
            scale=scale,
            max_steps=max_steps
        )
    elif distance_metric == "euclidean":
        walker = RandomWalkEuclidean(
            dataset,
            k,
            metric,
            target=target,
            probes=probes,
            scale=scale,
            max_steps=max_steps
        )
    else:
        raise NotImplementedError("Distance metric not implemented")

    qm, q = walker.run()
    print("Generated query with metric", qm)
    path = walker.get_path()
    plot_path(dataset, q, path)


if __name__ == "__main__":
    main()
