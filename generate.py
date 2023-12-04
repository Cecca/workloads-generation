"""
This module collects approaches to generate workloads for a given dataset.
"""

import numpy as np
import dimensionality_measures as dm
import read_data
import faiss
import utils
import time


def annealing(
    score, start_point, gen_neighbor, target_low, target_high, temperature, max_steps=100
):
    """
    Parameters
    ==========

    :score: a function accepting a single point and returning a floating point value: higher
            values correspond to more difficult points
    :start_point: the starting point of the annealing process
    :gen_neighbor: takes the current point and generates a neighbor
    :target_low: the lower bound to the score we want to achieve
    :target_high: the upper bound to the score we want to achieve
    :temperature: the temperature schedule, or how the process cools with the iterations
    :max_steps: the maximum number of steps we allow
    """
    import random
    from math import exp

    assert target_low <= target_high
    x, y = start_point, score(start_point)

    for step in range(max_steps):
        print(step)
        x_next = gen_neighbor(x)
        y_next = score(x_next)
        if target_low <= y_next <= target_high:
            print("Returning query point with score", y_next)
            return x_next
        elif y <= y_next <= target_low or target_high <= y_next <= y:
            # the next candidate goes towards the desired range
            x, y = x_next, y_next
            print("move to better candidate with score", y, "rc = ", np.exp(-y))
        else:
            # we pick the neighbor by the Metropolis criterion
            delta = abs(y - y_next)
            t = temperature(step)
            p = exp(-delta / t)
            print("temperature ", t, "delta", delta, "p=", p)
            if random.random() < p:
                x, y = x_next, y_next
                print("move to WORSE candidate with score", y, "rc = ", np.exp(-y))
            pass

    raise Exception("Could not find point in the desired range")


def fast_annealing_schedule(t1):
    def inner(step):
        return t1 / (step+1)
    return inner


def transform_rc(rc):
    """
    Scale transform the relative contrast so that higher values correspond to
    more difficulty queries
    """
    return -np.log(rc)


def relative_contrast_scorer(index, distance_metric, k):
    def inner(x):
        distances = utils.compute_distances(
            x, None, distance_metric, index
        )[0, :]
        rc = dm.compute_rc(distances, k, scale="linear")
        return transform_rc(rc)
    return inner


def neighbor_generator_angular(scale, rng):
    def inner(x):
        offset = rng.normal(scale=scale, size=x.shape[0]).astype(np.float32)
        neighbor = x + offset
        neighbor /= np.linalg.norm(neighbor)
        return neighbor
    return inner


def neighbor_generator_euclidean(scale, rng):
    def inner(x):
        offset = rng.normal(scale=scale, size=x.shape[0]).astype(np.float32)
        neighbor = x + offset
        return neighbor
    return inner


class RandomWalk(object):
    def __init__(
        self,
        dataset,
        k,
        metric,
        target_low,
        target_high,
        probes=4,
        scale=1.0,
        max_steps=100,
        seed=1234,
        startquery=None,
        distance_metric="angular",
    ):
        self.dataset = dataset
        self.index = faiss.IndexFlatL2(dataset.shape[1])
        self.index.add(dataset)
        self.k = k
        self.metric = metric
        self.target_low = target_low
        self.target_high = target_high
        self.probes = probes
        self.scale = scale
        self.cnt_steps = 0
        self.max_steps = max_steps
        self.seed = seed
        self.distance_metric = distance_metric
        self.dim = dataset.shape[1]
        # whether the direction of "harder" queries is increasing, metric-wise
        if "lid" == metric:
            self.compute_metric = lambda dists: dm.compute_lid(dists, k, "linear")
        elif "rc" == metric:
            self.compute_metric = lambda dists: dm.compute_rc(dists, k, "linear")
        elif "expansion" == metric:
            self.compute_metric = lambda dists: dm.compute_expansion(dists, k, "linear")
        else:
            raise Exception("Unknown metric %s" % metric)
        self.direction_increasing = metric in ["lid", "loglid"]
        print("random seed", seed)
        self.gen = np.random.default_rng(seed)
        self.path = []
        self.tested = []
        self.candidate = (
            startquery
            if startquery is not None
            else self.generate_random_point().astype(np.float32)
        )
        candidate_distances = utils.compute_distances(
            self.candidate, None, self.distance_metric, self.index
        )[0, :]
        self.candidate_metric = self.compute_metric(candidate_distances)
        assert self.easier_than(target_low, target_high)

    def harder_than(self, candidate_metric, target):
        if self.direction_increasing:
            return candidate_metric > target
        else:
            return candidate_metric < target

    def easier_than(self, candidate_metric, target):
        return not self.harder_than(candidate_metric, target)

    def generate_random_point(self):
        raise NotImplementedError()

    def generate_candidate(self, base):
        raise NotImplementedError()

    def move_next(self):
        step_start = time.time()
        print("Step", self.cnt_steps, "metric", self.candidate_metric)
        self.cnt_steps += 1

        candidates = [
            self.generate_candidate(self.candidate).astype(np.float32)
            for _ in range(self.probes)
        ]
        self.tested.extend(candidates)
        candidates = np.array(candidates)
        candidate_distances = utils.compute_distances(
            candidates, None, self.distance_metric, self.index
        )
        candidates = [
            (self.compute_metric(dists), c)
            for c, dists in zip(candidates, candidate_distances)
        ]
        # sort candidates by decreasing hardness
        candidates = sorted(
            candidates, reverse=self.direction_increasing, key=lambda tup: tup[0]
        )

        # partition the candidates
        harder = list(
            filter(lambda cand: self.harder_than(cand[0], self.target_high), candidates)
        )
        easier = list(
            filter(lambda cand: self.easier_than(cand[0], self.target_low), candidates)
        )
        hits = list(
            filter(
                lambda cand: self.easier_than(cand[0], self.target_high)
                and self.harder_than(cand[0], self.target_low),
                candidates,
            )
        )
        print("harder", [t[0] for t in harder])
        print("easier", [t[0] for t in easier])
        print("hits", [t[0] for t in hits])

        if len(hits) > 0:
            # we found at least a candidate within the target difficulty range
            new_metric, new_candidate = hits[0]
            pick = True
        elif len(easier) > 0:
            # we generated easier queries, start from the most difficult one amongst those,
            # leveraging the fact that they are already sorted by decreasing difficulty
            new_metric, new_candidate = easier[0]
            pick = self.harder_than(new_metric, self.candidate_metric)
        else:
            assert len(harder) > 0
            # we generated harder queries, move the the easiest among those
            new_metric, new_candidate = harder[-1]
            pick = self.easier_than(new_metric, self.candidate_metric)

        # pick the candidate with the best metric, if it is better than the
        # current best candidate
        # new_metric, new_candidate = candidates[0]
        # if self.harder_than(new_metric, self.candidate_metric):
        if pick:
            self.candidate_metric = new_metric
            self.candidate = new_candidate
            self.path.append(new_candidate)
        step_end = time.time()
        print("Step time", step_end - step_start, "seconds")

    def is_done(self):
        return (
            self.harder_than(self.candidate_metric, self.target_low)
            and self.easier_than(self.candidate_metric, self.target_high)
        ) or self.cnt_steps >= self.max_steps

    def run(self):
        while not self.is_done():
            self.move_next()
        return self.candidate_metric, self.candidate

    def get_path(self):
        return np.stack(self.path)

    def get_tested(self):
        return np.stack(self.tested)


class RandomWalkAngular(RandomWalk):
    def __init__(
        self,
        dataset,
        k,
        metric,
        target_low,
        target_high,
        probes=4,
        scale=1,
        max_steps=100,
        seed=1234,
        startquery=None,
    ):
        super().__init__(
            dataset,
            k,
            metric,
            target_low,
            target_high,
            probes,
            scale,
            max_steps,
            seed,
            startquery,
            distance_metric="angular",
        )

    def generate_random_point(self):
        query = self.gen.normal(size=self.dim)
        query /= np.linalg.norm(query)
        return query

    def generate_candidate(self, base):
        offset = self.gen.normal(scale=self.scale, size=self.dim)
        query = base + offset
        query /= np.linalg.norm(query)
        d = 1 - np.dot(base, query)
        print("   angular distance from base point", d)
        return query


class RandomWalkEuclidean(RandomWalk):
    def __init__(
        self,
        dataset,
        k,
        metric,
        target_low,
        target_high,
        probes=4,
        scale=1,
        max_steps=100,
        seed=1234,
        startquery=None,
    ):
        super().__init__(
            dataset,
            k,
            metric,
            target_low,
            target_high,
            probes,
            scale,
            max_steps,
            seed,
            startquery,
            distance_metric="euclidean",
        )

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

    plt.figure(figsize=(10, 10))
    plt.scatter(proj[:, 0], proj[:, 1], s=1, zorder=0)
    plt.scatter(q[:, 0], q[:, 1], s=20, c="red", zorder=10)
    plt.scatter(path[:, 0], path[:, 1], c="orange", zorder=3)
    plt.scatter(tested[:, 0], tested[:, 1], c="yellow", s=2, zorder=2)
    plt.plot(path[:, 0], path[:, 1], c="orange", zorder=1)
    plt.savefig("generated.png")

    plt.figure(figsize=(10, 10))
    plt.scatter(proj[:, 0], proj[:, 1], s=1, zorder=0)
    plt.scatter(q[:, 0], q[:, 1], s=20, c="red", zorder=10, edgecolors="black")
    plt.scatter(path[:, 0], path[:, 1], c="orange", zorder=3, edgecolors="black")
    plt.scatter(tested[:, 0], tested[:, 1], c="yellow", zorder=2, edgecolors="black")
    plt.plot(path[:, 0], path[:, 1], c="orange", zorder=1)
    scale = 4
    xmin = path[:, 0].min()
    ymin = path[:, 1].min()
    xmax = path[:, 0].max()
    ymax = path[:, 1].max()
    xext = abs(xmax - xmin)
    yext = abs(ymax - ymin)
    plt.xlim(xmin - scale * xext, xmax + scale * xext)
    plt.ylim(ymin - scale * yext, ymax + scale * yext)
    print(plt.xlim())
    print(plt.ylim())
    plt.tight_layout()
    plt.savefig("zoomed.png")


def generate_workload(
    dataset_input,
    queries_output,
    k,
    metric,
    target_low,
    target_high,
    num_queries,
    probes=8,
    scale=10,
    max_steps=300,
    seed=1234,
):
    dataset, distance_metric = read_data.read_hdf5(dataset_input, "train")
    print("loaded", dataset.shape)
    if distance_metric == "angular":
        walker_class = RandomWalkAngular
    elif distance_metric == "euclidean":
        walker_class = RandomWalkEuclidean
    else:
        raise NotImplementedError("Distance metric not implemented")

    gen = np.random.default_rng(seed)
    starting_ids = gen.choice(
        np.arange(dataset.shape[0]), size=num_queries, replace=False
    )
    queries = []
    for idx in starting_ids:
        print("Starting from point", idx)
        starting = dataset[idx, :]
        walker = walker_class(
            dataset,
            k,
            metric,
            target_low=target_low,
            target_high=target_high,
            probes=probes,
            scale=scale,
            max_steps=max_steps,
            startquery=starting,
            seed=seed,
        )
        qm, q = walker.run()
        print("Generated query with metric", qm)
        queries.append(q)
    queries = np.stack(queries)
    write_queries_hdf5(queries, queries_output)



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    parser.add_argument(
        "--k", type=int, required=True, help="Number of nearest neighbors to find"
    )
    parser.add_argument("--metric", type=str, required=True, help="Difficult metric")
    parser.add_argument(
        "--target-low", type=float, required=True, help="Target difficulty, lower bound"
    )
    parser.add_argument(
        "--target-high",
        type=float,
        required=True,
        help="Target difficulty, upper bound",
    )
    parser.add_argument(
        "--num-queries", type=int, default=100, help="Number of queries to generate"
    )
    parser.add_argument(
        "--queries-output", type=str, required=True, help="Path to queries output file"
    )
    parser.add_argument(
        "--probes", type=int, required=True, help="Number of probes to use"
    )
    parser.add_argument(
        "--scale", type=float, required=False, default=10.0, help="Noise scale"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=False,
        default=1000,
        help="Number of random walk steps (maximum)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="seed for the random number generator"
    )

    args = parser.parse_args()

    generate_workload(
        args.dataset,
        args.queries_output,
        args.k,
        args.metric,
        args.target_low,
        args.target_high,
        args.num_queries,
        args.probes,
        args.scale,
        args.max_steps,
        args.seed,
    )


def write_queries_hdf5(queries, path):
    import h5py

    with h5py.File(path, "w") as hfp:
        hfp["test"] = queries


def main_annealing():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    parser.add_argument(
        "--k", type=int, required=True, help="Number of nearest neighbors to find"
    )
    parser.add_argument("--metric", type=str, required=True, help="Difficult metric")
    parser.add_argument(
        "--target-low", type=float, required=True, help="Target difficulty, lower bound"
    )
    parser.add_argument(
        "--target-high",
        type=float,
        required=True,
        help="Target difficulty, upper bound",
    )
    parser.add_argument(
        "--scale", type=float, required=False, default=10.0, help="Noise scale"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=False,
        default=1000,
        help="Number of random walk steps (maximum)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="seed for the random number generator"
    )

    args = parser.parse_args()

    gen = np.random.default_rng(args.seed)

    dataset, distance_metric = read_data.read_hdf5(args.dataset, "train")
    print("loaded dataset with shape", dataset.shape)

    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)

    neighbor_generators = {
        "angular": neighbor_generator_angular(args.scale, gen),
        "euclidean": neighbor_generator_euclidean(args.scale, gen)
    }
    gen_neighbor = neighbor_generators[distance_metric]

    scoring_functions = {
        "rc": relative_contrast_scorer(index, distance_metric, args.k)
    }
    score = scoring_functions[args.metric]

    score_transforms = {
        "rc": transform_rc
    }
    score_transform = score_transforms[args.metric]
    target_low = score_transform(args.target_low)
    target_high = score_transform(args.target_high)

    num_queries = 1
    starting_ids = gen.choice(
        np.arange(dataset.shape[0]), size=num_queries, replace=False
    )

    q = annealing(
        score,
        dataset[starting_ids[0],:],
        gen_neighbor,
        target_low,
        target_high,
        fast_annealing_schedule(10.0),
        max_steps = args.max_steps
    )

if __name__ == "__main__":
    main_annealing()
    
