"""
This module collects approaches to generate workloads for a given dataset.
"""

from re import search
from icecream import ic
import os
import numpy as np
from scipy.spatial import distance
import read_data
import dimensionality_measures as dm
import metrics
from metrics import EmpiricalDifficultyHNSW, EmpiricalDifficultyIVF
import faiss
import utils
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import jax
import jax.numpy as jnp
import optax
from cache import MEM


@MEM.cache
def generate_queries_gaussian_noise(
    dataset,
    distance_metric,
    num_queries,
    scale,
    seed=1234,
):
    assert distance_metric in ["angular", "euclidean", "ip"]
    # multiply by the scale so that different scales with the same seed are not scaled versions of the same noise
    gen = np.random.default_rng(int(seed * scale))
    starting_ids = gen.choice(
        np.arange(dataset.shape[0]), size=num_queries, replace=False
    )
    pts = dataset[starting_ids, :]
    noise = gen.normal(0, scale, size=(num_queries, pts.shape[1]))
    pts += noise
    if distance_metric == "angular":
        pts /= np.linalg.norm(pts, axis=1)[:, np.newaxis]
    if distance_metric == "ip":
        # reset the last element
        pts[:,-1] = 0

    assert np.all(
        np.isfinite(pts)
    ), f"Some values are infinite or NaN in the queries just generated with gaussian noise with scale {scale}"

    return pts.astype(np.float32)


def generate_workload_gaussian_noise(
    dataset_input,
    queries_output,
    num_queries,
    scale,
    seed=1234,
):
    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train")
    print("loaded dataset with shape", dataset.shape)
    if scale in ["easy", "medium", "hard"]:
        index = faiss.IndexFlatL2(dataset.shape[1])
        index.add(dataset)
        diameter = _estimate_diameter(dataset, index, distance_metric)
        logging.info("diameter is %f", diameter)
        scale = {
            "easy": diameter / 100000,
            "medium": diameter / 10000,
            "hard": diameter / 1000,
        }[scale]
        logging.info("scale is %f", scale)

    queries = generate_queries_gaussian_noise(
        dataset,
        distance_metric,
        num_queries,
        scale,
        seed,
    )

    if queries_output.endswith(".bin"):
        queries.tofile(queries_output)
        check, _ = read_data.read_multiformat(queries_output, "", repair=False)
        assert np.all(np.isclose(check, queries))
    elif queries_output.endswith(".hdf5"):
        write_queries_hdf5(queries, queries_output)
    else:
        raise ValueError(f"Unknown format `{queries_output}`")


def _estimate_diameter(data, index, distance_metric):
    q = data[0]
    distances = utils.compute_distances(q, None, distance_metric, index)[0]
    return distances[-1] * 2


def generate_queries_annealing(
    dataset,
    distance_metric,
    k,
    metric,
    target_low,
    target_high,
    num_queries,
    scale="auto",
    max_steps=2000,
    initial_temperature=1.0,
    seed=1234,
    threads=os.cpu_count(),
    max_rounds=3,
):
    gen = np.random.default_rng(seed)

    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)

    orig_scale = scale
    if scale == "auto":
        if distance_metric == "angular":
            scale = 0.1
        else:
            diameter = _estimate_diameter(dataset, index, distance_metric)
            scale = diameter / 100
        logging.info("using automatic scale: %f", scale)
    else:
        logging.info("using user-defined scale of %f", scale)

    neighbor_generators = {
        "angular": neighbor_generator_angular(scale, gen),
        "euclidean": neighbor_generator_euclidean(scale, gen),
        "ip": neighbor_generator_ip(scale, gen),
    }
    gen_neighbor = neighbor_generators[distance_metric]

    scoring_functions = {
        "rc": relative_contrast_scorer(index, distance_metric, k),
        "faiss_ivf": faiss_ivf_scorer(index, dataset, distance_metric, k),
        "hnsw": hnsw_scorer(index, dataset, distance_metric, k),
    }
    score = scoring_functions[metric]

    score_transforms = {"rc": transform_rc, "faiss_ivf": lambda s: s}
    score_transform = score_transforms.get(metric, lambda s: s)
    target_low = score_transform(target_low)
    target_high = score_transform(target_high)

    queries = []

    return_any = max_rounds == "return-any"
    max_rounds = 1 if return_any else max_rounds

    round = 0
    while len(queries) < num_queries and round < max_rounds:
        round += 1
        nq = num_queries - len(queries)
        starting_ids = list(
            gen.choice(np.arange(dataset.shape[0]), size=nq, replace=False)
        )
        logging.debug("Starting from %s", starting_ids)
        logging.info(
            f"Round to generate {nq} queries from {starting_ids} target ({target_low}, {target_high})"
        )

        with ThreadPoolExecutor(threads) as pool:
            tasks = {
                pool.submit(
                    annealing,
                    score,
                    dataset[x, :],
                    gen_neighbor,
                    target_low,
                    target_high,
                    fast_annealing_schedule(initial_temperature),
                    max_steps=max_steps,
                ): x
                for x in starting_ids
            }
            for future in tasks:
                query, success = future.result()
                if not success:
                    logging.warning("Unable to generate query from %d", tasks[future])
                    if return_any:
                        logging.info("Appending anyway to result")
                        assert np.linalg.norm(query) > 0
                        assert np.all(np.isfinite(query))
                        queries.append(query)
                else:
                    queries.append(query)
        if len(queries) < nq / 2:
            logging.info(
                "not a single query generated after %d steps. Restarting", max_steps
            )
            if orig_scale == "auto":
                if distance_metric == "angular":
                    scale += 0.1
                else:
                    diameter = _estimate_diameter(dataset, index, distance_metric)
                    scale *= 10
                logging.info("restarting using automatic scale: %f", scale)
                neighbor_generators = {
                    "angular": neighbor_generator_angular(scale, gen),
                    "euclidean": neighbor_generator_euclidean(scale, gen),
                    "ip": neighbor_generator_ip(scale, gen),
                }
                gen_neighbor = neighbor_generators[distance_metric]

    queries = np.vstack(queries).astype(np.float32)
    assert queries.shape[0] == num_queries
    return queries


def annealing(
    score,
    start_point,
    gen_neighbor,
    target_low,
    target_high,
    temperature,
    max_steps=100,
    return_progress=False
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
    x_best, y_best = x, y
    y_start = y
    logging.info("start from score %f", y)
    if target_low <= y <= target_high:
        logging.info("point is already in the desired range")
        return x, True

    steps_since_last_improvement = 0
    steps_threshold = max(max_steps // 100, 10)
    logging.debug("steps threshold %d", steps_threshold)

    progress = []

    t_start = time.time()
    for step in range(max_steps):
        if steps_since_last_improvement >= steps_threshold:
            logging.debug("moving back to the previous best due to lack of improvement")
            x, y = x_best, y_best
            steps_since_last_improvement = 0

        if return_progress:
            rc = np.exp(-y)
            progress.append({
                "iteration": step,
                "rc": rc,
                "elapsed_s": time.time() - t_start
            })

        x_next = gen_neighbor(x)
        y_next = score(x_next)
        # FIXME: handle case of navigating towards easier points
        if target_low <= y_next <= target_high:
            logging.info(
                "Returning query point with score %f after %d iterations (%.2f s)",
                y_next,
                step,
                time.time() - t_start,
            )
            if return_progress:
                return x_next, True, progress
            else:
                return x_next, True
        # elif y <= y_next <= target_low or target_high <= y_next <= y:
        elif min(abs(y_next - target_low), abs(y_next - target_high)) <= min(
            abs(y - target_low), abs(y - target_high)
        ):
            # the next candidate goes towards the desired range
            x, y = x_next, y_next
            logging.debug(
                "new best score %f, (still %f to go)",
                y,
                min(abs(y_next - target_low), abs(y_next - target_high)),
            )
            x_best, y_best = x, y
            steps_since_last_improvement = 0
        else:
            # we pick the neighbor by the Metropolis criterion
            delta = abs(y - y_next)
            t = temperature(step)
            p = exp(-delta / t)
            if random.random() < p:
                x, y = x_next, y_next
                logging.debug(
                    "new score %f temperature %f (%d since last improvement, p=%f, delta=%f)",
                    y,
                    t,
                    steps_since_last_improvement,
                    p,
                    delta,
                )
            else:
                # logging.debug("rejecting proposal (temperature %f, p %f)", t, p)
                pass
            steps_since_last_improvement += 1
        if step % 50 == 0:
            logging.info(
                "%d/%d current score %f, (still %f to target)",
                step,
                max_steps,
                y,
                min(abs(y - target_low), abs(y - target_high)),
            )

    if return_progress:
        return x, False, progress
    else:
        return x, False


def fast_annealing_schedule(t1):
    def inner(step):
        return t1 / (step + 1)

    return inner


def transform_rc(rc):
    """
    Scale transform the relative contrast so that higher values correspond to
    more difficulty queries
    """
    return -np.log(rc)


def relative_contrast_scorer(index, distance_metric, k):
    def inner(x):
        distances = utils.compute_distances(x, None, distance_metric, index)[0, :]
        rc = dm.compute_rc(distances, k, scale="linear")
        return transform_rc(rc)

    return inner


def partition_by(candidates, fun):
    # first do an exponential search
    upper = 0
    lower = 0
    cur_res = None
    while upper < len(candidates):
        res = fun(candidates[upper])
        if res is not None:
            cur_res = res
            break
        lower = upper
        upper = upper * 2 if upper > 0 else 1

    # now we know that the predicate is satisfied between prev_ids (where it
    # is not satisfied) and cur_idx (where it is satisfied). So we do a binary search between the two
    while lower < upper:
        mid = (lower + upper) // 2
        mid_res = fun(candidates[mid])
        if mid_res is not None:
            cur_res = mid_res
            upper = mid
        else:
            lower = mid + 1

    return cur_res


def hnsw_scorer(
    exact_index, dataset, distance_metric, k, recall=0.95, n_list=None
):
    """
    Score a point by the fraction of distance computations (wrt to the total) that the
    hnsw index has to do to reach a given target recall.
    """
    difficulty_hnsw = EmpiricalDifficultyHNSW(
        dataset, recall, exact_index, distance_metric
    )

    def inner(x):
        return difficulty_hnsw.evaluate(x, k)

    return inner


def faiss_ivf_scorer(
    exact_index, dataset, distance_metric, k, recall=0.95, n_list=None
):
    """
    Score a point by the fraction of distance computations (wrt to the total) that the
    faiss ivf index has to do to reach a given target recall.
    """
    difficulty_ivf = EmpiricalDifficultyIVF(
        dataset, recall, exact_index, distance_metric
    )

    def inner(x):
        return difficulty_ivf.evaluate(x, k)

    return inner


def neighbor_generator_angular(scale, rng):
    a = scale / (1 - scale)
    b = 1

    def inner(x):
        # get orthogonal vector in a random direction
        y = rng.normal(size=x.shape[0])
        y[0] = (0.0 - np.sum(x[1:] * y[1:])) / x[0]
        y /= np.linalg.norm(y)
        # get a random value for the dot product
        d = 1 - rng.beta(a, b)
        # compute the neighbor position
        neighbor = x + y * np.tan(np.arccos(d))
        neighbor /= np.linalg.norm(neighbor)
        # check that the dot product is what we expect
        d_check = np.dot(x, neighbor)
        assert np.isclose(d, d_check)
        return neighbor

    return inner


def neighbor_generator_euclidean(scale, rng):
    def inner(x):
        direction = rng.normal(size=x.shape[0]).astype(np.float32)
        direction /= np.linalg.norm(direction)
        amount = rng.exponential(scale=scale)
        offset = direction * amount
        neighbor = x + offset
        # logging.debug("euclidean distance: %f", np.linalg.norm(x - neighbor))
        return neighbor

    return inner

def neighbor_generator_ip(scale, rng):
    def inner(x):
        direction = rng.normal(size=x.shape[0]).astype(np.float32)
        direction /= np.linalg.norm(direction)
        amount = rng.exponential(scale=scale)
        offset = direction * amount
        neighbor = x + offset
        neighbor[-1] = 0 # reset the last coordinate
        # logging.debug("euclidean distance: %f", np.linalg.norm(x - neighbor))
        return neighbor

    return inner



def _average_rc(data, distance_metric, k, sample_size=100, seed=1234):
    logging.info("computing the average RC")
    gen = np.random.default_rng(seed)
    indices = gen.integers(data.shape[0], size=sample_size)
    qs = data[indices, :]
    if distance_metric == "ip":
        qs[:,-1] = 0
    data = np.delete(data, indices, axis=0)
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    avg_rc = 0
    cnt = 0
    for qidx in range(sample_size):
        q = qs[qidx]
        distances = utils.compute_distances(q, None, distance_metric, data)[0]
        knn_dist = distances[k]
        # discard the distances that are 0
        if knn_dist <= 0.0:
            continue
        avg_dists = distances.mean()
        rc = avg_dists / knn_dist
        avg_rc += rc
        cnt += 1

    avg_rc = avg_rc / cnt
    print("Average relative contrast is", avg_rc)
    assert np.isfinite(avg_rc)
    return avg_rc


def generate_workload_annealing(
    dataset_input,
    queries_output,
    k,
    metric,
    target_class,
    num_queries,
    initial_temperature=1.0,
    scale="auto",
    max_steps=2000,
    seed=1234,
    threads=os.cpu_count(),
    max_rounds=3,
):
    assert target_class in ["easy", "medium", "hard"]

    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train")
    print("loaded dataset with shape", dataset.shape)

    if metric == "rc":
        avg_rc = _average_rc(dataset, distance_metric, k)
        if distance_metric == "angular":
            target_rc = {
                "easy": (avg_rc - 1) * 1.5 + 1,
                "medium": (avg_rc - 1) * 1.0 + 1,
                "hard": (avg_rc - 1) / 2 + 1,
            }[target_class]
        else:
            target_rc = {
                "easy": (avg_rc - 1) / 2 + 1,
                "medium": (avg_rc - 1) / 10 + 1,
                "hard": (avg_rc - 1) / 100 + 1,
            }[target_class]
        delta = 0.05 * target_rc
        target_low = target_rc + delta
        target_high = target_rc - delta
    else:
        raise NotImplemented("not yet implemented")

    queries = generate_queries_annealing(
        dataset,
        distance_metric,
        k,
        metric,
        target_low,
        target_high,
        num_queries,
        scale,
        max_steps,
        initial_temperature,
        seed,
        threads,
        max_rounds,
    )

    if queries_output.endswith(".bin"):
        assert np.all(np.isfinite(queries))
        queries.tofile(queries_output)
        check, _ = read_data.read_multiformat(queries_output, "", repair=False)
        assert np.all(np.isclose(check, queries))
    elif queries_output.endswith(".hdf5"):
        write_queries_hdf5(queries, queries_output)
    else:
        raise ValueError(f"Unknown format `{queries_output}`")


def write_queries_hdf5(queries, path):
    import h5py

    with h5py.File(path, "w") as hfp:
        hfp["test"] = queries


@jax.jit
def _euclidean(x, dataset):
    return jnp.linalg.norm(dataset - x, axis=1)


@jax.jit
def _embedded_ip(x, dataset):
    assert x.shape[0] == dataset.shape[1] - 1
    x = jnp.hstack((x, 0))
    return jnp.linalg.norm(dataset - x, axis=1)


@jax.jit
def _angular(x, dataset):
    return 1 - jnp.dot(dataset, x)


def _rc(x, dataset, distance_fn, k):
    dists = distance_fn(x, dataset)
    dists = jnp.sort(dists)
    return jnp.mean(dists) / dists[k]


def generate_query_sgd(
    dataset,
    distance_metric,
    k,
    target_low,
    target_high,
    learning_rate=1.0,
    max_iter=1000,
    seed=1234,
    start_point=None,
    return_progress=False,
    return_intermediate=False,
    scoring_function=None
):
    assert target_low <= target_high
    if distance_metric == "euclidean":
        distance_fn = _euclidean
    elif distance_metric == "ip":
        distance_fn = _embedded_ip
    elif distance_metric == "angular":
        distance_fn = _angular
    else:
        raise ValueError(f"unknown distance metric {distance_metric}")

    t_start = time.time()
    progress = []
    intermediate = []

    grad_fn = jax.value_and_grad(_rc)

    rng = np.random.default_rng(seed)
    optimizer = optax.adam(learning_rate)
    if start_point is None:
        x = dataset[rng.integers(dataset.shape[1]-1)] + rng.normal(size=dataset.shape[1])
    else:
        x = start_point + rng.normal(size=dataset.shape[1], scale=0.001)
    if distance_metric == "angular":
        x /= jnp.linalg.norm(x)
    if distance_metric == "ip":
        x = x[:-1]
    opt_state = optimizer.init(x)

    for i in range(max_iter):
        ic(x.shape, dataset.shape[1])
        rc, grads = grad_fn(x, dataset, distance_fn, k)
        assert np.isfinite(rc)
        if return_intermediate:
            intermediate.append(x)
        if return_progress:
            progress.append(
                {
                    "iteration": i,
                    "rc": rc,
                    "elapsed_s": time.time() - t_start
                }
            )
        if scoring_function is not None:
            score = scoring_function(x)
        else:
            score = rc
        logging.debug(
            "[%d] rc=%.4f score=%.4f (target range [%.4f, %.4f])",
            i,
            rc,
            score,
            target_low,
            target_high,
        )
        if target_low <= score and score <= target_high:
            break

        if distance_metric == "angular":
            # project the gradients on the tangent plane
            grads = grads - jnp.dot(grads, x) * x
            grads /= jnp.linalg.norm(grads)

        # FIXME: handle this case in general
        if scoring_function is not None:
            if score > target_high:
                grads = -grads
        else:
            if score < target_high:
                grads = -grads
        
        # if score < target_low:
        #     grads = -grads

        updates, opt_state = optimizer.update(grads, opt_state)
        x = optax.apply_updates(x, updates)
        if distance_metric == "angular":
            x /= jnp.linalg.norm(x)
        if distance_metric == "ip":
            assert x.shape[0] == dataset.shape[1] - 1

        assert np.all(np.isfinite(x))

    if return_intermediate:
        return np.stack(intermediate)
    if return_progress:
        return np.hstack((x, 0)), progress
    else:
        return np.hstack((x, 0))


#@MEM.cache
def generate_queries_sgd(
    dataset,
    distance_metric,
    k,
    num_queries,
    target_low,
    target_high,
    learning_rate=1.0,
    max_iter=1000,
    scoring_function=None,
    seed=1234,
    start_point=None,
    return_progress=False,
    threads=1,
):
    queries = []

    with ThreadPoolExecutor(threads) as pool:
        tasks = {
            pool.submit(
                generate_query_sgd,
                dataset,
                distance_metric,
                k,
                target_low,
                target_high,
                learning_rate=learning_rate,
                max_iter=max_iter,
                scoring_function=scoring_function,
                seed=seed,
            ): seed
            for seed in [seed + s for s in range(num_queries)]
        }
        for future in tasks:
            query = future.result()
            queries.append(query)

    return np.vstack(queries).astype(np.float32)


def generate_workload_sgd(
    dataset_input,
    queries_output,
    k,
    target_class,
    num_queries,
    learning_rate=1.0,
    max_steps=1000,
    seed=1234,
    threads=os.cpu_count(),
):
    assert target_class in ["easy", "medium", "hard", "hard+"]

    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train")
    print("loaded dataset with shape", dataset.shape)

    avg_rc = _average_rc(dataset, distance_metric, k)
    if distance_metric == "angular":
        target_rc = {
            "easy": (avg_rc - 1) * 1.5 + 1,
            "medium": (avg_rc - 1) * 1.0 + 1,
            "hard": (avg_rc - 1) / 2 + 1,
            "hard+": (avg_rc - 1) / 4 + 1,
        }[target_class]
    else:
        target_rc = {
            "easy": (avg_rc - 1) / 2 + 1,
            "medium": (avg_rc - 1) / 10 + 1,
            "hard": (avg_rc - 1) / 100 + 1,
            "hard+": (avg_rc - 1) / 500 + 1,
        }[target_class]
    delta = 0.05 * target_rc
    target_low = target_rc - delta
    target_high = target_rc + delta

    queries = []

    with ThreadPoolExecutor(threads) as pool:
        tasks = {
            pool.submit(
                generate_query_sgd,
                dataset,
                distance_metric,
                k,
                target_low,
                target_high,
                learning_rate=learning_rate,
                max_iter=max_steps,
                seed=seed,
            ): seed
            for seed in [seed + s for s in range(num_queries)]
        }
        for future in tasks:
            query = future.result()
            queries.append(query)

    queries = np.vstack(queries).astype(np.float32)

    if queries_output.endswith(".bin"):
        assert np.all(np.isfinite(queries))
        queries.tofile(queries_output)
        check, _ = read_data.read_multiformat(queries_output, "", repair=False)
        assert np.all(np.isclose(check, queries))
    elif queries_output.endswith(".hdf5"):
        write_queries_hdf5(queries, queries_output)
    else:
        raise ValueError(f"Unknown format `{queries_output}`")


def sample_indices(num_queries, dataset, seed):
    """Samples the required number of indices, valid in the given dataset"""
    gen = np.random.default_rng(seed)
    return gen.integers(0, dataset.shape[0] - 1, size=num_queries)


def annealing_measure_convergence(
    dataset_input,
    output,
    k,
    num_queries,
    initial_temperature=1.0,
    max_steps=2000,
    seed=1234,
    data_limit=read_data.MAX_DATA_LEN,
    target_class=None
):
    """Keeps track of the convergence of SGD as well as the running time."""
    import pandas as pd
    import json

    gen = np.random.default_rng(seed)

    results = []

    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train", data_limit=data_limit)

    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)

    metric = "rc"

    # set auto scale
    if distance_metric == "angular":
        scale = 0.1
    else:
        diameter = _estimate_diameter(dataset, index, distance_metric)
        scale = diameter / 100
    logging.info("using automatic scale: %f", scale)

    neighbor_generators = {
        "angular": neighbor_generator_angular(scale, gen),
        "euclidean": neighbor_generator_euclidean(scale, gen),
        "ip": neighbor_generator_ip(scale, gen),
    }
    gen_neighbor = neighbor_generators[distance_metric]

    scoring_functions = {
        "rc": relative_contrast_scorer(index, distance_metric, k),
        "faiss_ivf": faiss_ivf_scorer(index, dataset, distance_metric, k),
    }
    score = scoring_functions[metric]

    score_transforms = {"rc": transform_rc, "faiss_ivf": lambda s: s}
    score_transform = score_transforms[metric]

    if target_class is not None:
        avg_rc = _average_rc(dataset, distance_metric, k)
        if distance_metric == "angular":
            target_rc = {
                "easy": (avg_rc - 1) * 1.5 + 1,
                "medium": (avg_rc - 1) * 1.0 + 1,
                "hard": (avg_rc - 1) / 2 + 1,
            }[target_class]
        else:
            target_rc = {
                "easy": (avg_rc - 1) / 2 + 1,
                "medium": (avg_rc - 1) / 10 + 1,
                "hard": (avg_rc - 1) / 100 + 1,
            }[target_class]
        delta = 0.05 * target_rc
        target_low = target_rc + delta
        target_high = target_rc - delta
    else:
        target_low = 0.0
        target_high = 0.0

    indices = sample_indices(num_queries, dataset, seed)

    for q_idx in indices:
        x = dataset[q_idx]
        _, _, progress = annealing(
            score,
            x,
            gen_neighbor,
            score_transform( target_low ),
            score_transform( target_high ),
            fast_annealing_schedule(initial_temperature),
            max_steps=max_steps,
            return_progress=True,
        )
        progress = pd.DataFrame(progress)
        progress["query_index"] = q_idx
        results.append(progress)

    res = pd.concat(results)
    res["dataset"] = dataset_input
    # res["data_limit"] = data_limit
    res["method"] = "annealing"
    res["method_params"] = json.dumps({
        "initial_temperature": initial_temperature,
        "max_steps": max_steps,
        "seed": seed
    }, sort_keys=True)

    res.to_csv(output, index=False)


# TODO: should SGD start from a point in the dataset?
def sgd_measure_convergence(
    dataset_input,
    output,
    k,
    num_queries,
    learning_rate=1.0,
    max_steps=2000,
    seed=1234,
    data_limit=read_data.MAX_DATA_LEN,
    target_class=None
):
    """Keeps track of the convergence of SGD as well as the running time."""
    import pandas as pd
    import json

    results = []

    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train", data_limit=data_limit)
    if target_class is not None:
        avg_rc = _average_rc(dataset, distance_metric, k)
        if distance_metric == "angular":
            target_rc = {
                "easy": (avg_rc - 1) * 1.5 + 1,
                "medium": (avg_rc - 1) * 1.0 + 1,
                "hard": (avg_rc - 1) / 2 + 1,
            }[target_class]
        else:
            target_rc = {
                "easy": (avg_rc - 1) / 2 + 1,
                "medium": (avg_rc - 1) / 10 + 1,
                "hard": (avg_rc - 1) / 100 + 1,
            }[target_class]
        delta = 0.05 * target_rc
        target_low = target_rc - delta
        target_high = target_rc + delta
    else:
        target_low = 0.0
        target_high = 0.0

    indices = sample_indices(num_queries, dataset, seed)

    for q_idx in indices:
        start = dataset[q_idx]
        _, progress = generate_query_sgd(
            dataset,
            distance_metric,
            k,
            target_low,
            target_high, learning_rate,
            max_steps,
            seed=seed + q_idx,
            return_progress=True,
            start_point=start
        )
        progress = pd.DataFrame(progress)
        progress["query_index"] = q_idx
        results.append(progress)

    res = pd.concat(results)
    res["dataset"] = dataset_input
    # res["data_limit"] = data_limit
    res["method"] = "sgd"
    res["method_params"] = json.dumps({
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "seed": seed
    }, sort_keys=True)
    

    res.to_csv(output, index=False)


SEARCH_LT    = -1
SEARCH_FOUND = 0 
SEARCH_GT    = 1


def search_range_by(range, fun):
    lower = range[0]
    upper = range[1]

    cur_res = None
    while lower < upper:
        mid = (lower + upper) / 2
        mid_res = fun(mid)
        if mid_res == SEARCH_FOUND:
            return mid_res
        elif mid_res == SEARCH_LT:
            cur_res = mid_res
            upper = mid
        else:
            cur_res = mid_res
            lower = mid

    return cur_res


def generate_workload_empirical_difficulty(
    dataset_input,
    queries_output,
    k,
    index_name,
    empirical_difficulty_range: tuple[float, float],
    num_queries,
    learning_rate=1.0,
    max_steps=1000,
    seed=1234,
    threads=os.cpu_count(),
):
    """
    Create a workload where the average empirical difficulty
    is within the specified range, for the given index.
    The queries are generated with Hephaestus-Gradient.
    """

    dataset, distance_metric = read_data.read_multiformat(dataset_input, "train")
    print("loaded dataset with shape", dataset.shape)

    exact_index = faiss.IndexFlatL2(dataset.shape[1])
    exact_index.add(dataset)

    if index_name.lower() in ["hnsw", "faiss_hnsw", "faiss-hnsw"]:
        empirical_difficulty_evaluator = metrics.EmpiricalDifficultyHNSW(
            dataset, 0.95, exact_index, distance_metric
        )
    elif index_name.lower() in ["ivf", "faiss_ivf", "faiss-ivf"]:
        empirical_difficulty_evaluator = metrics.EmpiricalDifficultyIVF(
            dataset, 0.95, exact_index, distance_metric
        )
    elif index_name.lower() in ["messi"]:
        empirical_difficulty_evaluator = metrics.EmpiricalDifficultyMESSI(
            dataset
        )
    elif index_name.lower() in ["dstree"]:
        empirical_difficulty_evaluator = metrics.EmpiricalDifficultyDSTree(
            dataset
        )
    else:
        raise ValueError("unknown index `" + index_name + "`")

    queries = generate_queries_sgd(
        dataset,
        distance_metric,
        k,
        num_queries,
        target_low=empirical_difficulty_range[0],
        target_high=empirical_difficulty_range[1],
        learning_rate=learning_rate,
        max_iter=max_steps,
        scoring_function=lambda x: empirical_difficulty_evaluator.evaluate(x, k),
        threads=threads
    )

    if queries_output.endswith(".bin"):
        assert np.all(np.isfinite(queries))
        queries.tofile(queries_output)
        check, _ = read_data.read_multiformat(queries_output, "", repair=False)
        assert np.all(np.isclose(check, queries))
    elif queries_output.endswith(".hdf5"):
        write_queries_hdf5(queries, queries_output)
    else:
        raise ValueError(f"Unknown format `{queries_output}`")



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    path = ".data/text2image-ip-208-10M.bin"
    generate_workload_empirical_difficulty(
        path,
        "/tmp/queries.hdf5",
        empirical_difficulty_range=(0.1,0.2),
        k=10,
        index_name="messi",
        num_queries=1,
        learning_rate=10.0,
        max_steps=1000
    )
