import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from icecream import ic
from generate import generate_query_sgd


def simulate(
    n=30,
    k=1,
    outpath="example.png",
    seed=1234,
    with_legend=False,
    with_generate=False,
    target=1.4,
    delta=0.1,
    start=None,
    annotate=None,
):
    assert k > 0
    gen = np.random.default_rng(seed)
    xs = gen.uniform(0, 10, n)
    ys = gen.uniform(0, 10, n)

    points = np.vstack((xs, ys)).T

    padding = 1
    min_x, max_x = np.min(points[:, 0]) - padding, np.max(points[:, 0]) + padding
    min_y, max_y = np.min(points[:, 1]) - padding, np.max(points[:, 1]) + padding
    num_samples = 500

    grid_x = np.linspace(min_x, max_x, num_samples)
    grid_y = np.linspace(min_y, max_y, num_samples)

    rcs = np.zeros((num_samples, num_samples))

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            q = np.array([x, y])
            dists = np.linalg.norm(points - q, axis=1)
            dists = np.sort(dists)
            nearest = dists[k - 1]  # .min()
            rc = dists.mean() / nearest
            assert rc > 1
            rcs[i, j] = rc

    plt.figure(figsize=(5, 5))
    plt.imshow(
        rcs.T, norm=LogNorm(), origin="lower", extent=(min_x, max_x, min_y, max_y)
    )
    plt.scatter(xs, ys, c="white", edgecolor="black", linewidth=2, s=20, zorder=100)
    plt.gca().set_aspect("equal")

    if with_legend:
        plt.colorbar(
            ax=plt.gca(),
            norm=LogNorm(),
            values=np.linspace(rcs.min(), rcs.max(), 100),
            # ticks = [1, 100, 200, 300, 400, 500, 600]
        )
    if with_generate:
        if start is None:
            start = np.array(
                [
                    min_x + (max_x - min_x) / 2,
                    min_y + (max_y - min_y) / 2,
                ]
            )
            ic(start)

        path = generate_query_sgd(
            points,
            "euclidean",
            k=k,
            target_low=target - delta,
            target_high=target + delta,
            return_intermediate=True,
            start_point=start, 
            learning_rate=0.4,
            max_iter=3000,
            seed=1458,
        )
        q = path[-1]
        dists = np.linalg.norm(points - q, axis=1)
        dists = np.sort(dists)
        nearest = dists[k - 1]  # .min()
        rc = dists.mean() / nearest
        ic(path.shape)
        ic(rc)
        plt.scatter(path[-1, 0], path[-1, 1], c="white")
        # plt.scatter(path[1:, 0], path[1:, 1], c="white", s=5)
        plt.plot(path[:, 0], path[:, 1], c="white")

    if annotate is not None:
        dists = np.linalg.norm(points - np.array(annotate), axis=1)
        dists = np.sort(dists)
        nearest = dists[k - 1]  # .min()
        rc = dists.mean() / nearest
        # ic(rc)
        if rc > 2:
            color="white"
        else:
            color="black"
        plt.scatter(annotate[0], annotate[1], c=color, marker="x", s=100)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(outpath)


# simulate(outpath="example-k1.png", with_generate=True, target=8)
# simulate(k=10, outpath="example-k10.png", with_generate=True, target=1.04, delta=0.01)
marker = (3.47, 2.72)
# simulate(
#     outpath="example-k1-nopath.png", with_generate=False, target=8, annotate=marker
# )
# simulate(
#     k=10,
#     outpath="example-k10-nopath.png",
#     with_generate=False,
#     target=1.04,
#     delta=0.01,
#     annotate=marker,
# )

target = 2
delta = 0.1
start = (3.2, 7)
simulate(
    outpath="example-k1-with-path.png",
    with_generate=True,
    target=8,
    # target=target,
    # delta=delta,
    # start = start,
    annotate=marker,
)
simulate(
    k=10,
    outpath="example-k10-with-path.png",
    with_generate=True,
    # target = target,
    # delta = delta,
    target=1.04,
    delta=0.01,
    # start = start,
    annotate=marker,
)
