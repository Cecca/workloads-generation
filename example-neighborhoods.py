import pandas as pd
import networkx as nx
import numpy as np
import read_data as rd
from workflow.parameters import workloads
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

data, distance_metric = rd.read_multiformat(".data/glove-100-angular.hdf5", "train")
queries, distance_metric = rd.read_multiformat(".data/glove-100-angular.hdf5", "test")

ROOT = 2**64


def visualize_neighborhood(
    query, data, k, filename, mode="knn", title=None, hmessi=None, layout="spectral"
):
    distances = np.linalg.norm(data - query, axis=1)
    nn_indices = np.argsort(distances)[:k]
    nn_dists = distances[nn_indices]
    farthest = nn_dists[-1]
    rc = distances.mean() / farthest
    sdistances = np.sort(distances)
    exp = sdistances[2*k+1] / sdistances[k]
    ic(rc, exp)

    drawedges = []

    def weightfunc(d):
        return 1 / (d)

    G = nx.Graph()
    for i, d in zip(nn_indices, nn_dists):
        drawedges.append(("root", i))
        G.add_edge("root", i, weight=weightfunc(d))
        neigh_distances = np.linalg.norm(data - data[i], axis=1)
        if mode == "knn":
            further_neighbors = np.argsort(neigh_distances)[1 : k + 1]
        else:
            (further_neighbors,) = np.where(neigh_distances <= farthest)
        for j, d in zip(further_neighbors, neigh_distances[further_neighbors]):
            if j != i and j not in nn_indices:
                drawedges.append((i, j))
                G.add_edge(i, j, weight=weightfunc(d))
        for h in further_neighbors:
            for j in further_neighbors:
                if j > h:
                    d = np.linalg.norm(data[j] - data[h])
                    G.add_edge(h, j, weight=weightfunc(d))

    ic(G.number_of_nodes(), G.number_of_edges())
    plt.figure(figsize=(3,3))
    if title is not None:
        plt.title("%s (RC=%.3f, $\\mathcal{H}_{MESSI}$ = %.2f)" % (title, rc, round(hmessi, 2)))

    if layout == "spectral":
        pos = nx.spectral_layout(G, weight="weight")
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=1234, weight="weight")
    else:
        raise ValueError("unknown layout")
    nx.draw(G, pos, node_size=10, edgelist=drawedges, edge_color="lightgray")
    # nx.draw_networkx_nodes(G, pos, node_size=70, node_color="steelblue", nodelist=allnodes)
    nx.draw_networkx_nodes(G, pos, node_size=70, node_color="red", node_shape="s", nodelist=["root"])
    nx.draw_networkx_nodes(
        G, pos, node_size=50, node_color="orange", node_shape="^", nodelist=list(nn_indices)
    )
    plt.tight_layout(pad=0)
    plt.savefig(filename)

def get_data():
    df = pd.read_csv("results_nefeli/metrics/index-performance.csv")
    df = df[df["index_name"] == "messi"]
    df = df[df["dataset"].str.startswith("glove")]
    df = df[df["workload"].str.startswith("File")]
    df = df[df["k"] == 10]
    df = df[["distcomp", "query_index"]]
    df = df.sort_values("distcomp")

    metrics = pd.read_csv("results_nefeli/metrics/all-metrics.csv")
    metrics = metrics[metrics["dataset"].str.startswith("glove")]
    metrics = metrics[metrics["workload_description"].str.startswith("File")]
    metrics = metrics[["i", "rc_10"]].rename(columns={"i": "query_index"})
    difficult_metrics = metrics[metrics["rc_10"] < 2]
    easy_metrics = metrics[metrics["rc_10"] > 3]
    # selected_queries = pd.concat([difficult_metrics, easy_metrics])[["i", "rc_10"]].rename(columns={"i": "query_index"})
    # ic(selected_queries)

    # df = pd.merge(df, selected_queries)

    difficult = pd.merge( df[df["distcomp"] <= 0.999], difficult_metrics ).iloc[-2:].copy()
    difficult["label"] = "hard"
    ic(difficult)

    easy = pd.merge( df[df["distcomp"] < 0.15], easy_metrics ).iloc[-2:].copy()
    easy["label"] = "easy"

    res = pd.concat([difficult, easy])
    ic(res)
    return res

def main():
    k = 10
    layout = "spring"
    querydata = get_data()

    for _, row in querydata.iterrows():
        qidx = row["query_index"]
        visualize_neighborhood(
            queries[qidx],
            data,
            k,
            title=row["label"],
            hmessi=row["distcomp"],
            filename=f"neighborhood-{qidx}.png",
            mode="knn",
            layout=layout,
        )

main()
