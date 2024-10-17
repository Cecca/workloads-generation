import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
import pandas as pd

k = int(10)

# read and clean data
metrics = pd.read_csv(snakemake.input[1])[
    ["i", "rc_10", "rc_1", "dataset", "workload_description"]
]
metrics = metrics.rename(
    columns={"workload_description": "workload", "i": "query_index"}
)
metrics["k"] = np.where(metrics["rc_10"].isna(), 1, 10)
metrics["rc"] = np.where(metrics["rc_10"].isna(), metrics["rc_1"], metrics["rc_10"])

perf_dstree = pd.read_csv(snakemake.input[2])

perf = pd.read_csv(snakemake.input[0])
perf = perf[
    (~perf["workload"].str.startswith("Gaussian")) | (perf["index_name"] != "dstree")
]
perf = pd.concat([perf, perf_dstree])
perf = pd.merge(perf, metrics, on=["dataset", "workload", "query_index", "k"])
perf = perf[perf["k"] == k]
perf.rename(columns={"index_name": "index"}, inplace=True)
perf.replace("File.*", "Baseline", regex=True, inplace=True)
perf.replace("faiss_ivf", "IVF", regex=True, inplace=True)
perf.replace("faiss_hnsw", "HNSW", regex=True, inplace=True)
perf.replace("messi", "MESSI", regex=True, inplace=True)
perf.replace("dstree", "DSTree", regex=True, inplace=True)
perf.replace("fashion_mnist", "fashion-mnist", regex=True, inplace=True)
perf.replace({"dataset": "-.*"}, "", regex=True, inplace=True)
perf["method"] = perf["workload"].str.replace("\\(.*", "", regex=True)
perf["difficulty"] = perf["workload"].str.extract(r"(easy|medium|hard\+|hard)")
perf["difficulty"] = perf["difficulty"].fillna("-")
perf["method"] = perf["method"].str.replace("File", "Baseline")
perf["method"] = perf["method"].str.replace("Annealing", "Hephaestus-Annealing")
perf["method"] = perf["method"].str.replace("SGD", "Hephaestus-Gradient")
perf = perf[perf["difficulty"] != "hard+"]
perf["difficulty"] = perf["difficulty"].str.replace("+", "")

perf["difficulty"] = pd.Categorical(
    perf["difficulty"], categories=["-", "easy", "medium", "hard"], ordered=True
)
perf["method"] = pd.Categorical(
    perf["method"],
    categories=[
        "Baseline",
        "GaussianNoise",
        "Hephaestus-Annealing",
        "Hephaestus-Gradient",
    ],
    ordered=True,
)
perf = perf[perf["index"] != "messi_apprx"]
perf = perf[perf["dataset"] != "fashion"]
# perf["index"] = pd.Categorical(
#     perf["index"], categories=["MESSI", "DSTree", "IVF", "HNSW"], ordered=True
# )
perf = perf[np.isfinite(perf["rc"])]

perf = perf.groupby(["dataset", "index", "method", "difficulty"], as_index=False)[
    ["rc", "distcomp"]
].mean()
# perf = perf[perf["rc"] < 10]
perf["distcomp"] = np.minimum(perf["distcomp"], 1.0)
perf = perf[perf["dataset"] == "astro"]
perf = perf[perf["index"] == "HNSW"]

print(perf)


def doscatter(data, **kwargs):
    ax = sns.scatterplot(
        data,
        x="rc",
        y="distcomp",
        hue="method",
        style="difficulty",
        s=100,
        palette="tab10",
    )
    sns.lineplot(
        data,
        x="rc",
        y="distcomp",
        hue="method",
        # style="method",
        linewidth=1,
        palette="tab10",
    )
    plt.text(
        x=1.1,
        y=0.4,
        s="Hephaestus-Gradient",
        c="tab:red"
    )
    plt.text(
        x=1.27,
        y=0.21,
        s="Hephaestus-Annealing",
        c="tab:green"
    )
    plt.text(
        x=1.28,
        y=0.07,
        s="Baseline",
        c="tab:blue",
        ha="right"
    )
    plt.text(
        x=1.4,
        y=0.1,
        s="GaussianNoise",
        c="tab:orange",
        ha="left",
        fontvariant="small-caps"
    )
    plt.text(
        x=1.08,
        y=0.46,
        s="hard",
        fontstyle="italic"
    )
    plt.text(
        x=1.11,
        y=0.35,
        s="medium",
        fontstyle="italic"
    )
    plt.text(
        x=1.2,
        y=0.25,
        s="easy",
        fontstyle="italic"
    )

    xmax = data["rc"].max()
    ic(xmax, data["dataset"].unique())
    # ticks = [1, round(xmax, 1)]
    ticks = [1,round(( xmax - 1 )/4 + 1, 1), round(( xmax - 1 )/2 + 1, 1), round(xmax, 1)]
    ticks = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
    # ax.set_xscale("log", base=2)
    ax.set_xticks(ticks=ticks, labels=map(str, ticks))
    ax.set_xticks(ticks=[], minor=True)
    ax.set_xlabel("Relative contrast")
    ax.set_xlabel("Empirical hardness")


g = sns.FacetGrid(
    data=perf,
    col="dataset",
    row="index",
    sharex=False,
    sharey=True,
    legend_out=True,
    height=1.3*2.2,
    aspect=2,
    margin_titles=True,
)
g.map_dataframe(
    # sns.scatterplot,
    doscatter,
    x="rc",
    y="distcomp",
    hue="method",
    # style="difficulty",
    size="difficulty",
    sizes=(100, 100),
    palette="tab10",
)
# g.add_legend()

g.set_xlabels("Relative contrast")
g.set_ylabels("Empirical hardness")
g.savefig(snakemake.output[0], dpi=300)
