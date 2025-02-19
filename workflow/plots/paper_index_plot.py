import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from icecream import ic
import pandas as pd

DIFFICULTY_COLORS = dict(
    zip(["easy", "medium", "hard"],
    reversed(mpl.cm.viridis.resampled(3).colors))
)
DIFFICULTY_COLORS["-"] = "#515151"
ic(DIFFICULTY_COLORS)

SELECTION = ["astro", "deep1b", "glove", "sald"]

k = int(snakemake.wildcards["k"])

# read and clean data
metrics = pd.concat((
    pd.read_csv(snakemake.input[1]),
    pd.read_csv(snakemake.input[4])
))
metrics = metrics[["i", "rc_10", "rc_1", "dataset", "workload_description"]]
metrics = metrics.rename(columns={"workload_description": "workload", "i": "query_index"})
metrics["k"] = np.where(metrics["rc_10"].isna(), 1, 10)
metrics["rc"] = np.where(metrics["rc_10"].isna(), metrics["rc_1"], metrics["rc_10"])

perf_dstree = pd.read_csv(snakemake.input[2])

perf = pd.read_csv(snakemake.input[0])
perf = perf[( perf["index_name"] != "dstree" ) | (perf["workload"].str.startswith("File"))]
perf = pd.concat([perf, perf_dstree,  pd.read_csv(snakemake.input[3])])
perf = perf[perf["dataset"] != "text2image-euclidean-200-10M"]
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
perf = perf[perf["dataset"] != "rw"]

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
perf["index"] = pd.Categorical(
    perf["index"], categories=["MESSI", "DSTree", "IVF", "HNSW"], ordered=True
)
# perf = perf[np.isfinite(perf["rc"])]

perf = perf.groupby(["dataset", "index", "method", "difficulty"], as_index=False)[["distcomp"]].mean()
# perf = perf[perf["rc"] < 10]
perf["distcomp"] = np.minimum(perf["distcomp"], 1.0)
print(perf[( perf["method"] == "GaussianNoise" ) & (perf["index"] == "DSTree")])


def doplot(data, **kwargs):
    sns.barplot(
        data,
        width=0.8,
        gap=0.7,
        palette=DIFFICULTY_COLORS,
        **kwargs
    )
    sns.stripplot(
        data,
        jitter=False,
        dodge=True,
        palette=DIFFICULTY_COLORS,
        linewidth=0.4,
        edgecolor="gray",
        **kwargs
    )


def doscatter(data, **kwargs):
    ax = sns.scatterplot(
        data,
        x="rc",
        y="distcomp",
        hue="method",
        style="difficulty",
        s=100,
        # palette="tab10"
    )
    sns.lineplot(
        data,
        x="rc",
        y="distcomp",
        hue="method",
        # style="method",
        linewidth=1,
        # palette="tab10"
    )
    xmax = data["rc"].max()
    ic(xmax, data["dataset"].unique())
    ticks = [1, round(xmax, 1)]
    # ticks = [1,round(( xmax - 1 )/4 + 1, 1), round(( xmax - 1 )/2 + 1, 1), round(xmax, 1)]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks=ticks, labels=map(str, ticks))
    ax.set_xticks(ticks=[], minor=True)


g = sns.FacetGrid(
    data=perf, col="dataset", row="index", sharex=False, sharey=True, legend_out=True, height=2.2,
    margin_titles=True
)
#g.set(xscale="log")
g.map_dataframe(
    doplot, x="distcomp", y="method", hue="difficulty", #palette="tab10"#, errorbar=None
)
# g.add_legend()
g.savefig(snakemake.output[0])


g = sns.FacetGrid(
    data=perf[perf["dataset"].isin(SELECTION)],
    col="dataset", row="index", sharex=False, sharey=True, legend_out=True, height=2,
    margin_titles=True
)
#g.set(xscale="log")
g.map_dataframe(
    doplot, x="distcomp", y="method", hue="difficulty", #palette="tab10"#, errorbar=None
)
# g.add_legend()
g.savefig(snakemake.output[1])

# perf = pd.merge(perf, metrics, on=["dataset", "workload", "query_index", "k"])

# g = sns.FacetGrid(
#     data=perf, col="dataset", row="index", sharex=False, sharey=True, legend_out=True, height=2.2,
#     margin_titles=True
# )
# g.map_dataframe(
#     doscatter,
#     x="rc",
#     y="distcomp",
#     hue="method",
#     # style="difficulty",
#     size="difficulty",
#     sizes=(100, 100),
#     # palette="tab10"
# )
# g.add_legend()

# g.savefig(snakemake.output[2])
