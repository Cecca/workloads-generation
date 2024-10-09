import seaborn as sns
from icecream import ic
import pandas as pd

k = int( snakemake.wildcards["k"] )

# read and clean data
perf = pd.read_csv(snakemake.input[0])
perf = perf[perf["k"] == k]
perf.rename(columns={"index_name": "index"}, inplace=True)
perf.replace("File.*", "Baseline", regex=True, inplace=True)
perf.replace("faiss_ivf", "IVF", regex=True, inplace=True)
perf.replace("faiss_hnsw", "HNSW", regex=True, inplace=True)
perf.replace("fashion_mnist", "fashion-mnist", regex=True, inplace=True)
perf.replace({"dataset": "-.*"}, "", regex=True, inplace=True)
perf["method"] = perf["workload"].str.replace("\\(.*", "", regex=True)
perf["difficulty"] = perf["workload"].str.extract(r"(easy|medium|hard\+|hard)")
perf["difficulty"] = perf["difficulty"].fillna("-")
perf["method"] = perf["method"].str.replace("File", "Baseline")
perf["method"] = perf["method"].str.replace("Annealing", "Hephaestus-Annealing")
perf["method"] = perf["method"].str.replace("SGD", "Hephaestus-Gradient")
perf = perf[perf["difficulty"] != "hard"]
perf["difficulty"] = perf["difficulty"].str.replace("+", "")

perf["difficulty"] = pd.Categorical(
    perf["difficulty"],
    categories=["-", "easy", "medium", "hard"],
    ordered=True
)
perf["method"] = pd.Categorical(
    perf["method"],
    categories=["Baseline", "GaussianNoise", "Hephaestus-Annealing", "Hephaestus-Gradient"],
    ordered=True
)
perf = perf[perf["index"] != "messi_apprx"]
perf = perf[perf["dataset"] != "fashion"]
perf["index"] = pd.Categorical(
    perf["index"],
    categories=["messi", "dstree", "IVF", "HNSW"],
    ordered=True
)


def doplot(data, **kwargs):
    ax = sns.barplot(data, **kwargs)
    #ax.bar_label(ax.containers[0], fontsize=10)


g = sns.FacetGrid(data=perf, col="dataset", row="index", sharex=False)
g.map_dataframe(
    doplot,
    x = "distcomp",
    y = "method",
    hue = "difficulty",
    palette = "tab10",
    errorbar=None
)
g.add_legend(title="difficulty class")
g.savefig(snakemake.output[0])

