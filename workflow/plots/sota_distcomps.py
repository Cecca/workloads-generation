from icecream import ic
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np

df = pd.read_csv(snakemake.input["index_perf"])
df = df[df["k"] == 10]
df = df[df["workload"].str.startswith("File")]
df = df[df["index_name"] != "messi_apprx"]
df["dataset"] = df["dataset"].str.replace(".*/", "", regex=True)
df["dataset"] = df["dataset"].str.replace("-.*", "", regex=True)
df = df[df["dataset"] != "fashion_mnist"]
df = df[df["dataset"] != "rw"]
ndatasets = len( df["dataset"].unique() )
n_indices = len( df["index_name"].unique() )

df["index_name"] = df["index_name"].replace("faiss_ivf", "IVF", regex=True)
df["index_name"] = df["index_name"].replace("faiss_hnsw", "HNSW", regex=True)
df["index_name"] = df["index_name"].replace("messi", "MESSI", regex=True)
df["index_name"] = df["index_name"].replace("dstree", "DSTree", regex=True)

df["index"] = pd.Categorical(
    df["index_name"],
    categories=["MESSI", "DSTree", "IVF", "HNSW"],
    ordered=True
)

def annotations(data, **kwargs):
    ax = plt.gca()
    avgs = data.groupby(kwargs["y"], as_index=False)[[kwargs["x"]]].mean()
    for _, row in avgs.iterrows():
        dataset = row["dataset"]
        distcomp = row["distcomp"]
        distcomp_str = "{:.3f}".format(distcomp)
        ax.text(
            distcomp + 0.08,
            dataset,
            distcomp_str,
            va="center"
        )


# main plot
g = sns.FacetGrid(df, col="index", col_wrap=2, aspect=1.3)
g.map_dataframe(
    sns.barplot,
    y = "dataset",
    x = "distcomp",
)
g.map_dataframe(
    annotations,
    y = "dataset",
    x = "distcomp",
)
g.set_xlabels("fraction of distance computations")
g.savefig(snakemake.output["main"])

