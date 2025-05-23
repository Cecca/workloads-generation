from icecream import ic
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

df = pd.read_csv(snakemake.input["index_perf"])
#df = df[df["distcomp"] <= 0.2]
df = df[df["index_name"] != "messi_apprx"]
df = df[df["dataset"] != "text2image-euclidean-200-10M"]
# df = df[df["dataset"] != ""]
df["dataset"] = df["dataset"].str.replace(".*/", "", regex=True)
df["dataset"] = df["dataset"].str.replace("-.*", "", regex=True)
df = df[df["dataset"] != "fashion_mnist"]
df = df[df["dataset"] != "rw"]
df = df[df["index_name"] == "messi"]
ndatasets = len( df["dataset"].unique() )
n_indices = len( df["index_name"].unique() )
ic(ndatasets, n_indices)

# metrics = pd.read_csv(snakemake.input["metrics"])
# ic(metrics[["dataset", "workload_description", "i", "rc_1"]])

plt.figure(figsize=(6,3))
sns.stripplot(
    data = df,
    x = "distcomp",
    y = "dataset",
    # hue="index_name"
)
plt.xlabel(r"MESSI empirical hardness ($\mathcal{H}_{MESSI}$)")
plt.axvline(0.1, c="lightgray", linestyle="dotted")
plt.axvline(0.2, c="lightgray", linestyle="dotted")
plt.grid(which="major", axis="y")
plt.xlim((0,1))

plt.tight_layout(pad=0)
plt.savefig(snakemake.output[0], dpi=300)
