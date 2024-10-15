from icecream import ic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")

metric = snakemake.wildcards["metric"]

df = pd.read_csv(snakemake.input[0])
df["dataset"] = df["dataset"].str.replace("-.*", "", regex=True)
df = df[df["dataset"] != "fashion_mnist"]

# remove outliers
q = df[metric].quantile(0.99)
df = df[df[metric] < q]

def quantizer(group, column="rc_10"):
    group = group[group[column] >= 0]
    nbins = 100
    try:
        group["binned_col"] = pd.qcut(group[column], nbins)
    except:
        group["binned_col"] = pd.cut(group[column], nbins)
    group = group.groupby("binned_col", observed=True)[[column, "distcomp"]].mean().dropna().reset_index(drop=True)
    return group


def quantize_by(df, column="rc_10"):
    df = df.groupby("dataset").apply(quantizer, column, include_groups=False).reset_index()
    del df["level_1"]
    corrs = df.groupby("dataset").corr("kendall")["distcomp"].reset_index()
    corrs.columns = ["dataset", "metric", "correlation"]
    corrs = corrs[corrs["metric"] != "distcomp"]
    corrs["metric"] = corrs["metric"].replace()
    corrs.set_index("dataset", inplace=True)
    corrs = corrs["correlation"]
    return df, corrs

df, corrs = quantize_by(df, metric)

remapping = {
    "lid_10": "$LID_{10}$",
    "rc_10": "$RC_{10}$",
    "exp_20|10": r"$\operatorname{Expansion}_{20|10}$",
    "eps_0.05": r"$\alpha_{0.05, 10}$",
    "eps_0.1": r"$\alpha_{0.1, 10}$",
    "eps_0.5": r"$\alpha_{0.5, 10}$",
    "eps_1": r"$\alpha_{1, 10}$",
}

g = sns.FacetGrid(df, col="dataset", sharex=True, sharey=True, height=2.2)
g.set(yscale="log")
g.set(xscale="log")
g.set_axis_labels(x_var=remapping[metric],y_var="empirical hardness")
g.map_dataframe(
    sns.scatterplot,
    x = metric,
    y = "distcomp"
)


for (i, j, _), fdata in g.facet_data():
    dataset = fdata["dataset"].unique()[0]
    ax = g.facet_axis(i, j)
    xtext = 1 if metric in ["rc_10", "exp_20|10"] else 0
    ha = "right" if metric in ["rc_10", "exp_20|10"] else "left"
    if j == 0:
        ax.text(-0.3, 1.15, remapping[metric], transform=ax.transAxes, size=15)
    ax.text(xtext, 1, "$\\tau$=%.3f" % (corrs.loc[dataset]),
            size=10,
            transform=ax.transAxes, 
            ha=ha, 
            va="top",
            parse_math=True)
    if metric == "rc_10":
        ticks = [1, 2, 3, 4, 5, 6]
        ax.set_xticks(ticks=ticks, labels=[str(t) for t in ticks])
    if metric == "exp_20|10":
        ticks = [1, 1.1, 1.2, 1.3, 1.4]
        ic(ticks)
        ax.set_xticks(ticks=ticks, labels=[str(t) for t in ticks])
        ax.set_xticks(ticks=[], labels=[], minor=True)

g.tight_layout(pad=0)
g.savefig(snakemake.output[0])

