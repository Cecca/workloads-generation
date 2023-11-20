from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def load(rc):
    path = f"res_sift_sift-rc{rc}_10.csv"
    df = pd.read_csv(path)
    df["rc-group"] = str(rc)
    return df


data = pd.concat([
    load(rc)
    for rc in [3, 2, 1.4, 1.1, 1.04, 1.02]
])
averages = data.groupby("rc-group").mean().reset_index()
print(data.corr('kendall'))

sns.scatterplot(
    data = data,
    x = "rc_10",
    y = "distcomp",
    hue = "rc-group"
)
sns.scatterplot(
    data = averages,
    x = "rc_10",
    y = "distcomp",
    color = "black"
)
sns.lineplot(
    data = averages,
    x = "rc_10",
    y = "distcomp",
    color = "black",
    zorder=-1
)
# plt.xscale("log", base=1.01)
plt.savefig("plt.png")

