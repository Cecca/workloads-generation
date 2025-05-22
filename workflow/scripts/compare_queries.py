# Compares easy and difficult queries across indices, to see
# if difficult query is equally difficult on all datasets.

from icecream import ic
import polars as pl

def compute(hard=True, nqueries=200):
    data = pl.read_csv(snakemake.input["index_perf"])

    query_ids = (
        data
        .select("dataset", "workload", "query_index")
        .unique()
        .sort("dataset", "workload", "query_index")
        .with_row_index("query_id")
    )

    selected_queries = (
        data
        .join(query_ids, on=["dataset", "workload", "query_index"])
        .with_columns(pl.col("distcomp").rank(descending=hard).over(["dataset", "index_name"]).alias("rank"))
        .filter(pl.col("rank") <= nqueries)
        .select("dataset", "index_name", "query_id")
        .group_by("dataset", "index_name")
        .agg(pl.col("query_id"))
        .sort("dataset", "index_name")
    )

    cross = (
        selected_queries
        .join(selected_queries, on=["dataset", "index_name"], how="cross")
        .filter(pl.col("dataset") == pl.col("dataset_right"))
        # .filter(pl.col("index_name") < pl.col("index_name_right"))
        .select(
            "dataset",
            "index_name",
            "index_name_right",
            jaccard = pl.col("query_id").list.set_intersection(pl.col("query_id_right")).list.len() / pl.col("query_id").list.set_union(pl.col("query_id_right")).list.len()
        )
        .pivot(columns="index_name_right", index=["dataset", "index_name"], values="jaccard")
        .sort("dataset", "index_name")
        # .pivot("index_name_right", index=["dataset", "index_name"], values="jaccard")
    )
    ic(cross)

compute(hard=True)
compute(hard=False)
