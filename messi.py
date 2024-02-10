import os
import read_data as rd
import subprocess as sp
from icecream import ic

ISAX_BIN = os.environ.get(
    "MESSI_ISAX_BIN", "/home/qwang/projects/isax-modularized/build/isax"
)


def _run_messi(
    data,
    queries,
    k,
    database_size,
    query_size,
    series_length,
    sax_length,
    sax_cardinality,
    leaf_size,
    cpu_cores,
    log_filepath,
):
    args = [
        ISAX_BIN,
        "--database_filepath",
        data,
        "--query_filepath",
        queries,
        "--database_size",
        database_size,
        "--query_size",
        query_size,
        "--sax_length",
        sax_length,
        "--sax_cardinality",
        sax_cardinality,
        "--cpu_cores",
        cpu_cores,
        "--log_filepath",
        log_filepath,
        "--series_length",
        series_length,
        "--adhoc_breakpoints",
        "--leaf_size",
        leaf_size,
        "--k",
        k,
        "--with_id",
    ]
    sp.run(list(map(str, args)))

    counters = list()
    with open(log_filepath) as fp:
        for line in fp.readlines():
            if "l2square" in line and "query_engine.c" in line:
                toks = line.split()
                counts = {
                    "q_idx": int(toks[5]),
                    "l2square": int(toks[7]),
                    "sum2sax": int(toks[10]),
                }
                counters.append(counts)
    return counters


def messi_stats(data, queries, k, sax_length, sax_cardinality, leaf_size, cpu_cores):
    """Collect statistics on a MESSI with the given parameters"""
    database_size, series_length = rd.parse_filename(data)
    queries_size, queries_series_length = rd.parse_filename(queries)
    assert series_length, queries_series_length
    log = "/tmp/messi.log"
    if os.path.exists(log):
        os.remove(log)

    res = _run_messi(
        data,
        queries,
        k,
        database_size,
        queries_size,
        series_length,
        sax_length,
        sax_cardinality,
        leaf_size,
        cpu_cores,
        log_filepath=log,
    )

    os.remove(log)
    return res


if __name__ == "__main__":
    import pandas as pd

    DATA = ".data/sald-128-1000000.bin"
    QUERIES = r".data/generated/dataset~sald-128-1000000/k~10/num_queries~5/difficulty~faiss_ivf/target_lower~0.04/target_upper~0.060000000000000005/scale~10/initial_temperature~10.hdf5"
    TMP = "/tmp/sald-128-5.bin"

    rd.hdf5_to_bin(QUERIES, TMP, "test")

    res = messi_stats(
        DATA,
        TMP,
        k=1,
        sax_length=16,
        sax_cardinality=8,
        leaf_size=10000,
        cpu_cores=4,
    )
    print(pd.DataFrame(res))
