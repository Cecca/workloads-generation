from pathlib import Path
from icecream import ic
import numpy as np
import read_data

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def get_text2image(data_dir, data_out_fname, queries_out_fname, seed=1234, num_queries=1000):
    data_dir = Path(data_dir)
    train_raw_path = data_dir / "text2image-10M.fbin" 
    read_data.download(
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin",
        train_raw_path    
    )
    test_raw_path = data_dir / "text2image-queries-100k.fbin"
    read_data.download(
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
        test_raw_path
    )
    train_raw = read_fbin(train_raw_path)
    test_raw = read_fbin(test_raw_path)

    train_padding = np.zeros((train_raw.shape[0], 7), dtype=np.float32)
    test_padding = np.zeros((test_raw.shape[0], 7), dtype=np.float32)
    train_raw = np.hstack([train_raw, train_padding])
    test_raw = np.hstack([test_raw, test_padding])

    rng = np.random.default_rng(seed)
    test_idxs = rng.choice(np.arange(test_raw.shape[0]), num_queries)
    test_raw = test_raw[test_idxs,:]

    train_norms = np.linalg.norm(train_raw, axis=1)
    train_emb = np.c_[train_raw, np.sqrt(1 - train_norms**2)].astype(np.float32)
    test_emb = np.c_[test_raw, np.zeros(test_raw.shape[0])].astype(np.float32)
    ic(train_emb.shape, test_emb.shape)
    ic(test_emb[0])
    ic(test_emb.dtype)

    _, n, dims, metric = read_data.parse_filename(data_out_fname)
    assert (n, dims) == train_emb.shape
    assert metric == "ip"
    _, n, dims, metric = read_data.parse_filename(queries_out_fname)
    assert (n, dims) == test_emb.shape
    assert metric == "ip"
    assert np.all(test_emb[:,-1] == 0)

    train_emb.tofile(data_out_fname)
    test_emb.tofile(queries_out_fname)


data_dir = Path(snakemake.output["data"]).parent

get_text2image(
    data_dir,
    snakemake.output["data"],
    snakemake.output["queries"]
)
