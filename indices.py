from cache import MEM
import faiss
import numpy as np
from threading import Lock


# This lock protects the accesses to faiss global performance counters
FAISS_LOCK = Lock()


@MEM.cache
def build_faiss_ivf(dataset, n_list=None):
    if n_list is None:
        n_list = int(np.ceil(np.sqrt(dataset.shape[0])))
    quantizer = faiss.IndexFlatL2(dataset.shape[1])
    index = faiss.IndexIVFFlat(quantizer, dataset.shape[1], n_list, faiss.METRIC_L2)
    index.train(dataset)
    index.add(dataset)
    return index


@MEM.cache
def build_faiss_hnsw(dataset, index_params="HNSW32"):
    index = faiss.index_factory(dataset.shape[1], index_params)
    index.train(dataset)
    index.add(dataset)
    return index
