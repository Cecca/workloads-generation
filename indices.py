import os.path
fro cache import MEM
import faiss
import numpy as np
from threading import Lock
from icecream import ic
import subprocess as sp


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


def _find_executable(name):
    import os
    for dirpath, _, filenames in os.walk("."):
        for filename in filenames:
            if filename == name:
                return os.path.join(dirpath, filename)
    return None



class MessiWrapper(object):
    def __init__(
        self,
        data,
        executable=os.path.join(os.getenv("MESSI_INSTALL_DIR"), "bin/MESSI"),
        max_samples=None,
        leaf_size = 2000,
        min_leaf_size = 2000,
        initial_lbl_size = 2000,
        sax_card = 8
    ):
        import read_data as rd
        import logging

        if executable is None or not os.path.isfile(executable):
            # try to find the executable atuomatically
            executable = _find_executable("MESSI")
            print("Using", executable)

        assert os.path.isfile(executable)
        self._executable = executable
        if os.path.isfile(data):
            self._data_path = data
            _, features, samples, _ = rd.parse_filename(self._data_path)
            self._features = features
            self._samples = samples
            if max_samples is not None and max_samples < samples:
                self._samples = samples
        else:
            if max_samples is not None and max_samples < data.shape[0]:
                data = data[:max_samples,:]
            self._data_path = MessiWrapper._write_bin_file(data)
            self._features = data.shape[1]
            self._samples = data.shape[0]

        self.leaf_size = leaf_size
        self.min_leaf_size = min_leaf_size
        self.initial_lbl_size = initial_lbl_size
        self.sax_card = sax_card

        
    @staticmethod
    def _write_bin_file(data):
        """Writes the given data to a binary file, where the file name is the hash of the content"""
        import hashlib
        import os
        import read_data as rd

        key = hashlib.sha256(np.array(data).data.tobytes()).hexdigest()
        path = os.path.join("/tmp", key + ".bin")
        if not os.path.isfile(path):
            rd.write_bin(path, data)
        return path

    def queries_stats(self, queries, k):
        """Run the given queries through the index and collect statistics about the execution"""
        import subprocess as sp

        queries = np.array(queries)
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        qpath = self._write_bin_file(queries)
        q_samples = queries.shape[0]

        proc = sp.run([str(a) for a in [
            self._executable,
            "--dataset", self._data_path,
            "--initial-lbl-size", self.initial_lbl_size,
            "--leaf-size", self.leaf_size,
            "--min-leaf-size", self.min_leaf_size,
            "--queries", qpath,
            "--dataset-size", self._samples,
            "--queries-size", q_samples,
            "--timeseries-size", self._features,
            "--sax-cardinality", self.sax_card,
            "--topk",
            "--k-size", k,
            "--function-type", 3,
            "--cpu-type", 82,
            "--flush-limit", 300000,
            "--read-block", 20000,
            "--queue-number", 2,
            "--in-memory"
        ]], capture_output=True)
        if proc.returncode != 0:
            print(proc.stderr.decode())
            proc.check_returncode()
        output = proc.stdout.decode("utf-8")

        res = []
        for log_line in output.splitlines():
            if not log_line.startswith("query"):
                continue
            tokens = log_line.split()
            q_idx, dists = tuple(map(int, [tokens[1], tokens[3]]))
            distcomp = dists / self._samples 
            res.append((q_idx, distcomp))
        res.sort()
        res = [pair[1] for pair in res]
        res = np.array(res)
        return res


class DSTreeWrapper(object):
    def __init__(
        self,
        data,
        executable=os.path.join(os.getenv("DSTREE_INSTALL_DIR"), "bin/dstree"),
        max_samples=None,
    ):
        import read_data as rd

        if executable is None or not os.path.isfile(executable):
            # try to find the executable atuomatically
            executable = _find_executable("dstree")
            print("Using", executable)

        assert os.path.isfile(executable)
        self._executable = executable
        if os.path.isfile(data):
            self._data_path = data
            _, features, samples, _ = rd.parse_filename(self._data_path)
            self._features = features
            self._samples = samples
            if max_samples is not None and max_samples < samples:
                self._samples = samples
        else:
            if max_samples is not None and max_samples < data.shape[0]:
                data = data[:max_samples,:]
            self._data_path = MessiWrapper._write_bin_file(data)
            self._features = data.shape[1]
            self._samples = data.shape[0]

        self._index_file_name = self._get_index_file_name(data)

        if not os.path.exists(self._index_file_name):
            sp.run([str(a) for a in [self._executable,
                "--dataset", self._data_path,
                "--dataset-size", self._samples,
                "--buffer-size", 19531,
                "--leaf-size", 1000,
                "--index-path", self._index_file_name,
                "--ascii-input", 0,
                "--mode", 0,
                "--timeseries-size", self._features]]).check_returncode()

    @staticmethod
    def _get_index_file_name(data):
        import hashlib
        import os

        if os.path.isfile(data):
            return data + ".dstree"
        else:
            key = hashlib.sha256(np.array(data).data.tobytes()).hexdigest()
            path = os.path.join("/tmp", key + ".dstree")
            return path

    @staticmethod
    def _write_bin_file(data):
        """Writes the given data to a binary file, where the file name is the hash of the content"""
        import hashlib
        import os
        import read_data as rd

        key = hashlib.sha256(np.array(data).data.tobytes()).hexdigest()
        path = os.path.join("/tmp", key + ".bin")
        if not os.path.isfile(path):
            rd.write_bin(path, data)
        return path

    def queries_stats(self, queries, k):
        """Run the given queries through the index and collect statistics about the execution"""
        import subprocess as sp

        queries = np.array(queries)
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        qpath = self._write_bin_file(queries)
        q_samples = queries.shape[0]

        proc = sp.run([str(a) for a in [
            self._executable,
            "--dataset", self._data_path,
            "--dataset-size", self._samples,
            "--queries", qpath,
            "--queries-size", q_samples,
            "--buffer-size", 19531,
            "--leaf-size", 1000,
            "--index-path", self._index_file_name,
            "--ascii-input", 0,
            "--mode", 1,
            "--timeseries-size", self._features,
            "--k", k,
            "--epsilon", 0,
            "--delta", 1,
        ]], capture_output=True)
        if proc.returncode != 0:
            print(proc.stderr.decode())
            proc.check_returncode()
        output = proc.stdout.decode("utf-8")

        res = dict()
        for log_line in output.splitlines():
            if not log_line.startswith("Query_total_loaded_ts_count"):
                continue
            tokens = log_line.split()
            dists, q_idx = tuple(map(int, [tokens[1], tokens[3]]))
            distcomp = dists / self._samples 
            # TODO: ask if it's ok to accumulate counts this way?
            res[q_idx] = distcomp + res.get(q_idx, 0)
        res = sorted(res.items())
        res = [pair[1] for pair in res]
        res = np.array(res)
        return res


if __name__ == "__main__":
    import h5py
    file = ".data/glove-25-angular.hdf5"
    data = h5py.File(file)["train"][:]
    queries = h5py.File(file)["test"][:]
    dstree = DSTreeWrapper(data)
    distcomps = dstree.queries_stats(queries[:100,:], 1)
    ic(distcomps)
    ic(distcomps.mean())


