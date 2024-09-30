import h5py
import os
import numpy as np
import requests
from utils import compute_distances
import sys

# The maximum number of vectors to be read from each file
MAX_DATA_LEN = 5000000

DATA_DIR = os.environ.get("WORKGEN_DATA_DIR", ".data")  # /mnt/hddhelp/workgen_data/
GENERATED_DIR = os.path.join(DATA_DIR, "generated")

dataset_path = "/data/qwang/datasets/"  # TODO: simlinks in /mnt/hddhelp/workgen_data/
sald_noise_path = "/mnt/hddhelp/ts_benchmarks/datasets/sald/"
glove_noise_path = "/mnt/hddhelp/ts_benchmarks/datasets/annbench/glove100/"

DATASETS = {
    "astro": f"{dataset_path}astro-256-100m.bin",
    "deep1b": f"{dataset_path}deep1b-96-100m.bin",
    "f10": f"{dataset_path}f10-256-100m.bin",
    "f5": f"{dataset_path}f5-256-100m.bin",
    "rw": f"{dataset_path}rw-256-100m.bin",
    "seismic": f"{dataset_path}seismic-256-100m.bin",
    "sald": f"{dataset_path}sald-128-100m.bin",
    "fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
    "glove-100": "glove-100-angular.hdf5",
    "glove-25": "glove-25-angular.hdf5",
    "glove-200": "glove-200-angular.hdf5",
    "mnist": "mnist-784-euclidean.hdf5",
    "sift": "sift-128-euclidean.hdf5",
    "glove-100-bin": f"{glove_noise_path}glove-100-1183514-angular.bin",
    "sald-small": f"{DATA_DIR}sald-128-1m.bin",
}

WORKLOADS = {
    "astro": f"{dataset_path}astro-256-1k.bin",
    "deep1b": f"{dataset_path}deep1b-96-1k.bin",
    "f10": f"{dataset_path}f10-256-1k.bin",
    "f5": f"{dataset_path}f5-256-1k.bin",
    "rw": f"{dataset_path}rw-256-1k.bin",
    "seismic": f"{dataset_path}seismic-256-1k.bin",
    "sald": f"{dataset_path}sald-128-1k.bin",
    "sald-noise-50": f"{sald_noise_path}sald-128-1k-hard50p.bin",
    "sald-noise-30": f"{sald_noise_path}sald-128-1k-hard30p.bin",
    "sald-noise-10": f"{sald_noise_path}sald-128-1k-hard10p.bin",
    "fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
    "glove-100": "glove-100-angular.hdf5",
    "glove-25": "glove-25-angular.hdf5",
    "glove-25-lid-25": "/tmp/glove-25-lid-25.hdf5",
    "glove-25-lid-68": "/tmp/glove-25-lid-68.hdf5",
    "glove-200": "glove-200-angular.hdf5",
    "mnist": "mnist-784-euclidean.hdf5",
    "sift": "sift-128-euclidean.hdf5",
    "sift-lid20": "sift-lid20.hdf5",
    "sift-lid50": "sift-lid50.hdf5",
    "sift-lid100": "sift-lid100.hdf5",
    "sift-lid400": "sift-lid400.hdf5",
    "sift-rc2": "sift-rc2.hdf5",
    "sift-rc1.4": "sift-rc1.4.hdf5",
    "sift-rc1.1": "sift-rc1.1.hdf5",
    "sift-rc1.04": "sift-rc1.04.hdf5",
    "sift-rc1.02": "sift-rc1.02.hdf5",
    "sift-rc3": "sift-rc3.hdf5",
    "glove-noise-0": f"{glove_noise_path}glove-100-10k-hard0p-angular.bin",
    "glove-noise-10": f"{glove_noise_path}glove-100-10k-hard10p-angular.bin",
    "glove-noise-30": f"{glove_noise_path}glove-100-10k-hard30p-angular.bin",
    "glove-noise-50": f"{glove_noise_path}glove-100-10k-hard50p-angular.bin",
}


def download(url, local):
    if not os.path.isfile(local):
        print("Downloading from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


def _download_ann_benchmarks(data_path):
    """Downloads a dataset from ann-benchmarks.com"""
    url = f"http://ann-benchmarks.com/{data_path}"
    if not os.path.isfile(data_path):
        print("Downloading from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(data_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


def read_hdf5(filename, what, limit=None):
    with h5py.File(filename) as hfp:
        distance_metric = hfp.attrs.get("distance", None)
        if limit is not None:
            data = hfp[what][:limit]
        else:
            data = hfp[what][:]
    if distance_metric == "angular" and what in ["train", "test"]:
        # discard all-zero vectors (there are some in the nytimes dataset)
        mask = np.where(data.any(axis=1))[0]
        data = data[mask, :]
        norms = np.linalg.norm(data, axis=1)
        assert (norms > 0).all()
        data /= norms[:, np.newaxis]
    return data, distance_metric


def read_from_txt(filename):
    data = np.loadtxt(filename)
    return data


def read_from_bin(filename, sids, sdim):
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape(sids, sdim)


def read_multiformat(name, what, data_limit=MAX_DATA_LEN, repair=True):
    """
    Reads a (data or query) file with the given name. If the name does not correspond to an
    existing file, then a corresponding file is looked in the DATASETS or WORKLOADS dictionaries.

    ## Parameters

    :name: path or name of the data/queryset to load
    :what: whether to load the `train` or `test` set from the file (only
           works for HDF5 files)
    :data_limit: only load the first `data_limit` points from the file
    """
    if not os.path.isfile(name):
        if what == "train":
            path = DATASETS[name]
        elif what == "test":
            path = WORKLOADS[name]
        else:
            raise Exception("`what` can only be one of `train` or `test`")
    else:
        path = name

    if path.endswith(".txt"):
        data = read_from_txt(path)
        if data_limit is not None:
            data = data[:data_limit, :]
        distance_metric = "euclidean"
    elif path.endswith(".hdf5"):
        if not os.path.isfile(path):
            _download_ann_benchmarks(path)
        data, distance_metric = read_hdf5(path, what, data_limit)
    elif path.endswith(".bin"):
        _, data_samples, data_features, distance_metric = parse_filename(path)

        if data_limit is None:
            data_limit = data_samples
        elif data_limit > data_samples:
            data_limit = data_samples

        data = np.fromfile(
            path, dtype="float32", count=data_features * data_limit
        ).reshape(data_limit, data_features)
    else:
        print("Invalid file extension. Supported formats: .txt, .hdf5, .bin")
        sys.exit()

    if repair:
        # replace NaN with
        mask = np.isnan(data)
        nreplace = np.sum(mask)
        if nreplace > 0:
            print("Replaced", nreplace, "NaNs")
        data[mask] = 0.0

    assert np.all(
        np.isfinite(data)
    ), f"Some values are infinite or NaN in file: {path}\n{data}"

    if distance_metric == "angular":
        data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

    print("Loaded", data.shape[0], "vectors in", data.shape[1], "dimensions")

    return data, distance_metric


def hdf5_to_bin(input_path, output_path, what, fname_check=True, with_padding=True):
    assert what in ["train", "test"]
    data, input_distance_metric = read_hdf5(input_path, what)
    if with_padding:
        dim = data.shape[1]
        padsize = (8 - (dim % 8)) % 8
        print("Padding with", padsize, "dimensions")
        if padsize > 0:
            padding = np.zeros((data.shape[0], padsize), dtype=np.float32)
            data = np.hstack([data, padding])
    if fname_check:
        actual_n, actual_dim = data.shape
        _, expected_n, expected_dim, distance_metric = parse_filename(output_path)
        assert (
            actual_dim == expected_dim
        ), f"The output file should be named appropriately, i.e. it should contain the number of dimensions ({actual_dim}) in the filename"
        assert (
            actual_n == expected_n
        ), f"The output file should be named appropriately, i.e. it should contain the number of points ({actual_n}) in the filename"
        assert (
            input_distance_metric == distance_metric
        ), f"The output file should be named appropriately, i.e. it should contain the distance metric ({input_distance_metric}) in the filename"
    data.tofile(output_path)
    # check that the conversion produced the same files
    base, _ = read_multiformat(input_path, what)
    converted, _ = read_multiformat(output_path, what)
    if padsize is not None and padsize > 0:
        converted = converted[:, :-padsize]
    assert np.all(
        np.isclose(base, converted)
    ), "converted file and original do not have the same content"


def str_to_digits(sids_str):
    num_map = {"K": 1000, "M": 1000000, "B": 1000000000}
    sids = 0
    if sids_str.isdigit():
        sids = int(sids_str)
    else:
        sids = int(float(sids_str[:-1]) * num_map.get(sids_str[-1].upper(), 1))
    return sids


def parse_filename(filepath):
    """Parse the filename for metadata.

    Given a *.bin file, parse its filename to extract metadata:

        /some/path/to/name-[distance_metric]-features-samples.bin

    where `distance_metric` defaults to `euclidean` if missing.
    """
    # parse sdim and sids /path/to/file/deep1b-96-1k.bin
    file = filepath.rsplit("/", 1)[-1].split(".")[0]
    file_arr = file.split("-")
    assert len(file_arr) >= 3
    samples = str_to_digits(file_arr[-1])
    features = int(file_arr[-2])
    if len(file_arr) >= 4:
        distance_metric = file_arr[-3]
    else:
        distance_metric = "euclidean"
    name = file_arr[0]

    return name, samples, features, distance_metric
