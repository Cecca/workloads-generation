# Workloads Generation: Experimental Pipeline
This repository provides the experimental setup and evaluation scripts for the paper on generating query workloads with controllable hardness for high-dimensional vector similarity search.
It supports running comprehensive experiments to evaluate various query generation algorithms — Hephaestus-Annealing and Hephaestus-Gradient — designed to produce workloads with specific levels of empirical and explicative hardness. These workloads facilitate the empirical evaluation of index data structures under varying degrees of query difficulty.

To ensure generality, we evaluate performance across four different index types:

- **Exact indexes:** `MESSI` [1], `DSTree` [2]  
- **Approximate indexes:** `HNSW`, `IVF` (implemented via the `FAISS` library [3])

## Implementation

All algorithms are implemented in Python 3.12. We use **JAX** for automatic differentiation in the **Hephaestus-Gradient** algorithm, and the **Adam** optimizer (as implemented in **Optax**) to perform optimization steps. The learning rate is a configurable parameter to allow fine-tuning.

We also provide an easy-to-use standalone Python implementation of our methods, which can be found in this repository.[^4]

[^4]: See the [hephaestus](https://github.com/cecca/hephaestus) repository for the standalone implementation.



## Directory structure

```
DATA_DIR = # where data can be found
GENERATED_DIR = # Where generated workloads should go
RES_DIR = # where results should reside
```

In `RES_DIR` we have

```
$RES_DIR/dataset~NAME/workload_key~SHA/BINPATTERN.bin
```

Where `BINPATTERN` is

```
datasetname-distance-dimensions-n
```

## References

[1]: Botao Peng, Panagiota Fatourou, and Themis Palpanas. 2020. [MESSI: In-Memory
Data Series Indexing.]( https://arxiv.org/abs/2009.00786 ) In ICDE. IEEE, 337–348.

[2]: Yang Wang, Peng Wang, Jian Pei, Wei Wang, and Sheng Huang. 2013. [A Data-
adaptive and Dynamic Segmentation Index for Whole Matching on Time Series.](https://www.vldb.org/pvldb/vol6/p793-wang.pdf)
Proc. VLDB Endow. 6, 10 (2013), 793–804.  

[3]: Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
[The Faiss library.] (https://arxiv.org/abs/2401.08281) CoRR abs/2401.08281 (2024).
