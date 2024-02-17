## Directory structure

```
DATA_DIR = 3 where data can be found
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
