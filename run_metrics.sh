#!/bin/bash

dataset="sald"

queries=("sald-noise-1" "sald-noise-10" "sald-noise-30" "sald-noise-50")
ann_datasets=("glove-100" "fashion-mnist" "glove-200" "sift")

if [ $# -eq 0 ]; then
    for ((i=0; i<${#queries[@]}; i++))
    do
        par1="${dataset}"
        par2="${queries[i]}"
        echo "Running command for dataset=${par1} and query=${par2}"
        python /mnt/hddhelp/ts_benchmarks/metrics.py --dataset ${par1} --query ${par2} --k 100 --sample 0.001 --data_limit 1000000 --query_limit 10000
        echo "Command completed"
    done
else 
    if [ $1 == "ann" ]; then
        for ((i=0; i<${#ann_datasets[@]}; i++))
        do
            par1="${ann_datasets[i]}"
            echo "Running command for ${par1}"
            python /mnt/hddhelp/ts_benchmarks/metrics.py --dataset ${par1} --query ${par1} --k 100 --sample 0.001
            echo "Command completed"
        done
    else
        echo "Dataset parameter is unknown."
    fi
fi