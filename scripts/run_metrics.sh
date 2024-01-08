#!/bin/bash

dataset="sald"

queries=("sald" "sald-noise-10" "sald-noise-30" "sald-noise-50")
ann_datasets=("glove-100" "fashion-mnist" "glove-200" "sift")
glove_noise_queries=("glove-noise-0" "glove-noise-10" "glove-noise-30" "glove-noise-50")

if [ $# -eq 0 ]; then
    for ((i=0; i<${#queries[@]}; i++))
    do
        par1="${dataset}"
        par2="${queries[i]}"
        echo "Running command for dataset=${par1} and query=${par2}"
        python /mnt/hddhelp/ts_benchmarks/metrics.py --dataset ${par1} --query ${par2} --k 100 --sample 0.001 --data_limit 1000000 --query_limit 1000
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
        if [ $1 == "glove_noise" ]; then
            for ((i=0; i<${#glove_noise_queries[@]}; i++))
            do
                par1="glove-100-bin"
                par2="${glove_noise_queries[i]}"
                echo "Running command for ${par1}"
                python /mnt/hddhelp/ts_benchmarks/metrics.py --dataset ${par1} --query ${par2} --k 100 --sample 0.001 --data_limit 1000000 --query_limit 1000
                echo "Command completed"
            done
        else
            echo "Dataset parameter is unknown."
        fi
    fi
fi