#!/bin/bash

dataset="sald"

queries=("sald-noise-1" "sald-noise-10" "sald-noise-30" "sald-noise-50")

for ((i=0; i<${#queries[@]}; i++))
do
    par1="${dataset}"
    par2="${queries[i]}"
    echo "Running command with par1=${par1} and par2=${par2}"
    python /mnt/hddhelp/ts_benchmarks/metrics.py ${par1} ${par2} 100 0 0.5 1
    echo "Command completed"
done