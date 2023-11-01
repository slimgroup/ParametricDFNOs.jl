#!/bin/bash

# (nodes, gpus, ntasks, px, py, pz, dimx, dimy, dimz, dimt)
WEAK_SCALING_CONFIGURATIONS=(
    # "1 1 1 1 1 1 64 64 64 20"
    # "1 2 2 2 1 1 128 64 64 20"
    "1 4 4 2 2 1 128 128 64 20"
    # "2 8 8 2 2 2 128 128 128 20"
    # "4 16 16 4 2 2 256 128 128 20"
    # "8 32 32 4 4 2 256 256 128 20"
    # "16 64 64 4 4 4 256 256 256 20"
    # "32 128 128 8 4 4 512 256 256 20"
    # "64 256 256 8 8 4 512 512 256 20"
    # "128 512 512 8 8 8 512 512 512 20"
)

STRONG_SCALING_CONFIGURATIONS=(
    "1 1 1 1 1 1 1024 1024 1024 20"
    "1 2 2 2 1 1 1024 1024 1024 20"
    "1 4 4 2 2 1 1024 1024 1024 20"
    "2 8 8 2 2 2 1024 1024 1024 20"
    "4 16 16 4 2 2 1024 1024 1024 20"
    "8 32 32 4 4 2 1024 1024 1024 20"
    "16 64 64 4 4 4 1024 1024 1024 20"
    "32 128 128 8 4 4 1024 1024 1024 20"
    "64 256 256 8 8 4 1024 1024 1024 20"
    "128 512 512 8 8 8 1024 1024 1024 20"
)

if [[ "$1" == "weak" ]]; then
    CONFIGURATIONS=("${WEAK_SCALING_CONFIGURATIONS[@]}")
elif [[ "$1" == "strong" ]]; then
    CONFIGURATIONS=("${STRONG_SCALING_CONFIGURATIONS[@]}")
else
    echo "Invalid argument. Please specify 'weak' or 'strong'."
    exit 1
fi

for config_str in "${CONFIGURATIONS[@]}"
do
    config=($config_str)
    bash examples/scaling/scaling.sh "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}" "${config[9]}" "$1"
done
