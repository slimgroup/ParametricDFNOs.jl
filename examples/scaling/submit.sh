#!/bin/bash

# (nodes, gpus, ntasks, px, py, pz, dimx, dimy, dimz, dimt)
WEAK_SPATIAL_SCALING_CONFIGURATIONS=(
    "1 1 1 1 1 1 64 64 64 1"
    "1 2 2 2 1 1 128 64 64 1"
    "1 4 4 2 2 1 128 128 64 1"
    "2 8 8 2 2 2 128 128 128 1"
    "4 16 16 4 2 2 256 128 128 1"
    "8 32 32 4 4 2 256 256 128 1"
    "16 64 64 4 4 4 256 256 256 1"
    "32 128 128 8 4 4 512 256 256 1"
    "64 256 256 8 8 4 512 512 256 1"
    "128 512 512 8 8 8 512 512 512 1"
)

WEAK_TEMPORAL_SCALING_CONFIGURATIONS=(
    "1 1 1 1 1 1 64 64 64 10"
    "1 2 2 2 1 1 128 64 64 10"
    "1 4 4 2 2 1 128 128 64 10"
    "2 8 8 2 2 2 128 128 128 10"
    "4 16 16 4 2 2 256 128 128 10"
    "8 32 32 4 4 2 256 256 128 10"
    "16 64 64 4 4 4 256 256 256 10"
    "32 128 128 8 4 4 512 256 256 10"
    "64 256 256 8 8 4 512 512 256 10"
    "128 512 512 8 8 8 512 512 512 10"
)

TEST_SCALING_CONFIGURATIONS=(
    "1 4 4 2 2 1 64 64 32 20"
    "1 4 4 2 2 1 32 32 16 20"
    "1 4 4 2 2 1 16 16 8 20"
    "1 4 4 2 2 1 8 8 4 20"
    "1 4 4 2 2 1 512 512 256 1"
    "1 4 4 2 2 1 256 256 128 1"
    "1 4 4 2 2 1 128 128 64 1"
    "1 4 4 2 2 1 64 64 32 1"
    "1 4 4 2 2 1 32 32 16 1"
    "1 4 4 2 2 1 16 16 8 1"
    "1 4 4 2 2 1 8 8 4 1"
)

if [[ "$1" == "weak_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_SCALING_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_temporal" ]]; then
    CONFIGURATIONS=("${WEAK_TEMPORAL_SCALING_CONFIGURATIONS[@]}")
elif [[ "$1" == "test" ]]; then
    CONFIGURATIONS=("${TEST_SCALING_CONFIGURATIONS[@]}")
else
    echo "Invalid argument. Please specify 'weak_spatial' or 'weak_temporal' or 'test'."
    exit 1
fi

for config_str in "${CONFIGURATIONS[@]}"
do
    config=($config_str)
    bash examples/scaling/scaling.sh "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}" "${config[9]}" "$1"
done
