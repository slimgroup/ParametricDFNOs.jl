#!/bin/bash

# (nodes, gpus, ntasks, px, py, pz, dimx, dimy, dimz, dimt)
WEAK_SPATIAL_SCALING_CONFIGURATIONS=(
    "1 1 1 1 1 1 128 128 128 1"
    "1 2 2 2 1 1 256 128 128 1"
    "1 4 4 2 2 1 256 256 128 1"
    "2 8 8 2 2 2 256 256 256 1"
    "4 16 16 4 2 2 512 256 256 1"
    "8 32 32 4 4 2 512 512 256 1"
    "16 64 64 4 4 4 512 512 512 1"
    "32 128 128 8 4 4 1024 512 512 1"
    "64 256 256 8 8 4 1024 1024 512 1"
    "128 512 512 8 8 8 1024 1024 1024 1"
)

WEAK_SPATIAL_FORWARD_CONFIGURATIONS=(
    "1 1 64 64 64 20"
    "1 2 128 64 64 20"
    "1 4 128 128 64 20"
    "2 8 128 128 128 20"
    "4 16 256 128 128 20"
    "8 32 256 256 128 20"
    "16 64 256 256 256 20"
    "32 128 512 256 256 20"
    "64 256 512 512 256 20"
    "128 512 512 512 512 20"
)

WEAK_TEMPORAL_SAFE_CONFIGURATIONS=(
    "1 1 1 1 1 1 32 32 32 2"
    "1 2 2 2 1 1 64 32 32 2"
    "1 4 4 2 2 1 64 64 32 2"
    "2 8 8 2 2 2 64 64 64 2"
    "4 16 16 4 2 2 128 64 64 2"
    "8 32 32 4 4 2 128 128 64 2"
    "16 64 64 4 4 4 128 128 128 2"
    "32 128 128 8 4 4 256 128 128 2"
    "64 256 256 8 8 4 256 256 128 2"
    "128 512 512 8 8 8 256 256 256 2"
)

WEAK_SPATIAL_CONFIGURATIONS=(
    # "1 1 64 64 64 20 4"
    # "1 2 128 64 64 20 4"
    "1 4 128 128 64 10 4"
    "2 8 128 128 128 10 4"
    "4 16 256 128 128 10 4"
    "8 32 256 256 128 10 4"
    "16 64 256 256 256 10 4"
    "32 128 512 256 256 10 4"
    "64 256 512 512 256 10 4"
    "128 512 512 512 512 10 4"
)

WEAK_SPATIAL_SAFE_CONFIGURATIONS=(
    # "1 1 32 64 64 10 0"
    # "1 2 64 64 64 10 0"
    "1 4 128 64 64 10 0"
    "2 8 128 128 64 10 0"
    "4 16 128 128 128 10 0"
    "8 32 256 128 128 10 0"
    "16 64 256 256 128 10 0"
    # "32 128 256 256 256 10 0"
    # "64 256 512 256 256 10 0"
    # "128 512 512 512 256 10 0"
)

# Testing config:
# julia> using MPI
# julia> MPI.MPIPreferences.use_system_binary()
# julia> exit()
# julia>] instantiate

# salloc --nodes=1 --constraint=gpu --gpus=2 --qos=interactive --time=00:20:00 --account=m3863_g --ntasks=2 --gpus-per-task=1
# export PATH=$PATH:$HOME/.julia/bin
# export DFNO_3D_GPU=1
# export SLURM_CPU_BIND="cores"
# export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0
# srun --export=ALL julia-1.8 ./examples/scaling/scaling.jl 2 8 8 2 2 2 128 128 128 1 test_cloud

# 32^3 x 2 per GPU scales easily

TEST_SCALING_CONFIGURATIONS=(
    # "1 1 1 1 1 1 32 32 32 2"
    # "1 2 2 2 1 1 64 32 32 2"
    # "1 4 4 2 2 1 64 64 32 2"
    # "2 8 8 2 2 2 64 64 64 2"
    # "1 2 2 2 1 1 256 128 128 1"
    # "1 4 4 2 2 1 256 256 128 1"
    # "1 2 2 2 1 1 128 64 64 1"
    # "1 2 2 2 1 1 128 64 64 2"
    # "1 2 2 2 1 1 128 64 64 3"
    # "1 2 2 2 1 1 128 64 64 4"
    # "1 2 2 2 1 1 128 64 64 5"
    # "1 2 2 2 1 1 128 64 64 6"
    # "1 2 2 2 1 1 128 64 64 7"
    # "1 2 2 2 1 1 128 64 64 8"
    # "1 2 2 2 1 1 128 64 64 9"
    # "1 2 2 2 1 1 128 64 64 10"
    # "1 2 2 2 1 1 128 64 64 20"
    # "1 4 4 2 2 1 64 64 32 20"
    # "1 4 4 2 2 1 32 32 16 20"
    # "1 4 4 2 2 1 16 16 8 20"
    # "1 4 4 2 2 1 8 8 4 20"
    # "1 4 4 2 2 1 512 512 256 1"
    # "1 4 4 2 2 1 256 256 128 1"
    # "1 4 4 2 2 1 128 128 64 1"
    # "1 4 4 2 2 1 64 64 32 1"
    # "1 4 4 2 2 1 32 32 16 1"
    # "1 4 4 2 2 1 16 16 8 1"
    # "1 4 4 2 2 1 8 8 4 1"
)

if [[ "$1" == "weak_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_old_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_safe_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_SAFE_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_forward" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_FORWARD_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_temporal" ]]; then
    CONFIGURATIONS=("${WEAK_TEMPORAL_SCALING_CONFIGURATIONS[@]}")
elif [[ "$1" == "test" ]]; then
    CONFIGURATIONS=("${TEST_SCALING_CONFIGURATIONS[@]}")
else
    echo "Invalid argument. Please specify 'weak_spatial' or 'weak_temporal' or 'test' or 'weak_safe_spatial'."
    exit 1
fi

for config_str in "${CONFIGURATIONS[@]}"
do
    config=($config_str)
    bash examples/scaling/scaling.sh "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "$1"
done
