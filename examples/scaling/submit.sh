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

WEAK_SPATIAL_SAFE_CONFIGURATIONS=(
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
# srun --export=ALL julia-1.8 ./examples/scaling/scaling.jl 1 2 2 2 1 1 256 128 128 1 test

TEST_SCALING_CONFIGURATIONS=(
    # "1 2 2 2 1 1 256 128 128 1"
    # "1 4 4 2 2 1 256 256 128 1"
    "1 2 2 2 1 1 128 64 64 1"
    "1 2 2 2 1 1 128 64 64 2"
    "1 2 2 2 1 1 128 64 64 3"
    "1 2 2 2 1 1 128 64 64 4"
    "1 2 2 2 1 1 128 64 64 5"
    "1 2 2 2 1 1 128 64 64 6"
    "1 2 2 2 1 1 128 64 64 7"
    "1 2 2 2 1 1 128 64 64 8"
    "1 2 2 2 1 1 128 64 64 9"
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
    CONFIGURATIONS=("${WEAK_SPATIAL_SCALING_CONFIGURATIONS[@]}")
elif [[ "$1" == "weak_safe_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_SAFE_CONFIGURATIONS[@]}")
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
    bash examples/scaling/scaling.sh "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}" "${config[9]}" "$1"
done
