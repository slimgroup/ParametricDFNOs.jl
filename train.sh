#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --qos=shared
#SBATCH --account=m3863_g
#SBATCH --constraint=gpu

export SALLOC_ACCOUNT=m3863_g
export SBATCH_ACCOUNT=m3863_g

export DFNO_3D_GPU=1

srun -n 1 -c 2 julia-1.8 main.jl
