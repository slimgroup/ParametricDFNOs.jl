#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --constraint=gpu
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --qos=shared
#SBATCH --account=m3863_g

export SALLOC_ACCOUNT=m3863_g
export SBATCH_ACCOUNT=m3863_g

export DFNO_3D_GPU=1

srun julia-1.8 examples/perlmutter/train.jl
