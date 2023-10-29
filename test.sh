#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --qos=regular
#SBATCH --account=m3863_g

export SALLOC_ACCOUNT=m3863_g
export SBATCH_ACCOUNT=m3863_g

export DFNO_3D_GPU=1

srun -n 1 julia-1.8 main.jl
