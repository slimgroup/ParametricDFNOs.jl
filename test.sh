#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --qos=regular
#SBATCH --account=m3863

export SALLOC_ACCOUNT=m3863
export SBATCH_ACCOUNT=m3863

export DFNO_3D_GPU=0

srun -n 1 julia-1.8 main.jl
