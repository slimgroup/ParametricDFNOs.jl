#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --job-name Test_Run_10_Epochs_20_cube 
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=01:30:00
#SBATCH --account=m3863_g

# # OpenMP settings: (TODO: Figure this out)
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1

mpiexecjl --project=./ -n 4 julia-1.8 ./examples/perlmutter/train.jl
