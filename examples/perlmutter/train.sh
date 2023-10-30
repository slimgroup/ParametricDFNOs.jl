#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --job-name Test_Run_10_Epochs_20_cube 
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:20:00
#SBATCH --account=m3863_g

# # OpenMP settings: (TODO: Figure this out)
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

export DFNO_3D_GPU=1

# Applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 64 -c 2 --cpu_bind=cores julia-1.8 ./train.jl
