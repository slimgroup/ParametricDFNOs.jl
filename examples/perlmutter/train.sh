#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J Test_Run_10_Epochs_20_cube 
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 00:20:00
#SBATCH -A m3863_g

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 64 -c 2 --cpu_bind=cores -G 4 --gpu-bind=single:1  julia-1.8 /global/homes/r/richardr/code/dfno/examples/perlmutter/train.jl
