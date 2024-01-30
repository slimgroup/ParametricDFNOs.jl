#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=$1
#SBATCH --constraint=gpu
#SBATCH --gpus=$2
#SBATCH --ntasks=$2
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --job-name Test_Run_${3}_Epochs_${4}_cube 
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00
#SBATCH --account=m3863_g

# # OpenMP settings: (TODO: Figure this out)
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

srun --export=ALL julia-1.8 ./examples/perlmutter/train.jl $3 $4 $5

exit 0
EOT
