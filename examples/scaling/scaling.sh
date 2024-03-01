#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=$1
#SBATCH --constraint=gpu
#SBATCH --gpus=$2
#SBATCH --ntasks=$2
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --job-name Scaling_nodes=${1}_gpus=${2}_dimx=${3}_dimy=${4}_dimz=${5}_dimt=${6}_config=${8}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00
#SBATCH --account=m3863_g

nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
# export LD_LIBRARY_PATH=
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0
module load cudnn/8.9.3_cuda12 julia/1.9

srun julia ./examples/scaling/scaling.jl $1 $2 $3 $4 $5 $6 $7 $8

exit 0
EOT
