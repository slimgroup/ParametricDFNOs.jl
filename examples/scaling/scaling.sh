#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=$1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --job-name Scaling_nodes=${1}_gpus=${2}_ntasks=${3}_px=${4}_py=${5}_pz=${6}_dimx=${7}_dimy=${8}_dimz=${9}_nt=${10}_config=${11}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:20:00
#SBATCH --account=m3863

export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1

srun --ntasks=$3 --export=ALL,LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0 julia-1.8 ./examples/scaling/scaling.jl $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}

exit 0
EOT
