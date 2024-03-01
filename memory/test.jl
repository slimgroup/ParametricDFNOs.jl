using Pkg
Pkg.activate("./")

using MPI
using CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# module load cudnn/8.9.3_cuda12 julia/1.9

# Select GPU based on MPI rank
devices = CUDA.devices()

for d in devices
    println("Device: ", d)
end

CUDA.device!(rank % 4)

println(length(CUDA.devices()))
println(CUDA.device(), " @ ", rank)

MPI.Barrier(comm)

println(rank)

MPI.Barrier(comm)

# CUDA.memory_status()

if rank == 0
    c = CUDA.rand(100000)
end

MPI.Barrier(comm)

CUDA.memory_status()

MPI.Barrier(comm)

# CUDA.versioninfo()
# MPI.has_cuda()
# MPI.identify_implementation()

MPI.Finalize()
