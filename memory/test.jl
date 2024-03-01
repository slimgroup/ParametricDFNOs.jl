using Pkg
Pkg.activate("./")

using MPI
using CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Select GPU based on MPI rank
devices = CUDA.devices()
for d in devices
    println(d)
end

println(rank)

CUDA.memory_status()

if rank == 0
    c = CUDA.rand(1000)
end

MPI.Barrier(comm)

CUDA.memory_status()

MPI.Finalize()
