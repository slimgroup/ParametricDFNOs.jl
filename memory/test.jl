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

global i = 0
for d in devices
    (i == rank) && CUDA.device!(d)
    global i += 1
end

println(length(CUDA.devices()))

MPI.Barrier(comm)

println(rank)

MPI.Barrier(comm)

CUDA.memory_status()

if rank > 2
    c = CUDA.rand(1000)
end

MPI.Barrier(comm)

CUDA.memory_status()

MPI.Barrier(comm)

CUDA.versioninfo()
MPI.has_cuda()
MPI.identify_implementation()

MPI.Finalize()
