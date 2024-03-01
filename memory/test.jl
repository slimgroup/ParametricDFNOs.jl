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

MPI.Finalize()
