using Pkg
Pkg.activate("./")

using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

rank == 0 println(MPI.Comm_size(comm))

MPI.Finalize()
