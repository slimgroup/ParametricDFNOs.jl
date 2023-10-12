# mpiexecjl --project=./ -n 4 julia tests/test_reduce.jl

using Pkg
Pkg.activate("./")

using MPI

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Let's assume each process has an array of length 5
    local_array = [1, 2, 3, 4, 5]  # This will fill the array with the rank of the process

    # Reduce the arrays from all processes
    global_array = MPI.Reduce(local_array, MPI.SUM, 0, comm)

    # Only the root process (rank 0) will have the reduced result
    if rank == 0
        println("Reduced array: ", global_array)
    end

    MPI.Finalize()
end

main()
