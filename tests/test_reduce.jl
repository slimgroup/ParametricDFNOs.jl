using Pkg
Pkg.activate("./")
using MPI

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Let's assume each process has an array of length 5
    local_array = fill(rank, 5)  # This will fill the array with the rank of the process
    global_array = zeros(Int, 5)

    # Reduce the arrays from all processes
    MPI.Reduce(local_array, global_array, 0, comm)

    # Only the root process (rank 0) will have the reduced result
    if rank == 0
        println("Reduced array: ", global_array)
    end

    MPI.Finalize()
end

main()
