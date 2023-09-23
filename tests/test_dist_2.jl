using MPI
using ParametricOperators

# Initialize MPI
MPI.Init()

# Get the current MPI communicator
comm = MPI.COMM_WORLD

try
    # Define the operators
    A = ParMatrix(Float64, 4, 4)
    B = ParMatrix(Float64, 100, 100)

    # Create a Kronecker product operator
    K = A ⊗ B

    # Distribute the Kronecker product operator
    K_dist = distribute(K, [2, 2])

    # Define the input vector
    input = rand(100)

    # Apply the distributed Kronecker product operator on the input vector
    # output_dist = K_dist(input)

    println(Domain(K_dist))

catch e
    if MPI.Comm_rank(comm) == 0
        println(e)
    end
end
# Finalize MPI
MPI.Finalize()