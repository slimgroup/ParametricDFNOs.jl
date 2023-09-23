using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

function print_rd(op::Any)
    if rank == 0
        println(Range(op), " x ", Domain(op)," @ Rank ", rank)
    end
end

weight_mix = ParMatrix(Float32, 10, 20) ⊗ ParMatrix(Float32, 20, 30) # ⊗ ParMatrix(Float32, 30, 40)
print_rd(weight_mix)

weight_mix = distribute(weight_mix, [1, 2])
print_rd(weight_mix)

x = rand(DDT(weight_mix), Domain(weight_mix))
θ = init(weight_mix)

network = weight_mix(θ)
println("Root children: ", length(children(network)))

network * x

MPI.Finalize()

# using MPI
# using ParametricOperators

# # Initialize MPI
# MPI.Init()

# # Get the rank and size of the world communicator
# rank = MPI.Comm_rank(MPI.COMM_WORLD)
# size = MPI.Comm_size(MPI.COMM_WORLD)

# # Define a simple operator
# A = ParIdentity(10, 10)

# # Create a ParDistributed operator
# D = ParDistributed(A, rank, size)

# # Apply the operator to a vector
# x = rand(10)
# # y = D(x)

# # Finalize MPI
# MPI.Finalize()

# using ParametricOperators
# using MPI

# MPI.Init()

# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# root = 0

# x = nothing
# if rank == root
#     x = rand(4)
# end
# x = MPI.bcast(x, root, comm)

# A = ParMatrix(Float64, 4, 4)
# A = distribute(A, [1, 2])

# θ = init(A)
# y = A(θ) * x

# MPI.Finalize()

# mpiexecjl --project=./ -n 2 julia test_dist.jl

# First ParCompose constructs the following for each dimension:
#     [Repartition(current_comm) ...Distributed(Lower Identity Dimension), Broadcasted(Operator at the dimension), ...Distributed(Upper Identity Dimension)]
# adds Repartitoin(comm_out) at the end

# now, we init all
# next, call network * x, here is where the kronecker is computer. What is happening here ??