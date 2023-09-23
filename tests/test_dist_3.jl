using Pkg
Pkg.activate("./")

using MPI
using ParametricOperators

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

function print_rd(op::Any)
    if rank > -1
        println(Range(op), " x ", Domain(op)," @ Rank ", rank)
    end
end
weight_mix = ParMatrix(2, 2, 0) ⊗ ParMatrix(2, 2, 1)
# print_rd(weight_mix)

weight_mix = distribute(weight_mix, [1, 2])
# print_rd(weight_mix)

θ = init(weight_mix)

x = [1.0, 2.0]
network = weight_mix(θ)

# weights are only initialized on the root worker, look in ParBroadcasted
for (k, v) in θ
    println(k)
end

# println("Root children: ", length(children(network)))
println(network(x, rank))

MPI.Finalize()

# comm = MPI.COMM_WORLD

# comm_in  = MPI.Cart_create(comm, [1, 2, 2, 1])
# if comm_in == MPI.COMM_NULL
#     println("NULL COMM IN")
# end

# color = 0
# comm_0 = MPI.Comm_split(comm, color, MPI.comm_rank(comm))
# comm_out  = MPI.Cart_create(comm_0, [1, 1, 1, 1])
# if comm_out == MPI.COMM_NULL
#     println("NULL COMM OUT")
# end
# MPI.Comm_free(comm_out)