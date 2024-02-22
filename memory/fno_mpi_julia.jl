# source $HOME/.bash_profile
# mpiexecjl --project=./ -n <number_of_tasks> julia examples/scaling/scaling.jl
# mpiexecjl --project=./ -n 1 julia examples/scaling/gradient_scaling.jl 1 1 1 10 10 10 5

using Pkg
Pkg.activate("./")

# include("../src/models/DFNO_3D/DFNO_3D.jl")
# include("../src/utils.jl")

# using .DFNO_3D
# using .UTILS
using MPI
using Zygote
using DrWatson
using ParametricOperators
using CUDA
using Flux
using LinearAlgebra

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

nx, ny, nz, nt = parse.(Int, ARGS[1:4])
T = Float32

x = rand(T, 5, nx*ny*nz*nt)
y = rand(T, 1, nx*ny*nz*nt)

w1 = ParMatrix(T, 20, 5)
w2 = ParMatrix(T, 128, 20)
w3 = ParMatrix(T, 1, 128)

w1 = distribute(w1)
w2 = distribute(w2)
w3 = distribute(w3)

θ = init(w1)
init!(w2, θ)
init!(w3, θ)

gpu = ParametricOperators.gpu
x = x |> gpu
y = y |> gpu
θ = gpu(θ)

forward(θ, x) = norm(relu.(w3(θ)*relu.(w2(θ)*(w1(θ)*x)))-y)

gradient(θ -> forward(θ, x), θ)

MPI.Finalize()
