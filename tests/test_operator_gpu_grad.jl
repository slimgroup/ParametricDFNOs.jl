using Pkg
Pkg.activate("./")

using DrWatson
using MPI
using ParametricOperators
using Parameters
using Profile
using Shuffle
using Zygote
using PyPlot
using NNlib
using NNlibCUDA
using FNO4CO2
using JLD2
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
using Random
matplotlib.use("Agg")

cpu = ParametricOperators.cpu
# gpu = ParametricOperators.gpu
update = ParametricOperators.update!

T = Float32

op1 = ParMatrix(T, 20, 20)
# op2 = ParMatrixN(T, 20, 20)
op3 = ParDiagonal(T, 20)
op4 = ParRestriction(Complex{T}, 20, [1:4])
op5 = ParRestriction(Complex{T}, 20, [1:4, 10:12])
op6 = ParDFT(Complex{T}, 20)
op7 = ParDFT(T, 20)
op8 = ParIdentity(T, 20)

ops = [(op1, 1), (op3, 1), (op4, 0), (op5, 0), (op6, 0), (op7, 0), (op8, 0)]
θ = init(op1)
for (op, _) in ops
    init!(op, θ)
end

for (op, type) in ops
    x = rand(DDT(op), Domain(op))
    y = rand(RDT(op), Range(op))
    if type == 1
        grads = gradient(params -> norm(op(params) * x - y)/norm(y), θ)[1]
    elseif type == 2
        grads = gradient(params -> norm(op(params)(x) - y)/norm(y), θ)[1]
    end
end
