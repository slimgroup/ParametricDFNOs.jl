using Pkg
Pkg.activate("./")

using DrWatson
using ParametricOperators
using Parameters
using Profile
using Zygote
using PyPlot
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using Random
matplotlib.use("Agg")

println("Loaded Libraries . . .")
gpu = ParametricOperators.gpu
T = Float32

matrix = ParMatrix(T, 10, 10)
fourier = ParDFT(RDT(matrix), Range(matrix))

x = rand(DDT(matrix), Domain(matrix)) |> gpu
y = rand(RDT(fourier), Range(fourier)) |> gpu

println("Init Vectors . . .")

w = init(matrix) |> gpu

println("Init Weights . . .")

# out = fourier * matrix(w) * x
grads = gradient(params -> norm(fourier * matrix(params) * x), w)[1]
println(grads)
