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

x = [1.0, 1.0, 1.0, 1.0]

grads = gradient(params -> norm(rfft(params)), x)[1]
println(grads)

fourier = ParDFT(Float64, 4)

grads = gradient(params -> norm(fourier(params)), x)[1]
println(grads)
