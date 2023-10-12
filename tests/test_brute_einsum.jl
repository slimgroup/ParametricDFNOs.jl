using Pkg
Pkg.activate("./")

using Tullio
using Flux
using CUDA, CUDAKernels, KernelAbstractions

mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);

cu(A * B) ≈ mul(cu(A), cu(B)) # true
# mul(A |> gpu, B |> gpu)
