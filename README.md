# ParametricDFNOs.jl

This is a branch of the ParametricDFNOs.jl used to work with Tucker decomposed DFNOs.

<!-- [![][zenodo-img]][zenodo-status] -->

`ParametricDFNOs.jl` is a Julia Language-based scientific library designed to facilitate training Fourier Neural Operators involving large-scale data using [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl). We offer support for distributed 2D and 3D time varying problems.

## Setup

   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/your/environment")
   julia> Pkg.add("ParametricDFNOs")
   ```

This will add `ParametricDFNOs.jl` as dependency to your project

## Running Tuker-FNO gradient scaling tests and training
If you have mpiexecjl set up, you can run the following to train a 2D-TFNO

   ```julia
   julia> mpiexecjl --project=./ -n NTASKS julia examples/training/training_2d_tfno.jl
   ```  

   To run the gradient scaling test for 2D-TFNO, run the following  

     ```julia
   julia> mpiexecjl --project=./ -n NTASKS julia examples/scaling/gradient_scaling_2dt.jl
   ```  

## Contact

Srikanth Avasarala, [savasarala9@gatech.edu](mailto:savasarala9@gatech.edu) <br/>

[license-status]:LICENSE
<!-- [zenodo-status]:https://doi.org/10.5281/zenodo.6799258 -->
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
<!-- [zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic -->
