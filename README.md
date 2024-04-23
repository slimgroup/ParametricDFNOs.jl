# ParametricDFNOs.jl

[![][license-img]][license-status]
[![Documenter](https://github.com/slimgroup/ParametericDFNOs.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/slimgroup/ParametericDFNOs.jl/actions/workflows/Documenter.yml)
[![TagBot](https://github.com/slimgroup/ParametericDFNOs.jl/actions/workflows/TagBot.yml/badge.svg)](https://github.com/slimgroup/ParametericDFNOs.jl/actions/workflows/TagBot.yml)

<!-- [![][zenodo-img]][zenodo-status] -->

`ParametricDFNOs.jl` is a Julia Language-based scientific library designed to facilitate training Fourier Neural Operators involving large-scale data using [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl). We offer support for distributed 2D and 3D time varying problems.

## Setup

   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/your/environment")
   julia> Pkg.add("ParametricDFNOs")
   ```

This will add `ParametricDFNOs.jl` as dependency to your project

## Documentation

Check out the [Documentation](https://slimgroup.github.io/ParametricDFNOs.jl) for more or get started by running some [examples](https://github.com/turquoisedragon2926/ParametricDFNOs.jl-Examples)!

## Issues

This section will contain common issues and corresponding fixes. Currently, we only provide support for Julia-1.9

## Authors

Richard Rex, [richardr2926@gatech.edu](mailto:richardr2926@gatech.edu) <br/>

[license-status]:LICENSE
<!-- [zenodo-status]:https://doi.org/10.5281/zenodo.6799258 -->
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
<!-- [zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic -->
