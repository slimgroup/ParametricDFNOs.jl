using Documenter
using ParametricDFNOs

makedocs(
    sitename = "ParametricDFNOs.jl",
    doctest=false, clean=true,
    authors="Richard Rex",
    format = Documenter.HTML(),
    modules = [ParametricDFNOs],
    pages=[
        "Introduction" => "index.md",
        "Quick Start" => "quickstart.md",
        "Examples" => [
            "2D Forward and Gradient" => "examples/simple_2D.md",
            "2D Training" => "examples/training_2D.md",
            "3D Forward and Gradient" => "examples/simple_3D.md",
            "3D Custom Dataset" => "examples/custom_3D.md",
        ],
        "API" => [
            "2D Time varying" => "api/DFNO_2D.md",
            "3D Time varying" => "api/DFNO_3D.md"],
        "Future Work" => "future.md",
        "Citation" => "citation.md"
    ]
)

# Automatically deploy documentation to gh-pages.
deploydocs(
    repo = "github.com/slimgroup/ParametricDFNOs.jl.git",
    devurl = "dev",
    devbranch = "master",
    versions = ["dev" => "dev", "stable" => "v^"],
)
