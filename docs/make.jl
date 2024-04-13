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
            "2D Time varying" => "examples/2D.md",
            "3D Time varying" => "examples/3D.md",
            "Custom Dataset" => "examples/3D_dataset.md",
            "Checkpoints" => "examples/checkpoints.md",
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
