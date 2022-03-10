using IntroML
using Documenter

DocMeta.setdocmeta!(IntroML, :DocTestSetup, :(using IntroML); recursive=true)

makedocs(;
    modules=[IntroML],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/mlcolab/IntroML.jl/blob/{commit}{path}#{line}",
    sitename="IntroML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mlcolab.github.io/IntroML.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/mlcolab/IntroML.jl", devbranch="main")
