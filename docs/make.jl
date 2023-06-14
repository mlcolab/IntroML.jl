using IntroML
using Documenter
using PlutoSliderServer

const DOCS_PATH = @__DIR__
const SRC_PATH = joinpath(DOCS_PATH, "src")
const NB_PATH = joinpath(dirname(DOCS_PATH), "notebooks")

function build_notebooks(dir_path, out_path)
    PlutoSliderServer.export_directory(
        dir_path;
        Export_output_dir=out_path,
        Precompute_enabled=true,
        Precompute_max_filesize_per_group=1e9,
        Export_create_index=false,
    )
    return out_path
end

# build Pluto notebooks
build_notebooks(NB_PATH, SRC_PATH)

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

deploydocs(; repo="github.com/mlcolab/IntroML.jl", devbranch="main", push_preview=true)
