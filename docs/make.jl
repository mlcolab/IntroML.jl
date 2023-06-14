using IntroML
using Documenter
using PlutoSliderServer

const DOCS_PATH = @__DIR__
const SRC_PATH = joinpath(DOCS_PATH, "src")
const NB_PATH = joinpath(dirname(DOCS_PATH), "notebooks")

function build_notebook(nbpath, outpath)
    htmlpath = joinpath(outpath, first(splitext(nbpath)) * ".html")
    @info "Building notebook at $nbpath to HTML file at $htmlpath"
    PlutoSliderServer.export_notebook(
        nbpath;
        Export_output_dir=outpath,
        Precompute_enabled=true,
        Precompute_max_filesize_per_group=1e9,
        Export_create_index=false,
    )
    isfile(htmlpath) ||
        @warn "Failed to build notebook at $nbpath to HTML file at $htmlpath"
    return htmlpath
end

# build Pluto notebooks
for fn in readdir(NB_PATH)
    nbpath = joinpath(NB_PATH, fn)
    build_notebook(nbpath, SRC_PATH)
end

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
