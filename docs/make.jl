using IntroML
using Documenter
using Pluto

const DOCS_PATH = @__DIR__
const SRC_PATH = joinpath(DOCS_PATH, "src")
const NB_PATH = joinpath(dirname(DOCS_PATH), "notebooks")

function build_notebook(nbpath, htmlpath)
    @info "Building notebook at $nbpath to HTML file at $htmlpath"
    s = Pluto.ServerSession()
    nb = Pluto.SessionActions.open(s, nbpath; run_async=false)
    write(htmlpath, Pluto.generate_html(nb))
    return htmlpath
end

# build Pluto notebooks
for fn in readdir(NB_PATH)
    nbpath = joinpath(NB_PATH, fn)
    htmlpath = joinpath(SRC_PATH, first(splitext(fn)) * ".html")
    build_notebook(nbpath, htmlpath)
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
