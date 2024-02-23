using Documenter
using MyJuliaPkg
using Documenter.Remotes


makedocs(
    sitename = "MyJuliaPkg.jl",
    #format = Documenter.HTML(prettyurls = false),
    #format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true")
    format = Documenter.HTML(),
    #pages=["Home" => "index.md",],
    #modules = [MyJuliaPkg],
    authors="Stefano Covino",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/stefanocovino/MyJuliaPkg.jl", devbranch = "main")
#