using Documenter
using pValue
using Documenter.Remotes


makedocs(
    sitename = "pValue.jl",
    format = Documenter.HTML(prettyurls = true),
    #pages=["Home" => "index.md",],
    #modules = [p-Value],
    authors="Stefano Covino",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/stefanocovino/pValue.jl", devbranch = "main")
#
