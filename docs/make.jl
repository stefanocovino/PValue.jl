using Documenter
using PValue
using Documenter.Remotes


makedocs(
    sitename = "PValue.jl",
    format = Documenter.HTML(prettyurls = true),
    #pages=["Home" => "index.md",],
    #modules = [PValue],
    authors="Stefano Covino",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/stefanocovino/PValue.jl", devbranch = "main")
#
