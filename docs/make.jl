using Documenter
using MyJuliaPkg
using Documenter.Remotes


makedocs(
    sitename = "MyJuliaPkg.jl",
    #root = ".",
    #format = Documenter.HTML(prettyurls = false),
    format = Documenter.HTML(),
    #repo = Remotes.GitHub("stefanocovino/MyJuliaPkg.jl"),
    #modules = [MyJuliaPkg]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/stefanocovino/MyJuliaPkg.jl.git")
