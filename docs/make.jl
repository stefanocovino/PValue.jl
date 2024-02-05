using Documenter
using MyJuliaPkg
using Documenter.Remotes

makedocs(
    sitename = "MyJuliaPkg",
    format = Documenter.HTML(),
    #modules = [MyJuliaPkg]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#deploydocs(
#    repo = "github.com/stefanocovino/MyJuliaPkg.git",
#    versions = nothing
#)
