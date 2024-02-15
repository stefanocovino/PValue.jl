# MyJuliaPkg.jl

Documentation for MyJuliaPkg

A set of sparse julia functions for personal use.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/stefanocovino/MyJuliaPkg.jl.git")
```

will install this package.


```@contents
```

## Functions

```@docs
BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)
```

```@docs
Frequentist_p_value(ssrv,ndata,nvar)
```


```@docs
Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)
```


```@docs
Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)
```


```@docs
RMS(datavec,modvec)
```

```@docs
SSR(modvec,obsvec,errobsvec)
```



## Index

```@index
```

