# pValue.jl

Documentation for pValue.jl

A set of sparse julia functions for computing p-values in a "frequentist" or "Bayesian" scenario.



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

```@docs
WeightedArithmeticMean(x,ex)
```


## Index

```@index
```

