module PValue


using DataFrames
using Distributions
using Statistics
using StatsBase


export BIC
export Frequentist_p_value
export Gelman_Bayesian_p_value
export GetACF
export GetPACF
export Lucy_Bayesian_p_value
export RMS
export SSR
export WeightedArithmeticMean


"""

    BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)

Compute the [Bayes Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

# Arguments
- `lp` logarithm of the likelihood.
- `ndata` number of datapoints.
- `nvar` number of model parameters.


# Examples
```jldoctest

BIC(56.,100,3)

# output

-98.18448944203573
```
"""
function BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)
    return -2*lp + nvar*log(ndata)
end



"""

    Frequentist_p_value(ssrv,ndata,nvar)
    Frequentist_p_value(ssrv,ndof)
    
    
Compute the 'classic' frequentist [p-value](https://en.wikipedia.org/wiki/P-value).

# Arguments

- `ssrv` SSR, the sum of squared residuals.
- `ndata` number of datapoints.
- `nvar` number of fit parameters.
- `ndof` number of [degrees of freedom](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics))
(i.e. ``ndata-nvar``).

# Examples

```jldoctest

Frequentist_p_value(85.3,100,10)

# output

0.620453567577112
```
```jldoctest
Frequentist_p_value(85.3,90)

# output

0.620453567577112
```
"""
function Frequentist_p_value(ssrv,ndata,nvar)
    cs = Chisq(ndata-nvar)
    return ccdf(cs,ssrv)
end
function Frequentist_p_value(ssrv,ndof)
    cs = Chisq(ndof)
    return ccdf(cs,ssrv)
end




"""

    Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)

    
Compute a 'Bayesian [p-value](https://en.wikipedia.org/wiki/P-value)' following the recipe by [A. Gelman et al., 2013](http://www.stat.columbia.edu/~gelman/book/).

# Arguments

- `obsvec` datapoints.
- `errobsvec` datapoint uncertainties.
- `modvecs` model vector.
- `simvecs` simulated vector.

### Explanation

`obsvec` and `errobsvec` are length-'m' vectors of datapoints and relative uncertainties.
`modvecs` is a vector computed by the posterior distribution of parameters (`n` chains),
e.g. by a MCMC, where each component is a vector of `m` values computed using the fit function
and each set of parmeters from the posterior distribution. Finally, `simvecs` is like `modevecs`
with the addition of the predicted noise, i.e. these are simulated datapoints.

The routinely essentially compares the SSR (or, in principle, any test statistics) of
each model based on the derived posterior distribution of parameters vs the data and
to SSR computed by simulated data and again the posterior.


# Examples

A full example of application of the **Gelman_Bayesian_p_value** as well as the
**Lucy_Bayesian_p_value** and the **Frequentist_p_value** is reported in this
Jupyter [notebook](https://github.com/stefanocovino/PValue.jl/tree/main/docs/BayesianFitTest.ipynb).


"""
function Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)
    resvec = []
    for i in 1:length(modvecs)
        push!(resvec,SSR(modvecs[i],obsvec,errobsvec))
    end
    simvec = []
    for i in 1:length(simvecs)
        push!(simvec,SSR(modvecs[i],simvecs[i],errobsvec))
    end
    #
    finvec = simvec .> resvec
    ni = 0
    for i in 1:length(finvec)
        if finvec[i]
            ni += 1
        end
    end
    return ni/length(finvec)
end



"""

    GetACF(data::Vector{Float64},lags::Integer;sigma=1.96)

Compute the [AutoCorrelation Function[(https://en.wikipedia.org/wiki/Autocorrelation) for the given lags. It returns a dictionary with the ACF and the minimum and maximum uncertainties.

# Arguments
- `data` logarithm of the likelihood.
- `lags` last lag to be computed.
- `sigma` number of sigmas for the uncertainties.


# Examples
```jldoctest

GetACF([1.2,2.5,3.5,4.3],2)["ACF"]

# output

3-element Vector{Float64}:
  1.0
  0.23928737773637632
 -0.2945971122496507
```
"""
function GetACF(data, lags; sigma=1.96)
  cc = StatsBase.autocor(data,0:lags)
  return Dict("ACF" => cc, "errACFmin" => -1/length(data)-sigma*sqrt(1/length(data)), "errACFmax" => -1/length(data)+sigma*sqrt(1/length(data)))
end



"""

    GetPACF(data::Vector{Float64},lags::Integer;sigma=1.96)

Compute the [Partial AutoCorrelation Function[(https://en.wikipedia.org/wiki/Partial_autocorrelation_function) for the given lags. It returns a dictionary with the PACF and the minimum and maximum uncertainties.

# Arguments
- `data` logarithm of the likelihood.
- `lags` last lag to be computed.
- `sigma` number of sigmas for the uncertainties.


# Examples
```jldoctest

GetPACF([1.2,2.5,3.5,4.3],1)["PACF"]

# output

2-element Vector{Float64}:
 1.0
 0.7819548872180438
```
"""
function GetPACF(data::Vector{Float64}, lags::Integer; sigma=1.96)
  cc = StatsBase.pacf(data,0:lags)
  return Dict("PACF" => cc, "errPACFmin" => -1/length(data)-sigma*sqrt(1/length(data)), "errPACFmax" => -1/length(data)+sigma*sqrt(1/length(data)))
end



"""

    Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)

    
Compute a 'Bayesian [p-value](https://en.wikipedia.org/wiki/P-value)' following the recipe by [L.B. Lucy, 2016, A&A 588, 19](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A..19L/abstract).


# Arguments

- `obsvec` datapoints.
- `errobsvec` datapoint uncertainties.
- `nvars` number of parameters.



### Explanation

`obsvec` and `errobsvec` are length-`m` vectors of datapoints and relative uncertainties.
`modvecs` is a vector computed by the posterior distribution of parameters (`n` chains),
e.g. by a MCMC, where each component is a vector of `m` values computed using the fit
function and each set of parmeters from the posterior distribution. Finally, `nvars` is
the number of parameters.

This algorithm relies on the Chi2 distribution as in the 'frequentist' case. Howver
the SSR is not based only on a punt estimate but it is computed by the whole posterior
distribution of parameters.


# Examples

```jldoctest

x = [1,2,3,4,5]
y = [1.01,1.95,3.05,3.97,5.1]
ey = [0.05,0.1,0.11,0.17,0.2]

f(x;a=1.,b=0.) = a.*x.+b

# Sample from a fake posterior distribution
ch = DataFrame(a=[0.99,0.95,1.01,1.02,1.03], b=[0.,-0.01,0.01,0.02,-0.01])

res = []
for i in 1:nrow(ch)
    push!(res,f(x;a=ch[i,:a],b=ch[i,:b]))
end

Lucy_Bayesian_p_value(res,y,ey,2)

# output

0.7200318895143041
```
"""
function Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)
    resvec = []
    for i in 1:length(modvecs)
        push!(resvec,SSR(modvecs[i],obsvec,errobsvec))
    end
    meanres = mean(resvec)-nvars
    cs = Chisq(length(obsvec)-nvars)
    return ccdf(cs,meanres)
end




"""

    RMS(datavec,modvec)

Compute the [Root Mean Square value](https://en.wikipedia.org/wiki/Root_mean_square).


# Arguments

- `datavec` datapoints.
- `modvec` model values.


# Examples

```jldoctest

RMS([1.1,2.2],[1.15,2.15])

# output

0.050000000000000044
```
"""
function RMS(datavec,modvec)
    sqrt(sum((datavec-modvec).^2)/length(datavec))
end



"""

    SSR(modvec,obsvec,errobsvec)

Compute the [Sum of Squared Residuals](https://en.wikipedia.org/wiki/Residual_sum_of_squares).


# Arguments

- `modvec` model predictions.
- `obsvec` observed data.
- `errobsvec~ uncertainties.


# Examples

```jldoctest

SSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1])

# output

9.062500000000016
```
"""
function SSR(modvec,obsvec,errobsvec)
    sum(((modvec.-obsvec)./errobsvec).^2)
end



"""

    WeightedArithmeticMean(x,ex)

Compute the [Weighted Arithmetic Mean](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean).


# Arguments

- `x` input vector
- `ex` uncertainties.


# Examples

```jldoctest

x = [1.2,2.2,4.5,3,3.6]
ex = [0.2,0.2,0.5,0.1,0.6]

WeightedArithmeticMean(x,ex)

# output

(2.634301913536499, 0.07986523020975032)
```
"""
function WeightedArithmeticMean(x,ex)
    w = 1 ./ ex.^2
    n = sum( x.* w)
    d = sum(w)
    return n/d, sqrt(1/d)
end


end
