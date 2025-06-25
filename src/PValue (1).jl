module PValue


using AbstractFFTs
using DataFrames
using Distributions
using FFTW
using LinearAlgebra
using Statistics
using StatsBase


export ACF_EK
export BIC
export FourierPeriodogram
export Frequentist_p_value
export Gelman_Bayesian_p_value
export GetACF
export GetCrossCorr
export GetPACF
export Lucy_Bayesian_p_value
export RMS
export SigmaClip
export SSR
export WeightedArithmeticMean
export Z2N



"""
  ACF_EK(t::Vector{Float64},y::Vector{Float64},dy::Vector{Float64}; bins::Int=20)
  
Compute the auto-correlation function for data irregularly sampled, following the algorithm by [Edelson & Krolik (1988)](https://ui.adsabs.harvard.edu/abs/1988ApJ...333..646E/abstract). This a porting of the `python` [astroML version](https://github.com/astroML/astroML/tree/main/astroML/time_series).


# Arguments
- `t` is the vector of observing times.
- `y` is the vector of fluxes.
- `ey` is the vector of the uncertainties of the fluxes.



Returns a tuple of three value: the ACF, the ACF uncertainties, and the bin limits.

# Example
```julia
t = [1.1,2.3,3.2,4.5]
y = [1.,2.,0.5,0.3]
ey = 0.1 .* y

ACF_EK(t,y,ey,bins=2)
([0.6817802730844681, 0.2530024508265458], [0.4472135954999579, 0.5773502691896257], -3.4:3.40000000005:3.4000000001)

```
"""
function ACF_EK(t::Vector{Float64},y::Vector{Float64},dy::Vector{Float64}; bins::Int=20)
  #
  if length(y) != length(t)
      throw(ArgumentError("shapes of t and y must match"))
  end
  #
  if ndims(t) != 1
      throw(ArgumentError("t should be a 1-dimensional array"))
  end
  #
  if length(dy) != length(y) && length(dy) != 1
      throw(ArgumentError("shapes of y and dy must match or dy should be a single number"))
  end
  #
  if length(y) != length(bins) && length(bins) != 1
      throw(ArgumentError("shapes of y and bins must match or bins should be a single number"))
  end
  #
  if length(dy) == 1
      dy = dy .* ones(lengt(y))
  end
  # compute mean and standard deviation of y
  w = 1 ./ (dy .* dy)

  w = w / sum(w)

  mu = dot(w,y)
  sigma = std(y)

  dy2 = reshape(dy,(1,length(dy)))

  dt = (t .- reshape(t,(1,length(t))))'

  a1 = (y .- mu) * reshape(y .- mu,(1,length(y)))
  a2 = sqrt.((sigma.^2 .- dy.^2) * (sigma.^2 .- dy2.^2))

  UDCF = a1 ./ a2

  # determine binning
  if length(bins) == 1
      dt_min = minimum(dt)
      dt_max = maximum(dt)
      bins = range(start=dt_min,stop=dt_max+1e-10,length=bins+1)
  end

  ACF = zeros(length(bins)-1)
  M = zeros(length(bins)-1)

  for i in 1:(length(bins) - 1)
      flag = (dt .>= bins[i]) .& (dt .< bins[i + 1])
      M[i] = sum(flag)
      ACF[i] = sum(UDCF[flag])
  end

  ACF = ACF ./ M

  return ACF, sqrt.(2 ./ M), bins
end




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
    FourierPeriodogram(signal,fs;zerofreq=true)

Compute the discrete Fourier periodogram for the inpout signal.


# Arguments

- `signal` array of input data.
- `fs` sampling in frequency of the input data (1/dt).
- 'zerofreq' is true (false) to (not) include the zero frequency in the output.

Outputs are two arrays: the frequencies and the powers.


# Examples
```jldoctest

FourierPeriodogram([1.,2.,3.,4.],1.)

# output

([0.0, 0.25], [100.0, 8.000000000000002])
```
"""
function FourierPeriodogram(signal,fs;zerofreq=true)
    N = length(signal)
    freqs = fftfreq(N,fs)
    if zerofreq
        positive = freqs .>= 0
    else
        positive = freqs .> 0
    end
    ft = fft(signal)
    powers = abs.(ft).^2
    return freqs[positive], powers[positive]
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

Compute the [AutoCorrelation Function[(https://en.wikipedia.org/wiki/Autocorrelation) for the given lags. It returns a dictionary with the ACF and the minimum and maximum uncertainties against a white noise hypothesis.

# Arguments
- `data` is the vector of input data.
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

    GetCrossCorr(x::Vector{Float64},y::Vector{Float64},lags::Integer)

Compute the [crosscorrelation [(https://en.wikipedia.org/wiki/Cross-correlation) between the `x` and `y` datasets for the given lags. It returns the cross-correlation values.

# Arguments
- `x` is the first input vector.
- `y` is the second input vector.
- `lags` last lag to be computed.



# Examples
```julia

GetCrossCorr([1.2,2.5,3.5,4.3],[1.5,2.9,3.0,4.1],2)

# output

5-element Vector{Float64}:
 -0.1926156048478174
  0.1658715565267623
  0.9627857395579823
  0.15827215481804718
 -0.15637230439086838
```
"""
function GetCrossCorr(x::Vector{Float64},y::Vector{Float64},lags::Integer)
	cc = StatsBase.crosscor(x, y, -lags:lags; demean=true)
	return cc
end


"""

    GetPACF(data::Vector{Float64},lags::Integer;sigma=1.96)

Compute the [Partial AutoCorrelation Function[(https://en.wikipedia.org/wiki/Partial_autocorrelation_function) for the given lags. It returns a dictionary with the PACF and the minimum and maximum uncertainties.

# Arguments
- `data` is the vector of input data.
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
    SigmaClip(x, ex=ones(size(x)); sigmacutlevel=2)

Sigma-clipping filtering of an input array,

# Arguments

- `x` input array.
- `ex` uncertainties.
- `sigmacutlevel` sigma-clipping level.

It performs a one-iteration sigma clipping and reports a mask to select the
surviving elements in the input arrays or other related arrays.

# Examples
```jldoctest

x = [4.,6.,8.,1.,3.,5.,20.]
mask = SigmaClip(x)
x[mask]

# output

6-element Vector{Float64}:
 4.0
 6.0
 8.0
 1.0
 3.0
 5.0
```
"""
function SigmaClip(x, ex=ones(size(x)); sigmacutlevel=2)
    w = pweights(1 ./ ex.^2)
    m = mean(x,w)
    s = std(x,w)
    #println(m," ",s)
    #
    flt = (m-sigmacutlevel*s .<= x) .& (x .<= m+sigmacutlevel*s)
    return flt
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





"""
    Z2N(freqs, time)

Compute the Rayleigh power spectrum of a time series in a given range of frequencies.

# Arguments

- `freqs` is an array with frequencies in units of 1/[time].
- `time` is an array with the time series where to find a period.
- `harm` is the number of harmonics to be used in the analysis.


# Examples
```jldoctest

Z2N([1.,0.5,0.25], [1.,2.,2.5,3.5,5.])

# output

3-element Vector{Any}:
 0.4
 0.4000000000000002
 0.537258300203048
```
"""
function Z2N(freqs, time; harm=1)
    N = length(time)
    Z2n = []
    for ni in freqs
        aux = 0
        for k in 1:harm
            Phi = mod.(ni .* time,1)
            arg = k .* Phi*2.0*π
            phicos = cos.(arg)
            phisin = sin.(arg)
            aux = aux .+ (sum(phicos)^2 + sum(phisin)^2)
        end
        push!(Z2n,(2.0/N)*aux)
    end
    return Z2n
end





end
