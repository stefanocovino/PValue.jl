module MyJuliaPkg


using Distributions


export BIC
export Frequentist_p_value
export RMS
export SSR


"""

    BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)

Computes the [Bayes Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion).
'lp' is the logarithm of the likelihood, 'ndata' the number of datapoints and 'nvar' the number of model parameters. 


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
'ssrv' is the SSR, the sum of squared residuals, 'ndata' the number of datapoints and 'nvar' the number of 
fit parameters. Else, 'ndof' is the number of [degrees of freedom](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)) 
(i.e. ndata-nvar).

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

    RMS(datavec,modvec)

Computes the [Root Mean Square value](https://en.wikipedia.org/wiki/Root_mean_square).
'datavec' and 'modvec' are vectors formed by datapoints and model values to be compared with.


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

Computes the [Sum of Squared Residuals](https://en.wikipedia.org/wiki/Residual_sum_of_squares).
'modvec', 'obsvec' and 'errobsvec' are vectors with the model predictions, observed data and errors, respectively. 


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




end
