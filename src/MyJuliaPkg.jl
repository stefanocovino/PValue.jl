module MyJuliaPkg


export BIC
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
