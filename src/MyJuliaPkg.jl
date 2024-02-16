module MyJuliaPkg


using DataFrames
using Distributions
using Statistics


export BIC
export Frequentist_p_value
export Gelman_Bayesian_p_value
export Lucy_Bayesian_p_value
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

    Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)

    
Compute a 'Bayesian [p-value](https://en.wikipedia.org/wiki/P-value)' following the recipe by [A. Gelman et al., 2013](http://www.stat.columbia.edu/~gelman/book/).

### Explanation

'obsvec' and 'errobsvec' are length-'m' vectors of datapoints and relative uncertainties. 'modvecs' is a vector computed by the posterior distribution of parameters ('n' chains), e.g. by a MCMC, where each component is a vector of 'm' values computed using the fit function and each set of parmeters from the posterior distribution. Finally, 'simvecs' is like 'modevecs' with the addition of the predicted noise, i.e. these are simulated datapoints.


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

sim = []
for i in 1:length(res)
    rsim = []
    for j in 1:length(res[i])
        push!(rsim,res[i][j]+rand(Normal(0,ey[j]),1)[1])
    end
    push!(sim,rsim)
end

Gelman_Bayesian_p_value(res,sim,y,ey)

# output

0.8
```
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

    Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)

    
Compute a 'Bayesian [p-value](https://en.wikipedia.org/wiki/P-value)' following the recipe by [L.B. Lucy, 2016, A&A 588, 19](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A..19L/abstract).

### Explanation

'obsvec' and 'errobsvec' are length-'m' vectors of datapoints and relative uncertainties. 'modvecs' is a vector computed by the posterior distribution of parameters ('n' chains), e.g. by a MCMC, where each component is a vector of 'm' values computed using the fit function and each set of parmeters from the posterior distribution. Finally, 'nvars' is the number of parameters.


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
