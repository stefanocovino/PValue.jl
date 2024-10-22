var documenterSearchIndex = {"docs":
[{"location":"#PValue.jl","page":"PValue.jl","title":"PValue.jl","text":"","category":"section"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"Documentation for PValue.jl","category":"page"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"A set of sparse julia functions for computing p-values in a \"frequentist\" or \"Bayesian\" scenario.","category":"page"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"","category":"page"},{"location":"#Functions","page":"PValue.jl","title":"Functions","text":"","category":"section"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)","category":"page"},{"location":"#PValue.BIC-Tuple{AbstractFloat, Integer, Integer}","page":"PValue.jl","title":"PValue.BIC","text":"BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)\n\nCompute the Bayes Information Criterion.\n\nArguments\n\nlp logarithm of the likelihood.\nndata number of datapoints.\nnvar number of model parameters.\n\nExamples\n\n\nBIC(56.,100,3)\n\n# output\n\n-98.18448944203573\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"Frequentist_p_value(ssrv,ndata,nvar)","category":"page"},{"location":"#PValue.Frequentist_p_value-Tuple{Any, Any, Any}","page":"PValue.jl","title":"PValue.Frequentist_p_value","text":"Frequentist_p_value(ssrv,ndata,nvar)\nFrequentist_p_value(ssrv,ndof)\n\nCompute the 'classic' frequentist p-value.\n\nArguments\n\nssrv SSR, the sum of squared residuals.\nndata number of datapoints.\nnvar number of fit parameters. \nndof number of degrees of freedom\n\n(i.e. ndata-nvar).\n\nExamples\n\n\nFrequentist_p_value(85.3,100,10)\n\n# output\n\n0.620453567577112\n\nFrequentist_p_value(85.3,90)\n\n# output\n\n0.620453567577112\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)","category":"page"},{"location":"#PValue.Gelman_Bayesian_p_value-NTuple{4, Any}","page":"PValue.jl","title":"PValue.Gelman_Bayesian_p_value","text":"Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)\n\nCompute a 'Bayesian p-value' following the recipe by A. Gelman et al., 2013.\n\nArguments\n\nobsvec datapoints.\nerrobsvec datapoint uncertainties.\nmodvecs model vector.\nsimvecs simulated vector.\n\nExplanation\n\nobsvec and errobsvec are length-'m' vectors of datapoints and relative uncertainties.  modvecs is a vector computed by the posterior distribution of parameters (n chains),  e.g. by a MCMC, where each component is a vector of m values computed using the fit function  and each set of parmeters from the posterior distribution. Finally, simvecs is like modevecs  with the addition of the predicted noise, i.e. these are simulated datapoints.\n\nThe routinely essentially compares the SSR (or, in principle, any test statistics) of  each model based on the derived posterior distribution of parameters vs the data and  to SSR computed by simulated data and again the posterior.\n\nExamples\n\nA full example of application of the GelmanBayesianp_value as well as the  LucyBayesianp_value and the Frequentistpvalue is reported in this  Jupyter notebook.\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)","category":"page"},{"location":"#PValue.Lucy_Bayesian_p_value-NTuple{4, Any}","page":"PValue.jl","title":"PValue.Lucy_Bayesian_p_value","text":"Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)\n\nCompute a 'Bayesian p-value' following the recipe by L.B. Lucy, 2016, A&A 588, 19.\n\nArguments\n\nobsvec datapoints.\nerrobsvec datapoint uncertainties.\nnvars number of parameters.\n\nExplanation\n\nobsvec and errobsvec are length-m vectors of datapoints and relative uncertainties.  modvecs is a vector computed by the posterior distribution of parameters (n chains),  e.g. by a MCMC, where each component is a vector of m values computed using the fit  function and each set of parmeters from the posterior distribution. Finally, nvars is  the number of parameters.\n\nThis algorithm relies on the Chi2 distribution as in the 'frequentist' case. Howver  the SSR is not based only on a punt estimate but it is computed by the whole posterior  distribution of parameters.\n\nExamples\n\n\nx = [1,2,3,4,5]\ny = [1.01,1.95,3.05,3.97,5.1]\ney = [0.05,0.1,0.11,0.17,0.2]\n\nf(x;a=1.,b=0.) = a.*x.+b\n\n# Sample from a fake posterior distribution\nch = DataFrame(a=[0.99,0.95,1.01,1.02,1.03], b=[0.,-0.01,0.01,0.02,-0.01])\n\nres = []\nfor i in 1:nrow(ch)\n    push!(res,f(x;a=ch[i,:a],b=ch[i,:b]))\nend\n\nLucy_Bayesian_p_value(res,y,ey,2)\n\n# output\n\n0.7200318895143041\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"RMS(datavec,modvec)","category":"page"},{"location":"#PValue.RMS-Tuple{Any, Any}","page":"PValue.jl","title":"PValue.RMS","text":"RMS(datavec,modvec)\n\nCompute the Root Mean Square value.\n\nArguments\n\ndatavec datapoints.\nmodvec model values.\n\nExamples\n\n\nRMS([1.1,2.2],[1.15,2.15])\n\n# output\n\n0.050000000000000044\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"SSR(modvec,obsvec,errobsvec)","category":"page"},{"location":"#PValue.SSR-Tuple{Any, Any, Any}","page":"PValue.jl","title":"PValue.SSR","text":"SSR(modvec,obsvec,errobsvec)\n\nCompute the Sum of Squared Residuals.\n\nArguments\n\nmodvec model predictions.\nobsvec observed data.\n`errobsvec~ uncertainties. \n\nExamples\n\n\nSSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1])\n\n# output\n\n9.062500000000016\n\n\n\n\n\n","category":"method"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"WeightedArithmeticMean(x,ex)","category":"page"},{"location":"#PValue.WeightedArithmeticMean-Tuple{Any, Any}","page":"PValue.jl","title":"PValue.WeightedArithmeticMean","text":"WeightedArithmeticMean(x,ex)\n\nCompute the Weighted Arithmetic Mean.\n\nArguments\n\nx input vector\nex uncertainties.\n\nExamples\n\n\nx = [1.2,2.2,4.5,3,3.6]\nex = [0.2,0.2,0.5,0.1,0.6]\n\nWeightedArithmeticMean(x,ex)\n\n# output\n\n(2.634301913536499, 0.07986523020975032)\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"PValue.jl","title":"Index","text":"","category":"section"},{"location":"","page":"PValue.jl","title":"PValue.jl","text":"","category":"page"}]
}