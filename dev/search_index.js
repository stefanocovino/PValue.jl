var documenterSearchIndex = {"docs":
[{"location":"#MyJuliaPkg.jl","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"","category":"section"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"Documentation for MyJuliaPkg","category":"page"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"A set of sparse julia functions for personal use.","category":"page"},{"location":"#Installation","page":"MyJuliaPkg.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"using Pkg\nPkg.add(url=\"https://github.com/stefanocovino/MyJuliaPkg.jl.git\")","category":"page"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"will install this package.","category":"page"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"","category":"page"},{"location":"#Functions","page":"MyJuliaPkg.jl","title":"Functions","text":"","category":"section"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)","category":"page"},{"location":"#MyJuliaPkg.BIC-Tuple{AbstractFloat, Integer, Integer}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.BIC","text":"BIC(lp::AbstractFloat,ndata::Integer,nvar::Integer)\n\nComputes the Bayes Information Criterion. 'lp' is the logarithm of the likelihood, 'ndata' the number of datapoints and 'nvar' the number of model parameters.\n\nExamples\n\nBIC(56.,100,3)\n\n# output\n\n-98.18448944203573\n\n\n\n\n\n","category":"method"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"Frequentist_p_value(ssrv,ndata,nvar)","category":"page"},{"location":"#MyJuliaPkg.Frequentist_p_value-Tuple{Any, Any, Any}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.Frequentist_p_value","text":"Frequentist_p_value(ssrv,ndata,nvar)\nFrequentist_p_value(ssrv,ndof)\n\nCompute the 'classic' frequentist p-value. 'ssrv' is the SSR, the sum of squared residuals, 'ndata' the number of datapoints and 'nvar' the number of fit parameters. Else, 'ndof' is the number of degrees of freedom (i.e. ndata-nvar).\n\nExamples\n\nFrequentist_p_value(85.3,100,10)\n\n# output\n\n0.620453567577112\n\nFrequentist_p_value(85.3,90)\n\n# output\n\n0.620453567577112\n\n\n\n\n\n","category":"method"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)","category":"page"},{"location":"#MyJuliaPkg.Gelman_Bayesian_p_value-NTuple{4, Any}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.Gelman_Bayesian_p_value","text":"Gelman_Bayesian_p_value(modvecs,simvecs,obsvec,errobsvec)\n\nCompute a 'Bayesian p-value' following the recipe by A. Gelman et al., 2013.\n\nExplanation\n\n'obsvec' and 'errobsvec' are length-'m' vectors of datapoints and relative uncertainties. 'modvecs' is a vector computed by the posterior distribution of parameters ('n' chains), e.g. by a MCMC, where each component is a vector of 'm' values computed using the fit function and each set of parmeters from the posterior distribution. Finally, 'simvecs' is like 'modevecs' with the addition of the predicted noise, i.e. these are simulated datapoints.\n\nThe routinely essentially compares the SSR (or, in principle, any test statistics) of each model based on the derived posterior distribution of parameters vs the data and to SSR computed by simulated data and again the posterior.\n\nExamples\n\nA full example of application of the GelmanBayesianp_value as well as the LucyBayesianp_value and the Frequentistpvalue is reported in this Jupyter notebook.\n\n.\n\n\n\n\n\n","category":"method"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)","category":"page"},{"location":"#MyJuliaPkg.Lucy_Bayesian_p_value-NTuple{4, Any}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.Lucy_Bayesian_p_value","text":"Lucy_Bayesian_p_value(modvecs,obsvec,errobsvec,nvars)\n\nCompute a 'Bayesian p-value' following the recipe by L.B. Lucy, 2016, A&A 588, 19.\n\nExplanation\n\n'obsvec' and 'errobsvec' are length-'m' vectors of datapoints and relative uncertainties. 'modvecs' is a vector computed by the posterior distribution of parameters ('n' chains), e.g. by a MCMC, where each component is a vector of 'm' values computed using the fit function and each set of parmeters from the posterior distribution. Finally, 'nvars' is the number of parameters.\n\nThis algorithm relies on the Chi2 distribution as in the 'frequentist' case. Howver the SSR is not based only on a punt estimate but it is computed by the whole posterior distribution of parameters.\n\nExamples\n\nx = [1,2,3,4,5]\ny = [1.01,1.95,3.05,3.97,5.1]\ney = [0.05,0.1,0.11,0.17,0.2]\n\nf(x;a=1.,b=0.) = a.*x.+b\n\n# Sample from a fake posterior distribution\nch = DataFrame(a=[0.99,0.95,1.01,1.02,1.03], b=[0.,-0.01,0.01,0.02,-0.01])\n\nres = []\nfor i in 1:nrow(ch)\n    push!(res,f(x;a=ch[i,:a],b=ch[i,:b]))\nend\n\nLucy_Bayesian_p_value(res,y,ey,2)\n\n# output\n\n0.7200318895143041\n\n\n\n\n\n","category":"method"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"RMS(datavec,modvec)","category":"page"},{"location":"#MyJuliaPkg.RMS-Tuple{Any, Any}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.RMS","text":"RMS(datavec,modvec)\n\nComputes the Root Mean Square value. 'datavec' and 'modvec' are vectors formed by datapoints and model values to be compared with.\n\nExamples\n\nRMS([1.1,2.2],[1.15,2.15])\n\n# output\n\n0.050000000000000044\n\n\n\n\n\n","category":"method"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"SSR(modvec,obsvec,errobsvec)","category":"page"},{"location":"#MyJuliaPkg.SSR-Tuple{Any, Any, Any}","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.SSR","text":"SSR(modvec,obsvec,errobsvec)\n\nComputes the Sum of Squared Residuals. 'modvec', 'obsvec' and 'errobsvec' are vectors with the model predictions, observed data and errors, respectively.\n\nExamples\n\nSSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1])\n\n# output\n\n9.062500000000016\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"MyJuliaPkg.jl","title":"Index","text":"","category":"section"},{"location":"","page":"MyJuliaPkg.jl","title":"MyJuliaPkg.jl","text":"","category":"page"}]
}
