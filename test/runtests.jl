using AbstractFFTs
using DataFrames
using Distributions
using FFTW
using LinearAlgebra
using PValue
using Random
using Test


@testset "PValue.jl" begin
    # BIC
    @test BIC(56.,100,3) == -98.18448944203573
    #
    # Frequentist_p_value
    @test Frequentist_p_value(85.3,90) == 0.620453567577112
    #
    # Gelman's Bayesian p-value
    x = [1,2,3,4,5]
    y = [1.01,1.95,3.05,3.97,5.1]
    ey = [0.05,0.1,0.11,0.17,0.2]
    f(x;a=1.,b=0.) = a.*x.+b
    Random.seed!(123)
    ch = DataFrame(a=rand(Normal(1,0.1),1000), b=rand(Normal(0.,0.1),1000))
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
    @test Gelman_Bayesian_p_value(res,sim,y,ey) == 0.197
    #
    # Lucy's Bayesian p-value
    x = [1,2,3,4,5]
    y = [1.01,1.95,3.05,3.97,5.1]
    ey = [0.05,0.1,0.11,0.17,0.2]
    f2(x;a=1.,b=0.) = a.*x.+b
    ch = DataFrame(a=[0.99,0.95,1.01,1.02,1.03], b=[0.,-0.01,0.01,0.02,-0.01])
    res = []
    for i in 1:nrow(ch)
        push!(res,f2(x;a=ch[i,:a],b=ch[i,:b]))
    end
    @test Lucy_Bayesian_p_value(res,y,ey,2) == 0.7200318895143041
    #
    # RMS
    @test RMS([1.1,2.2],[1.15,2.15]) == 0.050000000000000044
    #
    # SSR
    @test SSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1]) == 9.062500000000016
    #
    # WeightedArithmeticMean
    wx = [1.2,2.2,4.5,3,3.6]
    wex = [0.2,0.2,0.5,0.1,0.6]
    @test WeightedArithmeticMean(wx,wex) == (2.634301913536499, 0.07986523020975032)
    #
    # GetACF
    @test GetACF(wx,2)["ACF"] == [1.0, 0.04658385093167703, -0.25931677018633537]
    #
    # GetPACF
    @test isapprox(GetPACF(wx,2)["PACF"], [ 1.0, 0.10253110253110313, -0.12812960235640872], atol=1e-10)
    #
    @test Z2N([1.,0.5,0.25], [1.,2.,2.5,3.5,5.]) == [0.4, 0.4000000000000002, 0.537258300203048]
    #
    @test FourierPeriodogram([1.,2.,3.,4.],1.) == ([0.0, 0.25], [100.0, 8.000000000000002])
    #
    t = [1.1,2.3,3.2,4.5]
    y = [1.,2.,0.5,0.3]
    ey = 0.1 .* y
    @test ACF_EK(t,y,ey,bins=2) == ([0.6817802730844681, 0.2530024508265458], [0.4472135954999579, 0.5773502691896257], -3.4:3.40000000005:3.4000000001)
    #
    @test GetCrossCorr([1.2,2.5,3.5,4.3],[1.5,2.9,3.0,4.1],2) == [ -0.1926156048478174, 0.1658715565267623, 0.9627857395579823, 0.15827215481804718, -0.15637230439086838 ]
    #
    x = [4.,6.,8.,1.,3.,5.,20.]
    mask = SigmaClip(x)
    @test x[mask] == [4.,6.,8.,1.,3.,5.]
    #
end
