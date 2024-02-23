using DataFrames
using Distributions
using MyJuliaPkg
using Test

@testset "MyJuliaPkg.jl" begin
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
    @test Gelman_Bayesian_p_value(res,sim,y,ey) == 0.205
    #
    # Lucy's Bayesian p-value
    x = [1,2,3,4,5]
    y = [1.01,1.95,3.05,3.97,5.1]
    ey = [0.05,0.1,0.11,0.17,0.2]
    f(x;a=1.,b=0.) = a.*x.+b
    ch = DataFrame(a=[0.99,0.95,1.01,1.02,1.03], b=[0.,-0.01,0.01,0.02,-0.01])
    res = []
    for i in 1:nrow(ch)
        push!(res,f(x;a=ch[i,:a],b=ch[i,:b]))
    end
    @test Lucy_Bayesian_p_value(res,y,ey,2) == 0.7200318895143041
    #
    # RMS
    @test RMS([1.1,2.2],[1.15,2.15]) == 0.050000000000000044
    #
    # SSR
    @test SSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1]) == 9.062500000000016
    #
end
