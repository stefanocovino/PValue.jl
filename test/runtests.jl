using MyJuliaPkg
using Test

@testset "MyJuliaPkg.jl" begin
    # BIC
    @test BIC(56.,100,3) == -98.18448944203573
    #
    # Frequentist_p_value
    @test Frequentist_p_value(85.3,90) == 0.620453567577112
    #
    # RMS
    @test RMS([1.1,2.2],[1.15,2.15]) == 0.050000000000000044
    #
    # SSR
    @test SSR([1.,2.,3.,4.],[1.1,1.9,3.05,3.8],[0.1,0.05,0.2,0.1]) == 9.062500000000016
    #
end
