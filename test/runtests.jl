using PSOGPU
using Test, StaticArrays, LinearAlgebra, Random

const GROUP = get(ENV, "GROUP", "CPU")

@testset "Rosenbrock test dimension = $(n)" for n in 2:4
    global N = n
    include("./regression.jl")
end

if GROUP != "CPU"
    @eval using $(Symbol(GROUP))

    @testset "Rosenbrock on gpu" begin
        include("./gpu.jl")
    end
end
