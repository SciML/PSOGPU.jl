using PSOGPU
using Test, StaticArrays, LinearAlgebra

@testset "Rosenbrock test dimension = $(n)" for n in 2:4
    global N = n
    include("./regression.jl")
end
