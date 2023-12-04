using PSOGPU
using Test, StaticArrays, LinearAlgebra, Random

const GROUP = get(ENV, "GROUP", "CPU")

@testset "Rosenbrock test dimension = $(n)" for n in 2:4
    global N = n
    include("./regression.jl")
end

if GROUP != "CPU"
    @eval using $(Symbol(GROUP))
    if GROUP == "CUDA"
        backend = CUDABackend()
    elseif GROUP == "AMDGPU"
        backend = ROCBackend()
    end
     @testset "Rosenbrock on gpu = $(n)" for n in 2:4
        global N = n
        include("./gpu.jl")
    end
end
