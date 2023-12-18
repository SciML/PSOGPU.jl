module PSOGPU

using SciMLBase, StaticArrays, Setfield, KernelAbstractions

import DiffEqGPU: GPUTsit5, vectorized_asolve, make_prob_compatible

## Use lb and ub either as StaticArray or pass them separately as CuArrays
## Passing as CuArrays makes more sense, or maybe SArray? The based on no. of dimension
struct PSOParticle{T1, T2 <: eltype(T1)}
    position::T1
    velocity::T1
    cost::T2
    best_position::T1
    best_cost::T2
end
struct PSOGBest{T1, T2 <: eltype(T1)}
    position::T1
    cost::T2
end

## required overloads for min or max computation on particles
function Base.isless(a::PSOGPU.PSOParticle{T1, T2},
        b::PSOGPU.PSOParticle{T1, T2}) where {T1, T2}
    a.best_cost < b.best_cost
end

function Base.typemax(::Type{PSOGPU.PSOParticle{T1, T2}}) where {T1, T2}
    PSOGPU.PSOParticle{T1, T2}(similar(T1),
        similar(T1),
        typemax(T2),
        similar(T1),
        typemax(T2))
end

include("./algorithms.jl")
include("./pso_cpu.jl")
include("./pso_gpu.jl")
include("./pso_async_gpu.jl")
include("./utils.jl")
include("./pso_sync_gpu.jl")
include("./ode_pso.jl")
include("./solve.jl")

export ParallelPSOKernel,
    ParallelSyncPSOKernel, ParallelPSOArray, SerialPSO, OptimizationProblem, solve

end
