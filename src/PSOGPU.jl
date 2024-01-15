module PSOGPU

using SciMLBase, StaticArrays, Setfield, KernelAbstractions
using QuasiMonteCarlo, Optimization, SimpleNonlinearSolve, ForwardDiff
import Adapt

# import DiffEqGPU: GPUTsit5, vectorized_asolve, make_prob_compatible

## Use lb and ub either as StaticArray or pass them separately as CuArrays
## Passing as CuArrays makes more sense, or maybe SArray? The based on no. of dimension
struct SPSOParticle{T1, T2 <: eltype(T1)}
    position::T1
    velocity::T1
    cost::T2
    best_position::T1
    best_cost::T2
end
struct SPSOGBest{T1, T2 <: eltype(T1)}
    position::T1
    cost::T2
end

mutable struct MPSOParticle{T}
    position::AbstractArray{T}
    velocity::AbstractArray{T}
    cost::T
    best_position::AbstractArray{T}
    best_cost::T
end
mutable struct MPSOGBest{T}
    position::AbstractArray{T}
    cost::T
end

## required overloads for min or max computation on particles
function Base.isless(a::PSOGPU.SPSOParticle{T1, T2},
        b::PSOGPU.SPSOParticle{T1, T2}) where {T1, T2}
    a.best_cost < b.best_cost
end

function Base.typemax(::Type{PSOGPU.SPSOParticle{T1, T2}}) where {T1, T2}
    PSOGPU.SPSOParticle{T1, T2}(similar(T1),
        similar(T1),
        typemax(T2),
        similar(T1),
        typemax(T2))
end

include("./algorithms.jl")
include("./utils.jl")
# include("./ode_pso.jl")
include("./kernels.jl")
include("./lowerlevel_solve.jl")
include("./solve.jl")
include("./lbfgs.jl")
include("./hybrid.jl")

export ParallelPSOKernel,
    ParallelSyncPSOKernel, ParallelPSOArray, SerialPSO, OptimizationProblem, solve
end
