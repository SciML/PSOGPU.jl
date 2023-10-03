module PSOGPU

using SciMLBase, StaticArrays, Setfield, CUDA

include("./algorithms.jl")
include("./pso_cpu.jl")
include("./pso_gpu.jl")
include("./pso_async_gpu.jl")

using Base

## required overloads for min or max computation on particles
function Base.isless(a::PSOGPU.PSOParticle{T1, T2},
    b::PSOGPU.PSOParticle{T1, T2}) where {T1, T2}
    a.cost < b.cost
end

function Base.typemax(::Type{PSOGPU.PSOParticle{T1, T2}}) where {T1, T2}
    PSOGPU.PSOParticle{T1, T2}(similar(T1),
        similar(T1),
        typemax(T2),
        similar(T1),
        typemax(T2))
end

function SciMLBase.__solve(prob::OptimizationProblem,
    opt::ParallelPSOCPU,
    args...;
    kwargs...)
    if typeof(opt.lb) <: AbstractFloat
        lb = [opt.lb for i in 1:length(prob.u0)]
        ub = [opt.ub for i in 1:length(prob.u0)]
    else
        lb = opt.lb
        ub = opt.ub
    end

    problem = Problem(prob.f, length(prob.u0), lb, ub)

    max_iter = 101

    gbest, = PSO(problem, prob.p, max_iter = max_iter, population = opt.num_particles)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end
export ParallelPSOCPU, OptimizationProblem, solve
end
