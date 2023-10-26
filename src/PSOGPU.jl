module PSOGPU

using SciMLBase, StaticArrays, Setfield, CUDA

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

struct ParallelPSO
    num_particles::Int
    async::Bool
    gpu::Bool
    threaded::Bool
end

function ParallelPSO(num_particles::Int;
    async = false,
    gpu = false, threaded = false)
    ParallelPSO(num_particles, async, gpu, threaded)
end

SciMLBase.allowsbounds(::ParallelPSO) = true
# SciMLBase.requiresbounds(::ParallelPSO) = true

struct GPU end
struct CPU end

include("./pso_cpu.jl")
include("./pso_gpu.jl")
include("./pso_async_gpu.jl")
include("./utils.jl")
include("./pso_sync_gpu.jl")

function SciMLBase.__solve(prob::OptimizationProblem,
    opt::ParallelPSO,
    args...;
    kwargs...)
    if !(opt.gpu)
        if opt.threaded
            gbest = PSO(prob; population = opt.num_particles, kwargs...)
        else
            init_gbest, particles = init_particles(prob, opt.num_particles)
            gbest = pso_solve_cpu!(prob, init_gbest, particles; kwargs...)
        end
    else
        if opt.async
            init_gbest, particles = init_particles(prob, opt.num_particles)
            gpu_particles = cu(particles)
            init_gbest = cu([init_gbest])
            gbest = pso_solve_async_gpu!(prob, init_gbest, gpu_particles; kwargs...)
        else
            init_gbest, particles = init_particles(prob, opt.num_particles)
            gpu_particles = cu(particles)
            init_gbest = cu([init_gbest])
            gbest = pso_solve_gpu!(prob, init_gbest, gpu_particles; kwargs...)
        end
    end

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end

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

export ParallelPSO, OptimizationProblem, solve
end
