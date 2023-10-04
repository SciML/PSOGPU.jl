module PSOGPU

using SciMLBase, StaticArrays, Setfield, CUDA

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

function uniform(dim::Int, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    arr = rand(T, dim)
    @inbounds for i in 1:dim
        arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i]
    end
    return arr
end

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

ParallelPSO(num_particles::Int;
    async = false,
    gpu = false, threaded = false) = ParallelPSO(num_particles, async, gpu, threaded)

struct GPU end
struct CPU end

function SciMLBase.__solve(prob::OptimizationProblem,
    opt::ParallelPSO,
    args...;
    kwargs...)

    if typeof(opt.lb) <: AbstractFloat 
        lb = [opt.lb for i in 1:length(prob.u0)]
        ub = [opt.ub for i in 1:length(prob.u0)]
    else
        lb = opt.lb
        ub = opt.ub
    end

    if !(opt.gpu)
        if opt.threaded
            gbest, = PSO(problem, prob.p, max_iter = max_iter, population = opt.num_particles)
        else
            gbest, cpu_particles = initialize_particles(prob, ::CPU, opt.num_particles, prob.p)    
            pso_solve_cpu!(prob, gbest, cpu_particles; kwargs...)
        end
    else
        if opt.async
            gbest, gpu_particles = PSOGPU.init_particles(prob, ::GPU, opt.num_particles, prob.p)
            pso_solve_async_gpu!(prob, gbest, gpu_particles; kwargs...)
        else
            gbest, gpu_particles = PSOGPU.init_particles(prob, ::GPU, opt.num_particles, prob.p)
            pso_solve_gpu!(prob, gbest, gpu_particles; kwargs...)
        end
    end

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end

export ParallelPSO, OptimizationProblem, solve
end
