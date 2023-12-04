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

struct ParallelPSOKernel{Backend}
    num_particles::Int
    async::Bool
    threaded::Bool
    backend::Backend
end
struct ParallelSyncPSO{Backend}
    num_particles::Int
    backend::Backend
end

function ParallelPSOKernel(num_particles::Int;
        async = false, threaded = false, backend = CPU())
    ParallelPSOKernel(num_particles, async, threaded, backend)
end

SciMLBase.allowsbounds(::ParallelPSOKernel) = true
SciMLBase.allowsbounds(::ParallelSyncPSO) = true
# SciMLBase.requiresbounds(::ParallelPSOKernel) = true

include("./pso_cpu.jl")
include("./pso_gpu.jl")
include("./pso_async_gpu.jl")
include("./utils.jl")
include("./pso_sync_gpu.jl")
include("./ode_pso.jl")

function SciMLBase.__solve(prob::OptimizationProblem,
        opt::ParallelPSOKernel,
        args...;
        kwargs...)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), length(prob.u0)) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), length(prob.u0)) : prob.ub

    prob = remake(prob; lb = lb, ub = ub)

    ## TODO: Compare the performance of KA kernels with CPU backend with CPU implementations
    if opt.backend isa CPU
        if opt.threaded
            gbest = PSO(prob; population = opt.num_particles, kwargs...)
        else
            init_gbest, particles = init_particles(prob, opt.num_particles)
            gbest = pso_solve_cpu!(prob, init_gbest, particles; kwargs...)
        end
    else
        backend = opt.backend
        init_gbest, particles = init_particles(prob, opt.num_particles)
        # TODO: Do the equivalent of cu()/roc()
        particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
        gpu_particles = KernelAbstractions.allocate(backend,
            particles_eltype,
            size(particles))
        copyto!(gpu_particles, particles)
        gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
        copyto!(gpu_init_gbest, [init_gbest])
        if opt.async
            gbest = pso_solve_async_gpu!(prob, gpu_init_gbest, gpu_particles; kwargs...)
        else
            gbest = pso_solve_gpu!(prob, gpu_init_gbest, gpu_particles; kwargs...)
        end
    end

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end

function SciMLBase.__solve(prob::OptimizationProblem,
        opt::ParallelSyncPSO,
        args...;
        kwargs...)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), length(prob.u0)) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), length(prob.u0)) : prob.ub

    prob = remake(prob; lb = lb, ub = ub)
    backend = opt.backend
    init_gbest, particles = init_particles(prob, opt.num_particles)
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend, particles_eltype, size(particles))
    copyto!(gpu_particles, particles)
    init_gbest = init_gbest
    gbest = pso_solve_sync_gpu!(prob, init_gbest, gpu_particles; kwargs...)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end

using Base

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

export ParallelPSOKernel, ParallelSyncPSO, OptimizationProblem, solve
end
