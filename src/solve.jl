function get_pos(particle)
    return particle.position
end
mutable struct PSOCache{TP, TAlg, TPart, TGbest}
    prob::TP
    alg::TAlg
    particles::TPart
    gbest::TGbest
end
struct HybridPSOCache{TPc, TSp, TAlg}
    pso_cache::TPc
    start_points::TSp
    alg::TAlg
end

function __init(prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel}, sampler::T,
        args...; kwargs...) where {T <: QuasiMonteCarlo.SamplingAlgorithm}
    backend = opt.backend

    particles = KernelAbstractions.allocate(
        backend, SPSOParticle{typeof(prob.u0), eltype(typeof(prob.u0))}, opt.num_particles)

    qmc_samples = QuasiMonteCarlo.sample(opt.num_particles, prob.lb, prob.ub, sampler)

    qmc_samples = adapt(backend, qmc_samples)

    kernel! = gpu_init_particles!(backend)

    kernel!(
        particles, qmc_samples, prob, opt, typeof(prob.u0), T; ndrange = opt.num_particles)

    best_particle = minimum(particles)
    init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    return particles, init_gbest
end

function __init(prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel}, sampler::T,
        args...; kwargs...) where {T <: GPUSamplingAlgorithm}
    backend = opt.backend

    particles = KernelAbstractions.allocate(
        backend, SPSOParticle{typeof(prob.u0), eltype(typeof(prob.u0))}, opt.num_particles)
    kernel! = gpu_init_particles!(backend)

    kernel!(particles, prob, opt, typeof(prob.u0), T; ndrange = opt.num_particles)

    best_particle = minimum(particles)

    init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    particles, init_gbest
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::ParallelPSOKernel, args...; sampler = GPUUniformSampler(), kwargs...)
    @assert prob.u0 isa SArray

    ## Bounds check
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    particles, _init_gbest = if lb === nothing || ub === nothing ||
                                (all(isinf, lb) && all(isinf, ub))
        __init(prob, opt, GPUUnboundedSampler(), args...; kwargs...)
    else
        __init(prob, opt, sampler, args...; kwargs...)
    end

    init_gbest = KernelAbstractions.allocate(opt.backend, typeof(_init_gbest), (1,))
    copyto!(init_gbest, [_init_gbest])

    return PSOCache{
        typeof(prob), typeof(opt), typeof(particles), typeof(init_gbest)}(
        prob, opt, particles, init_gbest)
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::ParallelSyncPSOKernel, args...; sampler = GPUUniformSampler(), kwargs...)
    @assert prob.u0 isa SArray

    ## Bounds check
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    particles, init_gbest = if lb === nothing || ub === nothing ||
                               (all(isinf, lb) && all(isinf, ub))
        __init(prob, opt, GPUUnboundedSampler(), args...; kwargs...)
    else
        __init(prob, opt, sampler, args...; kwargs...)
    end

    return PSOCache{
        typeof(prob), typeof(opt), typeof(particles), typeof(init_gbest)}(
        prob, opt, particles, init_gbest)
end

# function SciMLBase.init(
#         prob::OptimizationProblem, opt::ParallelSyncPSOKernel, args...; kwargs...)
#     backend = opt.backend
#     @assert prob.u0 isa SArray

#     ## Bounds check
#     lb, ub = check_init_bounds(prob)
#     lb, ub = check_init_bounds(prob)
#     prob = remake(prob; lb = lb, ub = ub)

#     particles = KernelAbstractions.allocate(
#         backend, SPSOParticle{typeof(prob.u0), eltype(typeof(prob.u0))}, opt.num_particles)
#     kernel! = gpu_init_particles!(backend)

#     kernel!(particles, prob, opt, typeof(prob.u0); ndrange = opt.num_particles)

#     best_particle = minimum(particles)
#     init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

#     return PSOCache{
#         typeof(prob), typeof(opt), typeof(particles), typeof(init_gbest)}(
#         prob, opt, particles, init_gbest)
# end

function SciMLBase.reinit!(cache::Union{PSOCache, HybridPSOCache}; kwargs...)
    reinit_cache!(cache, cache.alg)
end

function reinit_cache!(cache::PSOCache, opt::ParallelPSOKernel)
    prob = cache.prob
    backend = opt.backend
    particles = cache.particles

    kernel! = PSOGPU.gpu_init_particles!(backend)
    kernel!(particles, prob, opt, typeof(prob.u0); ndrange = opt.num_particles)

    best_particle = minimum(particles)
    _init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    copyto!(cache.gbest, [_init_gbest])

    return nothing
end

function reinit_cache!(cache::PSOCache, opt::ParallelSyncPSOKernel)
    prob = cache.prob
    backend = opt.backend
    particles = cache.particles

    kernel! = PSOGPU.gpu_init_particles!(backend)
    kernel!(particles, prob, opt, typeof(prob.u0); ndrange = opt.num_particles)

    best_particle = minimum(particles)

    init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    cache.gbest = init_gbest

    return nothing
end

function SciMLBase.solve!(
        cache::Union{PSOCache, HybridPSOCache}, args...; maxiters = 100, kwargs...)
    solve!(cache, cache.alg, args...; maxiters, kwargs...)
end

function SciMLBase.solve!(
        cache::PSOCache, opt::ParallelPSOKernel, args...; maxiters = 100, kwargs...)
    prob = cache.prob
    t0 = time()
    gbest, particles = vectorized_solve!(cache.prob,
        cache.gbest,
        cache.particles,
        opt,
        Val(opt.global_update),
        args...;
        maxiters, kwargs...)
    t1 = time()

    particles_positions = get_pos.(particles)
    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = t1 - t0))
end

function SciMLBase.solve!(
        cache::PSOCache, opt::ParallelSyncPSOKernel, args...; maxiters = 100, kwargs...)
    prob = cache.prob
    t0 = time()
    gbest, particles = vectorized_solve!(prob,
        cache.gbest,
        cache.particles,
        opt,
        args...;
        maxiters,
        kwargs...)
    t1 = time()

    particles_positions = get_pos.(particles)
    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = t1 - t0))
end

function SciMLBase.solve(prob::OptimizationProblem,
        opt::Union{ParallelPSOKernel, ParallelSyncPSOKernel, HybridPSO},
        args...; maxiters = 100, kwargs...)
    solve!(init(prob, opt, args...; kwargs...), opt, args...; maxiters, kwargs...)
end

function SciMLBase.__solve(prob::OptimizationProblem,
        opt::PSOAlgorithm,
        args...;
        maxiters = 100,
        kwargs...)
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    gbest, particles, solve_time = pso_solve(prob, opt, args...; maxiters, kwargs...)
    particles_positions = get_pos.(particles)
    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, prob.f(gbest.position, prob.p), original = particles_positions,
        stats = Optimization.OptimizationStats(; time = solve_time))
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelPSOArray,
        args...;
        kwargs...)
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    t0 = time()
    gbest, particles = vectorized_solve!(prob,
        init_gbest,
        particles,
        opt,
        args...;
        kwargs...)
    t1 = time()
    gbest, particles, t1 - t0
end

function pso_solve(prob::OptimizationProblem, opt::SerialPSO, args...; kwargs...)
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    t0 = time()
    gbest, particles = vectorized_solve!(prob, init_gbest, particles, opt; kwargs...)
    t1 = time()
    gbest, particles, t1 - t0
end
