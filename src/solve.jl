SciMLBase.supports_opt_cache_interface(::PSOAlgorithm) = true

function SciMLBase.__init(prob::OptimizationProblem, opt::PSOAlgorithm, data = Optimization.DEFAULT_DATA;
        save_best = true,
        callback = (args...) -> (false),
        progress = false,
        kwargs...)
    
    return Optimization.OptimizationCache(prob, opt, data; save_best, callback, progress, kwargs...)
end

function SciMLBase.__solve(cache::Optimization.OptimizationCache{F, RC, LB, UB, LC, UC, S, O, D, P, C}) where {F, RC, LB, UB, LC, UC, S, O <: PSOAlgorithm, D, P, C}
    lb, ub = check_init_bounds(cache)
    @set! cache.lb = lb
    @set! cache.ub = ub
    gbest, particles = pso_solve(cache, cache.opt)
    particles_positions = getfield.(particles, Ref(:position))
    SciMLBase.build_solution(cache, cache.opt,
        gbest.position, cache.f(gbest.position, cache.p), original = particles_positions)
end

function pso_solve(prob::Optimization.OptimizationCache,
        opt::ParallelPSOKernel,
        args...;
        kwargs...)
    backend = opt.backend
    @assert prob.u0 isa SArray
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    # TODO: Do the equivalent of cu()/roc()
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))
    copyto!(gpu_particles, particles)
    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])

    gbest, particles = vectorized_solve!(prob,
        gpu_init_gbest,
        gpu_particles,
        opt,
        Val(opt.global_update),
        args...;
        kwargs...)
    gbest, particles
end

function pso_solve(prob::Optimization.OptimizationCache,
        opt::ParallelPSOArray,
        args...;
        kwargs...)
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    gbest, particles = vectorized_solve!(prob, init_gbest, particles, opt, args...; kwargs...)
    gbest, particles
end

function pso_solve(prob::Optimization.OptimizationCache, opt::SerialPSO, args...; kwargs...)
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    gbest, particles = vectorized_solve!(prob, init_gbest, particles, opt; kwargs...)
    gbest, particles
end

function pso_solve(prob::Optimization.OptimizationCache,
        opt::ParallelSyncPSOKernel,
        args...;
        kwargs...)
    backend = opt.backend
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend, particles_eltype, size(particles))
    copyto!(gpu_particles, particles)
    init_gbest = init_gbest

    gbest = vectorized_solve!(prob, init_gbest, gpu_particles, opt; kwargs...)
    gbest
end
