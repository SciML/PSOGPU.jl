function get_pos(particle)
    return particle.position
end

mutable struct PSOCache{TP, TAlg, TPart, TGbest}
    prob::TP
    alg::TAlg
    particles::TPart
    gbest::TGbest
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::ParallelPSOKernel, args...; kwargs...)
    backend = opt.backend
    @assert prob.u0 isa SArray

    ## initialize cache

    ## Bounds check
    lb, ub = check_init_bounds(prob)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))

    # TODO: Do the equivalent of cu()/roc()
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))
    copyto!(gpu_particles, particles)
    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])
    return PSOCache{
        typeof(prob), typeof(opt), typeof(gpu_particles), typeof(gpu_init_gbest)}(
        prob, opt, gpu_particles, gpu_init_gbest)
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

function SciMLBase.solve(prob::OptimizationProblem, opt::ParallelPSOKernel,
        args...; maxiters = 100, kwargs...)
    solve!(init(prob, opt, args...; maxiters, kwargs...), opt)
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

    t0 = time()
    gbest, particles = vectorized_solve!(prob,
        gpu_init_gbest,
        gpu_particles,
        opt,
        Val(opt.global_update),
        args...;
        kwargs...)
    t1 = time()
    gbest, particles, t1 - t0
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

function pso_solve(prob::OptimizationProblem,
        opt::ParallelSyncPSOKernel,
        args...;
        kwargs...)
    backend = opt.backend
    init_gbest, particles = init_particles(prob, opt, typeof(prob.u0))
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend, particles_eltype, size(particles))
    copyto!(gpu_particles, particles)
    init_gbest = init_gbest

    t0 = time()
    gbest, particles = vectorized_solve!(prob,
        init_gbest,
        gpu_particles,
        opt,
        args...;
        kwargs...)
    t1 = time()
    gbest, particles, t1 - t0
end
