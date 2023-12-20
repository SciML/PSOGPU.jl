function SciMLBase.__solve(prob::OptimizationProblem, opt::PSOAlogrithm, args...; kwargs...)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    gbest = pso_solve(prob, opt, args...; kwargs...)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelPSOKernel,
        args...;
        kwargs...)
    backend = opt.backend
    @assert prob.u0 isa SArray
    init_gbest, particles = init_particles(prob, opt.num_particles, typeof(prob.u0))
    # TODO: Do the equivalent of cu()/roc()
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))
    copyto!(gpu_particles, particles)
    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])

    gbest = vectorized_solve!(prob,
        gpu_init_gbest,
        gpu_particles,
        opt,
        Val(opt.global_update),
        args...;
        kwargs...)
    gbest
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelPSOArray,
        args...;
        kwargs...)
    init_gbest, particles = init_particles(prob, opt.num_particles, typeof(prob.u0))
    gbest = vectorized_solve!(prob, init_gbest, particles, opt, args...; kwargs...)
    gbest
end

function pso_solve(prob::OptimizationProblem, opt::SerialPSO, args...; kwargs...)
    init_gbest, particles = init_particles(prob, opt.num_particles, typeof(prob.u0))
    gbest = vectorized_solve!(prob, init_gbest, particles, opt; kwargs...)
    gbest
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelSyncPSOKernel,
        args...;
        kwargs...)
    backend = opt.backend
    init_gbest, particles = init_particles(prob, opt.num_particles, typeof(prob.u0))
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend, particles_eltype, size(particles))
    copyto!(gpu_particles, particles)
    init_gbest = init_gbest

    gbest = vectorized_solve!(prob, init_gbest, gpu_particles, opt; kwargs...)
    gbest
end
