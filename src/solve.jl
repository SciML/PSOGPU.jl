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

    gbest
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelPSOArray,
        args...;
        kwargs...)
    # if opt.threaded
    gbest = PSO(prob; population = opt.num_particles, kwargs...)
    # else
    #     init_gbest, particles = init_particles(prob, opt.num_particles)
    #     gbest = pso_solve_cpu!(prob, init_gbest, particles; kwargs...)
    # end

    gbest
end

function pso_solve(prob::OptimizationProblem, opt::SerialPSO, args...; kwargs...)
    init_gbest, particles = init_particles(prob, opt.num_particles)
    gbest = pso_solve_cpu!(prob, init_gbest, particles; kwargs...)

    gbest
end

function pso_solve(prob::OptimizationProblem,
        opt::ParallelSyncPSO,
        args...;
        kwargs...)
    backend = opt.backend
    init_gbest, particles = init_particles(prob, opt.num_particles)
    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)
    gpu_particles = KernelAbstractions.allocate(backend, particles_eltype, size(particles))
    copyto!(gpu_particles, particles)
    init_gbest = init_gbest

    gbest = pso_solve_sync_gpu!(prob, init_gbest, gpu_particles; kwargs...)

    gbest
end
