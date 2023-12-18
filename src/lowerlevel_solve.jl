function vectorized_solve!(prob,
        gbest,
        gpu_particles, opt::ParallelSyncPSOKernel;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false)
    backend = get_backend(gpu_particles)

    update_particle_kernel = update_particle_states!(backend)

    for i in 1:maxiters
        update_particle_kernel(prob,
            gpu_particles,
            gbest,
            w, opt;
            ndrange = length(gpu_particles))
        best_particle = minimum(gpu_particles)
        gbest = PSOGBest(best_particle.position, best_particle.best_cost)
        w = w * wdamp
    end

    return gbest
end

function vectorized_solve!(prob,
        gbest,
        gpu_particles, opt::ParallelPSOKernel;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false)

    ## Initialize stuff

    backend = get_backend(gpu_particles)

    kernel = update_particle_states!(backend)

    for i in 1:maxiters
        ## Invoke GPU Kernel here
        kernel(prob, gpu_particles, gbest, w, opt; ndrange = length(gpu_particles))
        w = w * wdamp
    end

    return Array(gbest)[1]
end
