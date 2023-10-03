function update_particle_states_async!(prob,
    gpu_particles,
    gbest_ref,
    w, wdamp, max_iters;
    c1 = 1.4962f0,
    c2 = 1.4962f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    i > length(gpu_particles) && return

    gbest = gbest_ref[1]

    ## Access the particle
    @inbounds particle = gpu_particles[i]

    ## Run all generations
    for i in 1:max_iters
        updated_velocity = w .* particle.velocity .+
                           c1 .* rand(typeof(particle.velocity)) .*
                           (particle.best_position -
                            particle.position) .+
                           c2 .* rand(typeof(particle.velocity)) .*
                           (gbest.position - particle.position)

        @set! particle.velocity = updated_velocity

        @set! particle.position = particle.position + particle.velocity

        update_pos = max(particle.position, prob.lb)
        update_pos = min(update_pos, prob.ub)
        @set! particle.position = update_pos
        # @set! particle.position = min(particle.position, ub)

        @set! particle.cost = prob.f(particle.position, prob.p)

        if particle.cost < particle.best_cost
            @set! particle.best_position = particle.position
            @set! particle.best_cost = particle.cost
        end

        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end
        w = w * wdamp
    end

    @inbounds gpu_particles[i] = particle

    @inbounds gbest_ref[1] = gbest

    return nothing
end

function pso_solve_async_gpu!(prob,
    gbest,
    gpu_particles;
    max_iters = 100,
    w = 0.7298f0,
    wdamp = 1.0f0,
    debug = false)

    ## Initialize stuff

    gbest_ref = CuArray([gbest])

    kernel = @cuda launch=false update_particle_states_async!(prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, max_iters)

    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)

    if debug
        @show config.threads
        @show config.blocks
    end

    kernel(prob, gpu_particles, gbest_ref, w, wdamp, max_iters)

    best_particle = minimum(gpu_particles)
    return PSOGBest(best_particle.best_position, best_particle.best_cost)
end
