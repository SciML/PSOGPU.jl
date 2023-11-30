@kernel function update_particle_states!(prob, gpu_particles, gbest_ref, w; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        # i = 1

        ## Access the particle

        @inbounds gbest = gbest_ref[1]

        # gpu_particles = convert(MArray, gpu_particles)

        @inbounds particle = gpu_particles[i]
        ## Update velocity

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

        @inbounds gpu_particles[i] = particle

        @inbounds gbest_ref[1] = gbest

        # gpu_particles = convert(SArray, gpu_particles)
    end
end

function pso_solve_gpu!(prob,
        gbest,
        gpu_particles;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false)

    ## Initialize stuff

    backend = get_backend(gpu_particles)

    kernel = update_particle_states!(backend)

    for i in 1:maxiters
        ## Invoke GPU Kernel here
        kernel(prob, gpu_particles, gbest, w; ndrange = length(gpu_particles))
        w = w * wdamp
    end

    return Array(gbest)[1]
end
