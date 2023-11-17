@kernel function _update_particle_states!(prob, gpu_particles, gbest, w; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        # gpu_particles = convert(MArray, gpu_particles)

        @inbounds particle = gpu_particles[i]
        ## Update velocity

        updated_velocity = w .* particle.velocity .+
                           c1 .* rand(typeof(particle.velocity)) .* (particle.best_position -
                            particle.position) .+
                           c2 .* rand(typeof(particle.velocity)) .*
                           (gbest.position - particle.position)

        @set! particle.velocity = updated_velocity

        @set! particle.position = particle.position + particle.velocity

        update_pos = max(particle.position, prob.lb)
        update_pos = min(update_pos, prob.ub)
        @set! particle.position = update_pos

        @set! particle.cost = prob.f(particle.position, prob.p)

        if particle.cost < particle.best_cost
            @set! particle.best_position = particle.position
            @set! particle.best_cost = particle.cost
        end

        @inbounds gpu_particles[i] = particle
    end
end

function pso_solve_sync_gpu!(prob,
        gbest,
        gpu_particles;
        maxiters = 100,
        w = 0.7298f0,
        wdamp = 1.0f0,
        debug = false,
        backend = CPU())
    @show minimum(gpu_particles)

    update_particle_kernel = _update_particle_states!(backend)

    for i in 1:maxiters
        update_particle_kernel(prob, gpu_particles, gbest, w; ndrange=length(gpu_particles))
        best_particle = minimum(gpu_particles)
        gbest = PSOGBest(best_particle.position, best_particle.best_cost)
        w = w * wdamp
    end

    return gbest
end
