@inline function update_particle_state(particle, prob, gbest, w, c1, c2)
    updated_velocity = w .* particle.velocity .+
                       c1 .* rand(typeof(particle.velocity)) .*
                       (particle.best_position -
                        particle.position) .+
                       c2 .* rand(typeof(particle.velocity)) .*
                       (gbest.position - particle.position)

    @set! particle.velocity = updated_velocity

    @set! particle.position = particle.position + particle.velocity

    update_pos = max.(particle.position, prob.lb)
    update_pos = min.(update_pos, prob.ub)
    @set! particle.position = update_pos

    @set! particle.cost = prob.f(particle.position, prob.p)

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end
    particle
end

@kernel function update_particle_states!(prob, gpu_particles, gbest_ref, w,
        opt::ParallelPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    @inbounds gbest = gbest_ref[1]
    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2)

    ## NOTE: This causes thread races to update global best particle.
    if particle.best_cost < gbest.cost
        @set! gbest.position = particle.best_position
        @set! gbest.cost = particle.best_cost
    end

    @inbounds gbest_ref[1] = gbest

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states!(prob, gpu_particles, gbest, w,
        opt::ParallelSyncPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2)

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states_async!(prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters;
        c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    gbest = gbest_ref[1]

    ## Access the particle
    @inbounds particle = gpu_particles[i]

    ## Run all generations
    for i in 1:maxiters
        particle = update_particle_state(particle, prob, gbest, w, c1, c2)
        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end
        w = w * wdamp
    end

    @inbounds gpu_particles[i] = particle
    @inbounds gbest_ref[1] = gbest
end
