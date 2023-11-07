function update_particles(prob, gpu_particles, gbest_ref, w; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(gpu_particles) && return

    # i = 1

    ## Access the particle

    @inbounds gbest = gbest_ref[1]

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
    # @set! particle.position = min(particle.position, ub)
    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end

    @inbounds gpu_particles[i] = particle

    return nothing
end

gbest = minimum(gpu_particles)

@set! particle.cost = prob.f(particle.position, prob.p)
