function _update_particle_states!(prob, gpu_particles, gbest, w; c1 = 1.4962f0,
    c2 = 1.4962f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(gpu_particles) && return

    # i = 1

    ## Access the particle

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

    @inbounds gpu_particles[i] = particle

    return nothing
end


function calculate_loss!(prob, gpu_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(gpu_particles) && return


    @inbounds particle = gpu_particles[i]

    @set! particle.cost = prob.f(particle.position, prob.p)

    @inbounds gpu_particles[i] = particle

    return nothing
end
