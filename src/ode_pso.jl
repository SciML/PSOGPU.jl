function _update_particle_states!(lb, ub, gpu_particles, gbest, w; c1 = 1.4962f0,
    c2 = 1.4962f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(gpu_particles) && return

    @inbounds particle = gpu_particles[i]

    updated_velocity = w .* particle.velocity .+
                       c1 .* rand(typeof(particle.velocity)) .* (particle.best_position -
                        particle.position) .+
                       c2 .* rand(typeof(particle.velocity)) .*
                       (gbest.position - particle.position)

    @set! particle.velocity = updated_velocity

    @set! particle.position = particle.position + particle.velocity

    update_pos = max(particle.position, lb)
    update_pos = min(update_pos, ub)

    @set! particle.position = update_pos

    @inbounds gpu_particles[i] = particle

    return nothing
end

function _update_particle_costs!(losses, gpu_particles)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(losses) && return

    @inbounds particle = gpu_particles[i]

    @set! particle.cost = convert(typeof(particle.cost), losses[i])

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end

    @inbounds gpu_particles[i] = particle

    return nothing
end

function remake_prob_gpu(prob, gpu_particle)
    return make_prob_compatible(remake(prob, p = gpu_particle.position))
end

function parameter_estim_ode!(prob::ODEProblem,
    gpu_particles,
    gbest,
    data,
    lb,
    ub;
    ode_alg = GPUTsit5(),
    w = 0.72980,
    wdamp = 1.00,
    maxiters = 100, kwargs...)
    update_states! = @cuda launch=false _update_particle_states!(lb,
        ub,
        gpu_particles,
        gbest,
        w)

    improb = make_prob_compatible(prob)

    for i in 1:maxiters
        update_states!(lb, ub, gpu_particles, gbest, w)

        probs = remake_prob_gpu.(Ref(improb), gpu_particles)

        ts, us = vectorized_asolve(probs,
            prob,
            ode_alg; kwargs...)

        losses = sum((map(x -> sum(x .^ 2), data .- us)), dims = 1)

        @cuda _update_particle_costs!(losses, gpu_particles)

        # @show typeof(gpu_particles)

        best_particle = minimum(gpu_particles)
        gbest = PSOGPU.PSOGBest(best_particle.best_position, best_particle.best_cost)
        # @show gbest
        w = w * wdamp
    end
    return gbest
end
