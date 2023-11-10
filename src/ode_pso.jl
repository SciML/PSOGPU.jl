function _update_particle_states!(gpu_particles, lb, ub, gbest, w; c1 = 1.4962f0,
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
    @inbounds loss = losses[i]

    @set! particle.cost = loss

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end

    @inbounds gpu_particles[i] = particle

    return nothing
end

function default_prob_func(prob, gpu_particle)
    return remake(prob, p = gpu_particle.position)
end

function parameter_estim_ode!(prob::ODEProblem,
        gpu_particles,
        gbest,
        data,
        lb,
        ub;
        ode_alg = GPUTsit5(),
        prob_func = default_prob_func,
        w = 0.72980f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...)
    update_states! = @cuda launch=false PSOGPU._update_particle_states!(gpu_particles, lb,
        ub,
        gbest,
        w)

    losses = CUDA.ones(1, length(gpu_particles))
    update_costs! = @cuda launch=false PSOGPU._update_particle_costs!(losses, gpu_particles)

    config_states = launch_configuration(update_states!.fun)
    config_costs = launch_configuration(update_costs!.fun)

    improb = make_prob_compatible(prob)

    for i in 1:maxiters
        update_states!(gpu_particles,
            lb,
            ub,
            gbest,
            w;
            config_states.threads,
            config_states...)

        probs = prob_func.(Ref(improb), gpu_particles)

        ts, us = vectorized_asolve(probs,
            prob,
            ode_alg; kwargs...)

        sum!(losses, (map(x -> sum(x .^ 2), data .- us)))

        update_costs!(losses, gpu_particles; config_costs.threads, config_costs...)

        best_particle = mapreduce(x -> x,
            min,
            gpu_particles,
            init = PSOGPU.PSOParticle(gbest.position,
                gbest.position,
                gbest.cost,
                gbest.position,
                gbest.cost))

        gbest = PSOGPU.PSOGBest(best_particle.best_position, best_particle.best_cost)
        w = w * wdamp
    end
    return gbest
end
