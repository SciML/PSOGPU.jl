@kernel function _update_particle_states!(gpu_particles, lb, ub, gbest, w; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
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
    end
end

@kernel function _update_particle_costs!(losses, gpu_particles)
    i = @index(Global, Linear)
    if i <= length(losses)
        @inbounds particle = gpu_particles[i]
        @inbounds loss = losses[i]

        @set! particle.cost = loss

        if particle.cost < particle.best_cost
            @set! particle.best_position = particle.position
            @set! particle.best_cost = particle.cost
        end

        @inbounds gpu_particles[i] = particle
    end
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
        maxiters = 100,
        backend = CPU(), kwargs...)
    update_states! = PSOGPU._update_particle_states!(backend)

    losses = KernelAbstractions.ones(backend, 1, length(gpu_particles))
    update_costs! = PSOGPU._update_particle_costs!(backend)

    improb = make_prob_compatible(prob)

    for i in 1:maxiters
        update_states!(gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange=length(gpu_particles))

        probs = prob_func.(Ref(improb), gpu_particles)

        ts, us = vectorized_asolve(probs,
            prob,
            ode_alg; kwargs...)

        sum!(losses, (map(x -> sum(x .^ 2), data .- us)))

        update_costs!(losses, gpu_particles; ndrange=length(losses))

        best_particle = minimum(gpu_particles,
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
