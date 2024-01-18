@inline function update_particle_state(particle, prob, gbest, w, c1, c2, iter, opt)
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

    particle = handle_constraints(particle, prob, iter, opt)

    if particle.cost < particle.best_cost
        @set! particle.best_position = particle.position
        @set! particle.best_cost = particle.cost
    end
    particle
end

@inline function handle_constraints(particle, prob, iter, opt)
    if !isnothing(prob.f.cons)
        penalty = calc_penalty(particle.position, prob, iter + 1, opt.θ, opt.γ, opt.h)
        @set! particle.cost = prob.f(particle.position, prob.p) + penalty
    else
        @set! particle.cost = prob.f(particle.position, prob.p)
    end
    particle
end

@kernel function update_particle_states!(prob, gpu_particles, gbest_ref, w,
        opt::ParallelPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    @inbounds gbest = gbest_ref[1]
    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)

    ## NOTE: This causes thread races to update global best particle.
    if particle.best_cost < gbest.cost
        @set! gbest.position = particle.best_position
        @set! gbest.cost = particle.best_cost
    end

    @inbounds gbest_ref[1] = gbest

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states!(prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, block_particles, gbest, w,
        opt::ParallelSyncPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0) where {T1, T2}
    i = @index(Global, Linear)
    tidx = @index(Local, Linear)
    gidx = @index(Group, Linear)

    @uniform gs = @groupsize()[1]

    group_particles = @localmem SPSOGBest{T1, T2} (gs)

    if tidx == 1
        fill!(group_particles, SPSOGBest(gbest.position, convert(typeof(gbest.cost), Inf)))
    end

    @synchronize

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)

    @inbounds group_particles[tidx] = SPSOGBest(particle.best_position, particle.best_cost)

    stride = gs ÷ 2

    while stride >= 1
        @synchronize
        if tidx <= stride
            @inbounds if group_particles[tidx].cost > group_particles[tidx + stride].cost
                group_particles[tidx] = group_particles[tidx + stride]
            end
        end
        stride = stride ÷ 2
    end

    @synchronize

    if tidx == 1
        @inbounds block_particles[gidx] = group_particles[tidx]
    end

    @inbounds gpu_particles[i] = particle
end

@kernel function update_particle_states_async!(prob,
        gpu_particles,
        gbest_ref,
        w, wdamp, maxiters, opt;
        c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    gbest = gbest_ref[1]

    ## Access the particle
    @inbounds particle = gpu_particles[i]

    ## Run all generations
    for i in 1:maxiters
        particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)
        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end
        w = w * wdamp
    end

    @inbounds gpu_particles[i] = particle
    @inbounds gbest_ref[1] = gbest
end
