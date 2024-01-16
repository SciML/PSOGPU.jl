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

@kernel function update_particle_states!(prob,
        gpu_particles::AbstractArray{SPSOParticle{T1, T2}}, gbest_ref, w,
        opt::ParallelPSOKernel, lock; c1 = 1.4962f0,
        c2 = 1.4962f0) where {T1, T2}
    i = @index(Global, Linear)
    # FIXME: Determine the right amount of shmem to use

    @uniform gs = @groupsize()[1]

    best_queue = @localmem SPSOGBest{T1, T2} (gs)
    queue_num = @localmem UInt32 1

    @inbounds gbest = gbest_ref[1]
    @inbounds particle = gpu_particles[i]

    # Initialize cost to be Inf
    for bq_idx in 1:gs
        best_queue[bq_idx] = SPSOGBest(particle.best_position,
            convert(typeof(particle.cost), Inf))
    end

    @synchronize

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)
    @inbounds gpu_particles[i] = particle

    @synchronize

    if particle.best_cost < gbest.cost
        queue_idx = @atomic queue_num[1] += UInt32(1)
        @inbounds best_queue[queue_idx] = SPSOGBest(particle.best_position,
            particle.best_cost)
    end
    @synchronize
    if i <= first(@ndrange())
        tidx = @index(Local, Linear)
        if tidx == 1
            if queue_num[1] > 1
                # Find best fit in block
                for j in 2:queue_num[1]
                    @inbounds if best_queue[j].cost < best_queue[1].cost
                        best_queue[1] = best_queue[j]
                    end
                end

                # Take lock
                while true
                    res = @atomicreplace lock[1] UInt32(0)=>UInt32(1)
                    if res.success
                        break
                    end
                end

                # Update global best fit
                gbest = @inbounds gbest_ref[1]
                @inbounds if best_queue[1].cost < gbest.cost
                    gbest_ref[1] = best_queue[1]
                end

                # Release lock
                @atomicswap lock[1] = 0
            end
        end
    end
end

@kernel function update_particle_states!(prob, gpu_particles, gbest, w,
        opt::ParallelSyncPSOKernel; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)

    @inbounds particle = gpu_particles[i]

    particle = update_particle_state(particle, prob, gbest, w, c1, c2, i, opt)

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
