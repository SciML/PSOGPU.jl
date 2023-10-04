function init_particles(prob, ::GPU, n_particles, data_dict)
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p

    gbest_position = uniform(dim, lb, ub)
    gbest_position = SVector{length(gbest_position), eltype(gbest_position)}(gbest_position)
    gbest_cost = cost_func(gbest_position, p)
    particles = PSOParticle[]
    for i in 1:n_particles
        position = uniform(dim, lb, ub)
        position = SVector{length(position), eltype(position)}(position)
        velocity = @SArray zeros(eltype(position), dim)
        cost = cost_func(position, p)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, PSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest_cost
            gbest_position = copy(best_position)
            gbest_cost = copy(best_cost)
        end
    end
    gbest = PSOGBest(gbest_position, gbest_cost)
    return gbest, convert(Vector{typeof(particles[1])}, particles)
end

function update_particle_states!(prob, gpu_particles, gbest_ref, w; c1 = 1.4962f0,
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
    return nothing
end

function pso_solve_gpu!(prob,
    gbest,
    gpu_particles;
    maxiters = 100,
    w = 0.7298f0,
    wdamp = 1.0f0,
    debug = false)

    ## Initialize stuff

    gbest_ref = CuArray([gbest])

    kernel = @cuda launch=false update_particle_states!(prob, gpu_particles, gbest_ref, w)

    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)

    if debug
        @show config.threads
        @show config.blocks
    end

    for i in 1:maxiters
        ## Invoke GPU Kernel here
        kernel(prob, gpu_particles, gbest_ref, w)
        w = w * wdamp
    end

    return gbest_ref
end
