# https://stackoverflow.com/questions/65342388/why-my-code-in-julia-is-getting-slower-for-higher-iteration

function PSO(problem::OptimizationProblem,
    data_dict;
    max_iter = 100,
    population = 100,
    c1 = 1.4962,
    c2 = 1.4962,
    w = 0.7298,
    wdamp = 1.0,
    verbose = false)
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f
    p = prob.p

    gbest, particles = initialize_particles(problem, population, data_dict)

    # main loop
    for iter in 1:max_iter
        Threads.@threads for i in 1:population
            particles[i].velocity .= w .* particles[i].velocity .+
                                     c1 .* rand(dim) .* (particles[i].best_position .-
                                      particles[i].position) .+
                                     c2 .* rand(dim) .*
                                     (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, lb)
            particles[i].position .= min.(particles[i].position, ub)

            particles[i].cost = cost_func(particles[i].position, data_dict)

            if particles[i].cost < particles[i].best_cost
                particles[i].best_position = copy(particles[i].position)
                particles[i].best_cost = copy(particles[i].cost)

                if particles[i].best_cost < gbest.cost
                    gbest.position = copy(particles[i].best_position)
                    gbest.cost = copy(particles[i].best_cost)
                end
            end
        end
        w = w * wdamp
        if verbose && iter % 50 == 1
            println("Iteration " * string(iter) * ": Best Cost = " * string(gbest.cost))
            println("Best Position = " * string(gbest.position))
            println()
        end
    end
    gbest, particles
end

function initialize_particles(problem, ::CPU, population, data_dict)
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest_position = uniform(dim, lb, ub)
    gbest = PSOGbest(gbest_position, cost_func(gbest_position, data_dict))

    particles = PSOParticle[]
    for i in 1:population
        position = uniform(dim, lb, ub)
        velocity = zeros(dim)
        cost = cost_func(position, data_dict)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, PSOParticle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, particles
end


function update_particle_states_cpu!(prob, particles, gbest_ref, w; c1 = 1.4962f0,
    c2 = 1.4962f0)
    # i = 1

    ## Access the particle

    # gpu_particles = convert(MArray, gpu_particles)

    gbest = gbest_ref[]

    for i in eachindex(particles)
        @inbounds particle = particles[i]
        ## Update velocity

        updated_velocity = w * particle.velocity +
                           c1 .* rand(typeof(particle.velocity)) .*
                           (particle.best_position -
                            particle.position) +
                           c2 * rand(typeof(particle.velocity)) .*
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

        particles[i] = particle
    end
    gbest_ref[] = gbest
    return nothing
end

function pso_solve_cpu!(prob,
    gbest,
    cpu_particles;
    maxiters = 100,
    w = 0.7298f0,
    wdamp = 1.0f0,
    debug = false,
    threaded = false)

    sol_ref = Ref(gbest)
    if threaded
        PSO(prob,
        data_dict;
        max_iter = 100,
        population = 100,
        c1 = 1.4962,
        c2 = 1.4962,
        w = 0.7298,
        wdamp = 1.0,
        verbose = false)
    else
        for i in 1:maxiters
            ## Invoke GPU Kernel here
            update_particle_states_cpu!(prob, cpu_particles, sol_ref, w)
            w = w * wdamp
        end
    end

    return sol_ref[]
end
