# Based on: https://stackoverflow.com/questions/65342388/why-my-code-in-julia-is-getting-slower-for-higher-iteration

mutable struct Particle{T}
    position::Array{T, 1}
    velocity::Array{T, 1}
    cost::T
    best_position::Array{T, 1}
    best_cost::T
end
mutable struct Gbest{T}
    position::Array{T, 1}
    cost::T
end

function PSO(prob::OptimizationProblem;
        maxiters = 100,
        population = 100,
        c1 = 1.4962,
        c2 = 1.4962,
        w = 0.7298,
        wdamp = 1.0,
        verbose = false)
    dim = length(prob.u0)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), dim) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), dim) : prob.ub
    cost_func = prob.f
    p = prob.p

    gbest, particles = init_particles(prob, population, true, CPU())

    # main loop
    for iter in 1:maxiters
        Threads.@threads for i in 1:population
            particles[i].velocity .= w .* particles[i].velocity .+
                                     c1 .* rand(dim) .* (particles[i].best_position .-
                                      particles[i].position) .+
                                     c2 .* rand(dim) .*
                                     (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, lb)
            particles[i].position .= min.(particles[i].position, ub)

            particles[i].cost = cost_func(particles[i].position, prob.p)

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
    gbest
end

function theta_paper(x)
    if x < 0.001
        return 10.0
    elseif x <= 0.1
        return 20.0
    elseif x <= 1.0
        return 100.0
    else
        return 300.0
    end
end

function gamma_paper(x)
    if x < 1
        return 1
    else
        return 2
    end
end

function update_particle_states_cpu!(prob, particles, gbest_ref, w, lb, ub, iter;
        # c1 = 1.4962f0,
        # c2 = 1.4962f0,
        c1 = 2.0f0,
        c2 = 2.0f0,
        theta = theta_paper, gamma = gamma_paper, h = (x) -> sqrt)
    # i = 1

    ## Access the particle

    # gpu_particles = convert(MArray, gpu_particles)

    gbest = gbest_ref[]

    for i in eachindex(particles)
        @inbounds particle = particles[i]
        ## Update velocity

        χ = 0.73

        updated_velocity = w * particle.velocity +
                           c1 .* randn(typeof(particle.velocity)) .*
                           (particle.best_position -
                            particle.position) +
                           c2 * randn(typeof(particle.velocity)) .*
                           (gbest.position - particle.position)

        updated_velocity = χ * updated_velocity

        updated_velocity = min.(updated_velocity, Ref(4.0))

        @set! particle.velocity = updated_velocity

        @set! particle.position = particle.position + particle.velocity

        update_pos = max.(particle.position, lb)
        update_pos = min.(update_pos, ub)

        @set! particle.position = update_pos

        if !isnothing(prob.f.cons)
            penalty = calc_penalty(particle.position, prob, iter + 1; theta, gamma, h)
            @set! particle.cost = prob.f(particle.position, prob.p) + penalty
        else
            @set! particle.cost = prob.f(particle.position, prob.p)
        end

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
        w = 1.2f0,
        wdamp = 0.9975f0,
        debug = false,
        theta = theta_paper, gamma = gamma_paper, h = sqrt)
    sol_ref = Ref(gbest)
    lb = prob.lb === nothing ? fill(eltype(prob.u0)(-Inf), length(prob.u0)) : prob.lb
    ub = prob.ub === nothing ? fill(eltype(prob.u0)(Inf), length(prob.u0)) : prob.ub
    for i in 1:maxiters
        update_particle_states_cpu!(prob,
            cpu_particles,
            sol_ref,
            w,
            lb,
            ub,
            i;
            theta = theta_paper,
            gamma = gamma_paper,
            h = sqrt)
        w = w * wdamp
    end

    return sol_ref[]
end
