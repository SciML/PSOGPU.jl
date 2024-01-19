@kernel function _update_particle_states!(gpu_particles, lb, ub, gbest, w; c1 = 1.4962f0,
        c2 = 1.4962f0)
    i = @index(Global, Linear)
    if i <= length(gpu_particles)
        @inbounds particle = gpu_particles[i]

        updated_velocity = w .* particle.velocity .+
                           c1 .* rand(typeof(particle.velocity)) .*
                           (particle.best_position -
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

function parameter_estim_ode!(prob::ODEProblem, cache,
        lb,
        ub;
        ode_alg = GPUTsit5(),
        prob_func = default_prob_func,
        w = 0.72980f0,
        wdamp = 1.0f0,
        maxiters = 100, kwargs...)
    (losses, gpu_particles, gpu_data, gbest) = cache
    backend = get_backend(gpu_particles)
    update_states! = PSOGPU._update_particle_states!(backend)
    update_costs! = PSOGPU._update_particle_costs!(backend)

    improb = make_prob_compatible(prob)

    for i in 1:maxiters
        update_states!(gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = length(gpu_particles))

        probs = prob_func.(Ref(improb), gpu_particles)

        ts, us = vectorized_asolve(probs,
            prob,
            ode_alg; kwargs...)

        sum!(losses, (map(x -> sum(x .^ 2), gpu_data .- us)))

        update_costs!(losses, gpu_particles; ndrange = length(losses))

        best_particle = minimum(gpu_particles)

        gbest = PSOGPU.SPSOGBest(best_particle.best_position, best_particle.best_cost)
        w = w * wdamp
    end
    return gbest
end

@inline function ode_loss(particle, second_arg; kwargs...)
    prob, ode_alg = second_arg
    prob = remake(prob, p = particle)

    us = OrdinaryDiffEq.__solve(prob,
            ode_alg; kwargs...).u

    sum(abs2, gpu_data .- us)
end

function parameter_estim_odehybrid(prob::ODEProblem, cache,
    lb,
    ub;
    ode_alg = GPUTsit5(),
    prob_func = default_prob_func,
    w = 0.72980f0,
    wdamp = 1.0f0,
    maxiters = 100, 
    local_maxiters = 100,
    abstol = nothing,
    reltol = nothing,
    kwargs...)

    (losses, gpu_particles, gpu_data, gbest) = cache
    backend = get_backend(gpu_particles)
    update_states! = PSOGPU._update_particle_states!(backend)
    update_costs! = PSOGPU._update_particle_costs!(backend)

    improb = make_prob_compatible(prob)

    for i in 1:maxiters
        update_states!(gpu_particles,
            lb,
            ub,
            gbest,
            w;
            ndrange = length(gpu_particles))

        probs = prob_func.(Ref(improb), gpu_particles)

        ts, us = vectorized_asolve(probs,
            prob,
            ode_alg; kwargs...)
    
        sum!(losses, (map(x -> sum(x .^ 2), gpu_data .- us)))

        update_costs!(losses, gpu_particles; ndrange = length(losses))

        w = w * wdamp
    end

    f = Base.Fix2(ode_loss, (improb, Tsit5()))

    ∇f = (θ, p) -> autodiff_deferred(Reverse, f, Active, Active(θ))[1][1]

    kernel = simplebfgs_run!(backend)
    x0s = get_pos.(gpu_particles)
    result = KernelAbstractions.allocate(backend, typeof(prob.u0), length(x0s))
    nlprob = NonlinearProblem{false}(∇f, prob.p)

    nlalg = SimpleBroyden(; linesearch = Val(true))

    kernel(nlprob,
        x0s,
        result,
        nlalg,
        local_maxiters,
        abstol,
        reltol;
        ndrange = length(x0s))

    t1 = time()
    sol_bfgs = (x -> ode_loss(x, (prob, Tsit5()))).(result)
    sol_bfgs = (x -> isnan(x) ? convert(eltype(prob.p), Inf) : x).(sol_bfgs)

    minobj, ind = findmin(sol_bfgs)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(OptimizationFunction(ode_loss), prob.p), opt,
        view(result, ind), minobj)
end