function uniform(dim::Int, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    arr = rand(T, dim)
    @inbounds for i in 1:dim
        arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i]
    end
    return arr
end

function init_particles(prob, n_particles)
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

function init_particles(prob, population, ::CPU)
    dim = length(prob.u0)
    lb = prob.lb
    ub = prob.ub
    cost_func = prob.f

    gbest_position = uniform(dim, lb, ub)
    gbest = Gbest(gbest_position, cost_func(gbest_position, data_dict))

    particles = Particle[]
    for i in 1:population
        position = uniform(dim, lb, ub)
        velocity = zeros(dim)
        cost = cost_func(position, data_dict)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, Particle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, particles
end
