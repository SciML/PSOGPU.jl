## Testing compatibility of the function prob

## Use lb and ub either as StaticArray or pass them separately as CuArrays
## Passing as CuArrays makes more sense, or maybe SArray? The based on no. of dimension

using CUDA, StaticArrays, PSOGPU, Setfield

device!(2)

lb = @SArray [-1.0, -1.0]

ub = @SArray [1.0, 1.0]

struct PSOParticle{T1, T2 <: eltype(T1)}
    position::T1
    velocity::T1
    cost::T2
    best_position::T1
    best_cost::T2
end

struct PSOGBest{T1, T2 <: eltype(T1)}
    position::T1
    cost::T2
end

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

function update_particle_states!(prob, gpu_particles, gbest, w; c1 = 1.4962f0,
    c2 = 1.4962f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(gpu_particles) && return

    # i = 1

    ## Access the particle

    # gpu_particles = convert(MArray, gpu_particles)

    @inbounds particle = gpu_particles[i]
    ## Update velocity

    updated_velocity = w * particle.velocity +
                       c1 .* rand(typeof(particle.velocity)) .* (particle.best_position -
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

    # sync_threads();
    if particle.best_cost < gbest.cost
        @set! gbest.position = particle.best_position
        @set! gbest.cost = particle.best_cost
    end

    gpu_particles[i] = particle

    # gpu_particles = convert(SArray, gpu_particles)
    return nothing
end

n_particles = 10_000
gbest, particles = init_particles(prob, n_particles)

gpu_particles = CuArray(particles)

function pso_solve_gpu(prob,
    gbest,
    gpu_particles;
    max_iters = 100,
    w = 0.7298f0,
    wdamp = 1.0f0,
    debug = false)

    ## Initialize stuff

    kernel = @cuda launch=false update_particle_states!(prob, gpu_particles, gbest, w)

    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(gpu_particles), config.threads)

    blocks = max(cld(length(gpu_particles), threads), config.blocks)
    threads = cld(length(gpu_particles), blocks)

    for i in 1:max_iters
        ## Invoke GPU Kernel here
        kernel(prob, gpu_particles, gbest, w)
        w = w * wdamp
    end

    return gbest
end

function update_particle_states_cpu!(prob, particles, gbest, w; c1 = 1.4962f0,
    c2 = 1.4962f0)
    # i = 1

    ## Access the particle

    # gpu_particles = convert(MArray, gpu_particles)

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

        # sync_threads();
        if particle.best_cost < gbest.cost
            @set! gbest.position = particle.best_position
            @set! gbest.cost = particle.best_cost
        end

        particles[i] = particle
    end
    return nothing
end

function pso_solve_cpu(prob,
    gbest,
    cpu_particles;
    max_iters = 100,
    w = 0.7298f0,
    wdamp = 1.0f0,
    debug = false)
    for i in 1:max_iters
        ## Invoke GPU Kernel here
        update_particle_states_cpu!(prob, cpu_particles, gbest, w)
        w = w * wdamp
    end

    return gbest
end

using BenchmarkTools
@benchmark pso_solve_cpu($prob, $gbest, $particles)

@benchmark pso_solve_gpu($prob, $gbest, $gpu_particles)

## Solving the rosenbrock problem

lb = @SArray Float32[-1.0, -1.0]

ub = @SArray Float32[1.0, 1.0]

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = @SArray zeros(Float32, 2)
p = @SArray Float32[2.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n_particles = 10_000
gbest, particles = init_particles(prob, n_particles)

gpu_particles = cu(particles)

t1 = @elapsed pso_solve_cpu(prob, gbest, particles)

t2 = @elapsed pso_solve_gpu(prob, gbest, gpu_particles)
