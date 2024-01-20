using PSOGPU, StaticArrays, KernelAbstractions, Optimization
using CUDA

device!(2)

N = 3
function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end
x0 = @SArray zeros(Float32, N)
p = @SArray Float32[1.0, 100.0]
lb = @SArray fill(Float32(-1.0), N)
ub = @SArray fill(Float32(10.0), N)
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p; lb = lb, ub = ub)

n_particles = 10_000

sol = solve(prob, ParallelSyncPSOKernel(n_particles; backend = CPU()), maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelSyncPSOKernel(n_particles; backend = CUDABackend()),
    maxiters = 500)

@show sol.objective
@show sol.stats.time

sol = solve(prob,
    ParallelPSOKernel(n_particles; backend = CUDABackend(), global_update = false),
    maxiters = 500)

@show sol.objective
@show sol.stats.time

cpu_times = Float64[]
gpu_sync_times = Float64[]
gpu_async_times = Float64[]

Ns = [2^i for i in 3:2:20]
for n_particles in Ns
    @info n_particles
    ## CPU solve
    backend = CPU()
    opt = ParallelSyncPSOKernel(n_particles; backend)
    init_gbest, particles = PSOGPU.init_particles(prob, opt, typeof(prob.u0))

    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)

    backend_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))

    copyto!(backend_particles, particles)

    PSOGPU.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    el_time = @elapsed PSOGPU.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    push!(cpu_times, el_time)
    ## GPU Solve

    backend = CUDABackend()

    opt = ParallelSyncPSOKernel(n_particles; backend)

    particles_eltype = eltype(particles) === Float64 ? Float32 : eltype(particles)

    backend_particles = KernelAbstractions.allocate(backend,
        particles_eltype,
        size(particles))

    copyto!(backend_particles, particles)

    PSOGPU.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    el_time = @elapsed PSOGPU.vectorized_solve!(prob,
        init_gbest,
        backend_particles,
        opt; maxiters = 500)

    push!(gpu_sync_times, el_time)

    opt = ParallelPSOKernel(n_particles; backend, global_update = false)

    gpu_init_gbest = KernelAbstractions.allocate(backend, typeof(init_gbest), (1,))
    copyto!(gpu_init_gbest, [init_gbest])

    PSOGPU.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    el_time = @elapsed PSOGPU.vectorized_solve!(prob,
        gpu_init_gbest,
        backend_particles,
        opt, Val(opt.global_update); maxiters = 500)

    push!(gpu_async_times, el_time)
end

@show cpu_times
@show gpu_sync_times
@show gpu_async_times

using Plots

xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)

yticks = 10 .^ round.(range(1, -3, length = 11), digits = 2)

plt = plot(Ns,
    gpu_sync_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: GPU",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Bechmarking the 10D Rosenbrock Problem",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    dpi = 600,
    color = :Green)

plt = plot!(Ns,
    cpu_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelSyncPSOKernel: CPU",
    marker = :circle,
    color = :Orange)

plt = plot!(Ns,
    gpu_async_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "ParallelPSOKernel (Async): GPU",
    marker = :circle,
    color = :Green)

@show mean(cpu_times ./ gpu_sync_times)

@show mean(cpu_times ./ gpu_async_times)
