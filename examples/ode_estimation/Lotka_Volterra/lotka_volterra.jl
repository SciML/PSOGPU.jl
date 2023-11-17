using StaticArrays, SciMLBase, OrdinaryDiffEq

function f(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    return SVector{2}(dx, dy)
end

u0 = @SArray [1.0; 1.0]
tspan = (0.0, 10.0)
p = @SArray [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(f, u0, tspan, p)

t = 0.0:0.1:10.0

function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Tsit5(), saveat = t))
    sum(abs2, 1 .- pred)
end

optprob = OptimizationProblem(loss, prob.p, (prob, t))

using PSOGPU
using CUDA

CUDA.allowscalar(false)

n_particles = 10_000

gbest, particles = PSOGPU.init_particles(optprob, n_particles)

gpu_data = cu([@SArray ones(length(prob.u0))])

gpu_particles = cu(particles)

lb = SVector{length(optprob.u0), eltype(optprob.u0)}(fill(eltype(optprob.u0)(-Inf),
    length(optprob.u0))...)

ub = SVector{length(optprob.u0), eltype(optprob.u0)}(fill(eltype(optprob.u0)(Inf),
    length(optprob.u0))...)

@time gsol = PSOGPU.parameter_estim_ode!(prob,
    gpu_particles,
    gbest,
    gpu_data,
    lb,
    ub; saveat = t, dt = 0.1)
