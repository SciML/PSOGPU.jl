using StaticArrays, SciMLBase

function f(u, p, t)
    dx = p[1] * u[1] - u[1] * u[2]
    dy = -3 * u[2] + u[1] * u[2]
    return SVector{2}(dx,dy)
end

u0 = @SArray [1.0; 1.0]
tspan = (0.0, 10.0)
p = @SArray [1.5]
prob = ODEProblem(f, u0, tspan, p)

using OrdinaryDiffEq
sol = solve(prob, Tsit5())
t = collect(range(0f0, stop = 10.0, length = 200))
using RecursiveArrayTools # for VectorOfArray
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])

data = convert(Array, randomized)

n_particles = 1000

maxiters = 1000

n = 1

lb = @SArray ones(n)

lb = -10*lb

ub = @SArray ones(n)

ub = 10*ub


function loss(u,p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Tsit5(), saveat = t))
    sum(abs2, data .- pred)
end

optprob = OptimizationProblem(loss, [1.0], (prob,t);lb, ub)

using PSOGPU
using CUDA


gbest, particles = PSOGPU.init_particles(optprob, n_particles)

gpu_data = cu([SVector{length(prob.u0), eltype(prob.u0)}(@view data[:,i]) for i in 1:length(t)])


gpu_particles = cu(particles)


@time gsol = PSOGPU.parameter_estim_ode!(prob,
    gpu_particles,
    gbest,
    gpu_data,
    lb,
    ub; saveat = t, dt = 0.1)
