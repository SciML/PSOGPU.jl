using SimpleChains, StaticArrays, OrdinaryDiffEq

u0 = @SArray [2.0, 0.0]
datasize = 30
tspan = (0.0, 1.5)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix [-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(static(2),
    Activation(x -> x .* 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2)))

p_nn = SimpleChains.init_params(sc, Float64)

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps)
end

nn = (u, p, t) -> sc(u, p)

p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

prob_nn = ODEProblem(nn_fn, u0, tspan, (sc, p_static))

n_particles = 10_000

function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Tsit5(), saveat = t))
    sum(abs2, data .- pred)
end

lb = SVector{length(optprob.u0), eltype(optprob.u0)}(fill(eltype(optprob.u0)(-1),
    length(optprob.u0))...)
ub = SVector{length(optprob.u0), eltype(optprob.u0)}(fill(eltype(optprob.u0)(1),
    length(optprob.u0))...)

optprob = OptimizationProblem(loss, prob_nn.p[2], (prob_n, tsteps); lb = lb, ub = ub)

using PSOGPU
using CUDA

gbest, particles = PSOGPU.init_particles(optprob, n_particles)

gpu_data = cu([SVector{length(prob_nn.u0), eltype(prob_nn.u0)}(@view data[:, i])
               for i in 1:length(tsteps)])

gpu_particles = cu(particles)

CUDA.allowscalar(false)

gg = copy(gpu_particles)

function prob_func(prob, gpu_particle)
    return remake(prob, p = (prob.p[1], gpu_particle.position))
end

@time gsol = PSOGPU.parameter_estim_ode!(prob_nn,
    gg,
    gbest,
    gpu_data,
    lb,
    ub; saveat = tsteps, dt = 0.1, prob_func = prob_func, maxiters = 1000)
