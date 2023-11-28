using SimpleChains, StaticArrays, OrdinaryDiffEq, SciMLSensitivity
u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(static(2),
    Activation(x -> x .* 3),
    TurboDense{true}(tanh, static(2)),
    TurboDense{true}(identity, static(2)))

p_nn = SimpleChains.init_params(sc)

function nn_fn(u::T, p, t)::T where {T}
    nn, ps = p
    return nn(u, ps, t)
end

nn = (u, p, t) -> sc(u, p)

p_static = SArray{Tuple{size(p_nn)...}}(p_nn...)

prob_nn = ODEProblem(nn_fn, u0, tspan, (sc, p_static))


function predict_neuralode(p)
    _prob = remake(_prob; p = (sc, p))
    Array(solve(_monteprob, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0), dt = 0.1f0,
        trajectories = 2, adaptive = true, saveat = tsteps)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end

sol = predict_neuralode(p_static)
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


optprob = OptimizationProblem(loss_neuralode, [1.0], (prob,t);lb, ub)

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

