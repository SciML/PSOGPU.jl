using Lux, Optimization, Zygote, OrdinaryDiffEq, CUDA, SciMLSensitivity, Random, ComponentArrays
import DiffEqFlux: NeuralODE 

CUDA.allowscalar(false) # Makes sure no slow operations are occuring

#rng for Lux.setup
rng = Random.default_rng()
# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = gpu(solve(prob_trueode, Tsit5(), saveat = tsteps))


dudt2 = Chain(x -> x.^3, Dense(2, 50, tanh), Dense(50, 2))
u0 = Float32[2.0; 0.0] |> gpu
p, st = Lux.setup(rng, dudt2) 
p = p |> ComponentArray |> gpu
st = st |> gpu

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  gpu(first(prob_neuralode(u0,p,st)))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
# Callback function to observe training
list_plots = []

iter = 0
callback = function (p, l, pred; doplot = false)
  global list_plots, iter
  if iter == 0
    list_plots = []
  end
  iter += 1
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, Array(ode_data[1,:]), label = "data")
  scatter!(plt, tsteps, Array(pred[1,:]), label = "prediction")
  push!(list_plots, plt)
  if doplot
    display(plot(plt))
  end
  return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
result_neuralode = Optimization.solve(optprob, ParallelPSO(100, gpu = true), maxiters = 300)
