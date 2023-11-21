using ParameterizedFunctions, OrdinaryDiffEq, Optimization
using OptimizationBBO, Plots, ForwardDiff, BenchmarkTools
using StaticArrays

g1 = @ode_def LorenzExample begin
    dx = σ*(y-x)
    dy = x*(ρ-z) - y
    dz = x*y - β*z
end σ ρ β
p = [10.0,28.0,2.66] # Parameters used to construct the dataset
r0 = [1.0; 0.0; 0.0]                #[-11.8,-5.1,37.5] PODES Initial values of the system in space # [0.1, 0.0, 0.0]
tspan = (0.0, 30.0)                 # PODES sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(g1, r0, tspan,p)
tspan2 = (0.0, 3.0)                 # Xiang test sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(g1, r0, tspan2,p)

dt = 30.0/3000
tf = 30.0
tinterval = 0:dt:tf
t  = collect(tinterval)

h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

# Generate Data
data_sol_short = solve(prob_short,Vern9(),saveat=t_short,reltol=1e-9,abstol=1e-9)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob,Vern9(),saveat=t,reltol=1e-9,abstol=1e-9)
sizesol = size(data_sol)
data = Array(data_sol)

plot(data_sol_short,vars=(1,2,3)) # the short solution
plot(data_sol,vars=(1,2,3)) # the longer solution
interpolation_sol = solve(prob,Vern7(),saveat=t,reltol=1e-12,abstol=1e-12)
plot(interpolation_sol,vars=(1,2,3))

function loss(u, p)
    odeprob, t = p
    prob = remake(odeprob; p = u)
    pred = Array(solve(prob, Vern9(), saveat = t))
    sum(abs2, data .- pred)
end

Xiang2015Bounds = Tuple{Float64, Float64}[(9, 11), (20, 30), (2, 3)] # for local optimizations
xlow_bounds = @SVector [9.0,20.0,2.0]
xhigh_bounds = @SVector [11.0,30.0,3.0]
LooserBounds = Tuple{Float64, Float64}[(0, 22), (0, 60), (0, 6)] # for global optimization
GloIniPar = @SVector [0.0, 0.5, 0.1] # for global optimizations
LocIniPar = @SVector [9.0, 20.0, 2.0] # for local optimization

optprob = OptimizationProblem(loss, LocIniPar, (prob, t), lb = xlow_bounds, ub = xhigh_bounds)

using PSOGPU
using CUDA

CUDA.allowscalar(false)

n_particles = 10_000

gbest, particles = PSOGPU.init_particles(optprob, n_particles)

gpu_data = cu(data)

gpu_particles = cu(particles)

@time gsol = PSOGPU.parameter_estim_ode!(prob,
    gpu_particles,
    gbest,
    gpu_data,
    lb = xlow_bounds, ub = xhigh_bounds; saveat = t, dt = 0.1)