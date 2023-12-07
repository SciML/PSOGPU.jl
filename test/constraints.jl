using ModelingToolkit, Optimization, PSOGPU

@variables x[1:2]

obj = (x[1] - 2)^2 + (x[2] - 1)^2

cons = [
    -x[1] + 2 * x[2] - 1 ≲ 1e-5,
    +x[1] - 2 * x[2] + 1 ≲ 1e-5,
    (x[1]^2) / 4 + x[2]^2 - 1 ≲ 1e-5,
]

@named sys = OptimizationSystem(obj, x[1:2], [], constraints = cons)

prob = OptimizationProblem(sys, [1.0, 1.0], lb = [0.0, 0.0], ub = [2.0, 2.0])

using StaticArrays

T1 = Float64

opt_prob = remake(prob; u0 = SArray{Tuple{length(prob.u0)}, T1}(prob.u0),
    p = prob.p isa SciMLBase.NullParameters || prob.p === nothing ? prob.p :
        SArray{Tuple{length(prob.p)}, T1}(prob.p),
    lb = SArray{Tuple{length(prob.lb)}, T1}(prob.lb),
    ub = SArray{Tuple{length(prob.ub)}, T1}(prob.ub))

sol = solve(opt_prob,
    ParallelPSOKernel(100; gpu = false, threaded = false),
    maxiters = 1000)

########

using ModelingToolkit, Optimization, PSOGPU

@variables x[1:2]

obj = (x[1] - 10)^3 + (x[2] - 20)^3

cons = [
    100 - (x[1] - 5)^2 - (x[2] - 5)^2 ≲ 1e-5,
    (x[1] - 6)^2 + (x[2] - 5)^2 - 82.81 ≲ 1e-5,
]

@named sys = OptimizationSystem(obj, x[1:2], [], constraints = cons)

prob = OptimizationProblem(sys, [50.0, 50.0], lb = [13.0, 0.0], ub = [100.0, 100.0])

using StaticArrays

T1 = Float64

opt_prob = remake(prob; u0 = SArray{Tuple{length(prob.u0)}, T1}(prob.u0),
    p = prob.p isa SciMLBase.NullParameters || prob.p === nothing ? prob.p :
        SArray{Tuple{length(prob.p)}, T1}(prob.p),
    lb = SArray{Tuple{length(prob.lb)}, T1}(prob.lb),
    ub = SArray{Tuple{length(prob.ub)}, T1}(prob.ub))

init_gbest, particles = PSOGPU.init_particles(opt_prob, 100)

gbest = PSOGPU.pso_solve_cpu!(opt_prob, init_gbest, particles; maxiters = 1000)

sol = solve(opt_prob,
    ParallelPSOKernel(1000; gpu = false, threaded = false),
    maxiters = 1000)
