using ModelingToolkit, Optimization, PSOGPU

@variables x[1:2]

obj = (x[1] - 2)^2 + (x[2] - 1)^2

cons = [
    x[1] ~ 2*x[2] - 1,
    (x[1]^2)/4 + x[2]^2 - 1 ≲ 0
]

@named sys = OptimizationSystem(obj, x[1:2], [], constraints=cons)
prob = OptimizationProblem(sys, [1.0, 1.0], lb = [0.0, 0.0], ub = [10.0, 10.0])

sol = solve(prob, ParallelPSOKernel(100; gpu = false, threaded = true), maxiters = 1000)