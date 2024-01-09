using PSOGPU, Optimization
using Zygote, StaticArrays

function objf(x, p)
    return x[1]^2 + x[2]^2
end

optprob = OptimizationFunction(objf, Optimization.AutoZygote())
x0 = zeros(2) .+ 1
prob = OptimizationProblem(optprob, x0)
l1 = objf(x0, nothing)
sol = Optimization.solve(prob,
    PSOGPU.LBFGS(),
    maxiters = 10)

N = 10
function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end
x0 = rand(Float32, N)
p = Float32[1.0, 100.0]
optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(optf, x0, p)
l0 = rosenbrock(x0, p)

@time sol = Optimization.solve(prob,
    PSOGPU.LBFGS(),
    maxiters = 100,
    )
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.ParallelPSOArray(100),
    maxiters = 100,
    )
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.HybridPSOLBFGS(pso = PSOGPU.ParallelPSOKernel(30)),
    EnsembleThreads(),
    maxiters = 100,
    )
@show sol.objective