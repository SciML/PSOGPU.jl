using PSOGPU, Optimization, CUDA, StaticArrays

function objf(x, p)
    return 1 - x[1]^2 - x[2]^2
end

optprob = OptimizationFunction(objf, Optimization.AutoEnzyme())
x0 = rand(2)
x0 = SVector{2}(x0)
prob = OptimizationProblem(optprob, x0)
l1 = objf(x0, nothing)
sol = Optimization.solve(prob,
    PSOGPU.LBFGS(),
    maxiters = 10)

N = 10
function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end
x0 = @SArray rand(Float32, N)
p = @SArray Float32[1.0, 100.0]
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, p)
l0 = rosenbrock(x0, p)

@time sol = Optimization.solve(prob,
    PSOGPU.LBFGS(; threshold = 7),
    maxiters = 20)
@show sol.objective
@time sol = Optimization.solve(prob,
    PSOGPU.ParallelPSOKernel(100, backend = CUDABackend()),
    maxiters = 100)
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.HybridPSO(; backend = CUDABackend()),
    maxiters = 30)
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.HybridPSO(; local_opt = PSOGPU.BFGS(), backend = CUDABackend()),
    maxiters = 30)
@show sol.objective

optf = OptimizationFunction(rosenbrock, Optimization.AutoEnzyme())
prob = OptimizationProblem(optf, x0, p)
l0 = rosenbrock(x0, p)

@time sol = Optimization.solve(prob,
    PSOGPU.LBFGS(; threshold = 7),
    maxiters = 20)
@show sol.objective
@time sol = Optimization.solve(prob,
    PSOGPU.ParallelPSOKernel(100, backend = CUDABackend()),
    maxiters = 100)
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.HybridPSO(; backend = CUDABackend()),
    locaL_maxiters = 30)
@show sol.objective

@time sol = Optimization.solve(prob,
    PSOGPU.HybridPSO(; local_opt = PSOGPU.BFGS(), backend = CUDABackend()),
    local_maxiters = 30)
@show sol.objective
