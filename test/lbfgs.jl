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
    PSOGPU.LBFGS(1e-3, 10),
    maxiters = 10)

N = 10
function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end
x0 = zeros(Float32, N)
p = Float32[1.0, 100.0]
optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(optf, x0, p)

sol = Optimization.solve(prob,
    PSOGPU.LBFGS(1e-3, 10),
    maxiters = 10
    )