## Solving the rosenbrock problem

lb = @SArray ones(Float32, N)
lb = -1 * lb
ub = @SArray ones(Float32, N)

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

x0 = @SArray zeros(Float32, N)
p = @SArray Float32[2.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n_particles = 100

sol = solve(prob, ParallelPSO(n_particles; gpu = false, threaded = true), maxiters = 500)

@test norm(sol.u - ub) < 3e-2

sol = solve(prob, ParallelPSO(n_particles; gpu = false, threaded = false), maxiters = 500)

@test norm(sol.u - ub) < 3e-2
