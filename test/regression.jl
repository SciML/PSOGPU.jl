## Solving the rosenbrock problem
Random.seed!(1)

lb = @SArray ones(Float32, N)
lb = -1 * lb
ub = @SArray fill(Float32(10.0), N)

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

x0 = @SArray zeros(Float32, N)
p = @SArray Float32[1.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

n_particles = 1000

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = true),
    maxiters = 500)

@test sol.objective < 1e-4

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = false),
    maxiters = 500)

@test sol.objective < 1e-4

lb = @SVector fill(Float32(-Inf), N)
ub = @SVector fill(Float32(Inf), N)
prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub, N)

n_particles = 2000

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = true),
    maxiters = 500)

@test sol.objective < 1e-4

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = false),
    maxiters = 500)

@test sol.objective < 1e-4

prob = OptimizationProblem(rosenbrock, x0, p)

n_particles = 2000

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = true),
    maxiters = 500)

@test sol.objective < 1e-4

sol = solve(prob,
    ParallelPSOKernel(n_particles; threaded = false),
    maxiters = 500)

@test sol.objective < 1e-4
