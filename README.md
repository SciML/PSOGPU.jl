# PSOGPU

[![Build Status](https://github.com/utkarsh530/PSOGPU.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/utkarsh530/PSOGPU.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/utkarsh530/PSOGPU.jl/graph/badge.svg?token=H5U5UAIRXX)](https://codecov.io/gh/utkarsh530/PSOGPU.jl)

Accelerating convex/non-convex optimization with GPUs using Particle-Swarm based methods

Supports generic Julia's SciML interface

```julia

using PSOGPU, StaticArrays, CUDA

lb = @SArray [-1.0f0, -1.0f0]
ub = @SArray [1.0f0, 1.0f0]

function rosenbrock(x, p)
    sum(p[2] * (x[i + 1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:(length(x) - 1))
end

x0 = @SArray zeros(Float32, 2)
p = @SArray Float32[2.0, 100.0]

prob = OptimizationProblem(rosenbrock, x0, p; lb = lb, ub = ub)

## select the no. of particles
n_particles = 100

## initialize particles on the grid and the best current position
gbest, particles = PSOGPU.init_particles(prob, n_particles)

## Offload particles to GPU
particles = cu(particles)

sol = PSOGPU.pso_solve_gpu!(prob, gbest, particles)

sol[].position
```
