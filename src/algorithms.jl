
abstract type PSOAlogrithm end

"""
```julia
ParallelPSOKernel(num_particles = 100)
```

Particle Swarm Optimization on a GPU. Creates and launches a kernel which updates the particle states in parallel
on a GPU. Static Arrays for parameters in the `OptimizationProblem` are required for successful GPU compilation.

## Positional arguments:

- num_particles: Number of particles in the simulation
- global_update: defaults to `true`. Setting it to false allows particles to evolve completely on their own,
  i.e. no information is sent about the global best position.
- backend: defaults to `CPU()`. The KernelAbstractions backend for performing the computation.

## Limitations

Running the optimization with `global_update=true` updates the global best positions with possible thread races.
This is the price to be paid to fuse all the updates into a single kernel. Techniques such as queue lock and atomic
updates can be used to fix this.

"""
struct ParallelPSOKernel{Backend} <: PSOAlogrithm
    num_particles::Int
    global_update::Bool
    backend::Backend
end

"""
```julia
ParallelSyncPSOKernel(num_particles = 100)
```

Particle Swarm Optimization on a GPU. Creates and launches a kernel which updates the particle states in parallel
on a GPU. However, it requires a synchronization after each generation to calculate the global best position of the particles.

## Positional arguments:

- num_particles: Number of particles in the simulation
- backend: defaults to `CPU()`. The KernelAbstractions backend for performing the computation.

"""
struct ParallelSyncPSOKernel{Backend} <: PSOAlogrithm
    num_particles::Int
    backend::Backend
end

"""
```julia
ParallelPSOArray(num_particles = 100)
```
Particle Swarm Optimization on a CPU. It keeps the arrays used in particle data structure
to be Julia's `Array`, which may be better for high-dimensional problems.

## Positional arguments:

- num_particles: Number of particles in the simulation


## Limitations

Running the optimization updates the global best positions with possible thread races.
This is the price to be paid to fuse all the updates into a single kernel. Techniques such as queue lock and atomic
updates can be used to fix this.

"""
struct ParallelPSOArray <: PSOAlogrithm
    num_particles::Int
end

"""
```julia
SerialPSO(num_particles = 100)
```
Serial Particle Swarm Optimization on a CPU.

## Positional arguments:

- num_particles: Number of particles in the simulation

"""
struct SerialPSO <: PSOAlogrithm
    num_particles::Int
end

function ParallelPSOKernel(num_particles::Int;
        global_update = false, backend = CPU())
    ParallelPSOKernel(num_particles, global_update, backend)
end

function ParallelSyncPSOKernel(num_particles::Int;
        backend = CPU())
    ParallelSyncPSOKernel(num_particles, backend)
end

SciMLBase.allowsbounds(::PSOAlogrithm) = true
