abstract type PSOAlgorithm end

struct ParallelPSOCPU{T} <: PSOAlgorithm
    lb::T
    ub::T
    num_particles::Int
end

function ParallelPSOCPU(; num_particles = 3)
    ParallelPSOCPU{typeof(Inf)}(-Inf, Inf, num_particles)
end

function ParallelPSOCPU(lb::T, ub::T; num_particles = 3) where {T}
    ParallelPSOCPU{T}(lb, ub, num_particles)
end
