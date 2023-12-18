
abstract type PSOAlogrithm end

struct ParallelPSOKernel{Backend} <: PSOAlogrithm
    num_particles::Int
    async::Bool
    backend::Backend
end

struct ParallelSyncPSOKernel{Backend} <: PSOAlogrithm
    num_particles::Int
    backend::Backend
end

struct ParallelPSOArray <: PSOAlogrithm
    num_particles::Int
end

struct SerialPSO <: PSOAlogrithm
    num_particles::Int
end

function ParallelPSOKernel(num_particles::Int;
        async = false, backend = CPU())
    ParallelPSOKernel(num_particles, async, backend)
end

function ParallelSyncPSO(num_particles::Int;
        backend = CPU())
    ParallelSyncPSOKernel(num_particles, async, backend)
end

SciMLBase.allowsbounds(::PSOAlogrithm) = true
