const GROUP = get(ENV, "GROUP", "CPU")
const backend = if GROUP == "CUDA"
    using CUDA
    CUDA.CUDABackend()
elseif GROUP == "AMDGPU"
    using AMDGPU
    AMDGPU.ROCBackend()
else
    using KernelAbstractions
    backend = CPU()
end
