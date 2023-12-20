using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")

if GROUP != "CPU"
    @safetestset "GPU optimizers tests" include("./gpu.jl")
end
