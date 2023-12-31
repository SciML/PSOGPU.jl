using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")

if GROUP != "CPU"
    @safetestset "GPU optimizers tests" include("./gpu.jl")
    @safetestset "GPU optimizers with constriants tests" include("./constraints.jl")
end
