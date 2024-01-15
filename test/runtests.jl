using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")

if GROUP != "CPU"
    @safetestset "GPU optimizers tests" include("./gpu.jl")
    @safetestset "GPU optimizers with constraints tests" include("./constraints.jl")
    @safetestset "GPU hybrid optimizers" include("./lbfgs.jl")
end
