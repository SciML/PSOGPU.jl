using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "CPU")

@safetestset "Regression tests" include("./regression.jl")
@safetestset "Reinitialization tests" include("./reinit.jl")

#TODO: Curent throws warning for redefinition with the use of @testset multiple times. Migrate to TestItemRunners.jl
@testset for GROUP in unique(("CPU", GROUP))
    @testset "$(GROUP) optimizers tests" include("./gpu.jl")
    @testset "$(GROUP) optimizers with constraints tests" include("./constraints.jl")
    @testset "$(GROUP) hybrid optimizers" include("./lbfgs.jl")
end
