module PSOGPU

using SciMLBase

include("./algorithms.jl")
include("./pso_cpu.jl")

function SciMLBase.__solve(prob::OptimizationProblem,
    opt::ParallelPSOCPU,
    args...;
    kwargs...)
    if typeof(opt.lb) <: AbstractFloat
        lb = [opt.lb for i in 1:length(prob.u0)]
        ub = [opt.ub for i in 1:length(prob.u0)]
    else
        lb = opt.lb
        ub = opt.ub
    end

    problem = Problem(prob.f, length(prob.u0), lb, ub)

    max_iter = 101

    gbest, = PSO(problem, prob.p, max_iter = max_iter, population = opt.num_particles)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        gbest.position, gbest.cost)
end
export ParallelPSOCPU, OptimizationProblem, solve
end
