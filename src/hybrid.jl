struct HybridPSOLBFGS
    pso::PSOAlgorithm
    lbfgs::LBFGS
end

function HybridPSOLBFGS(; pso = PSOGPU.ParallelPSOKernel(100 ; global_update = false), lbfgs = LBFGS(1e-3, 10))
    HybridPSOLBFGS(pso, lbfgs)
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::HybridPSOLBFGS, ensemblealg = EnsembleThreads(), args...; kwargs...)
    lb, ub = check_init_bounds(prob)
    prob = remake(prob; lb = lb, ub = ub)

    sol_pso = solve(prob, opt.pso, args...; kwargs...)

    x0s = sol_pso.original
    prob = remake(prob; lb = nothing, ub = nothing)
    ensemble_prob = EnsembleProblem(prob, x0s)

    sol_lbfgs = solve(ensemble_prob, opt.lbfgs, ensemblealg, args...; trajectories = opt.pso.num_particles)

    minobj, ind = findmin(i -> sol_lbfgs[i].objective, 1:length(sol_lbfgs))

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        sol_lbfgs[ind].u, minobj)
end