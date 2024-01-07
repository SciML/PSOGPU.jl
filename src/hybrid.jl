struct HybridPSOLBFGS
    pso::PSOAlgorithm
    lbfgs::LBFGS
end

function HybridPSOLBFGS(; pso = PSOGPU.ParallelPSOKernel(100 ; global_update = false), lbfgs = LBFGS(1e-6, 10))
    HybridPSOLBFGS(pso, lbfgs)
end

SciMLBase.supports_opt_cache_interface(opt::HybridPSOLBFGS) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::HybridPSOLBFGS,
    data = Optimization.DEFAULT_DATA; save_best = true,
    callback = (args...) -> (false),
    progress = false, kwargs...)
    return Optimization.OptimizationCache(prob, opt, data; save_best, callback, progress,
        kwargs...)
end

function SciMLBase.__solve(cache::Optimization.OptimizationCache{F, RC, LB, UB, LC, UC, S, O, D, P, C}) where {F, RC, LB, UB, LC, UC, S, O <: HybridPSOLBFGS, D, P, C}
    psoalg = cache.opt.pso
    lbfgsalg = cache.opt.lbfgs
    @set! cache.opt = psoalg
    sol_pso = solve!(cache)

    x0s = sol_pso.original
    cache.lb = nothing
    cache.ub = nothing

    @set! cache.opt = lbfgsalg
    cache_vector = map(x0s) do x0
        reinit!(cache, u0 = x0)
    end

    sol_lbfgs = solve!.(cache_vector)

    minobj, ind = findmin(i -> sol_lbfgs[i].objective, 1:length(sol_lbfgs))

    SciMLBase.build_solution(cache, cache.opt,
        sol_lbfgs[ind].u, minobj)
end