struct HybridPSOLBFGS
    pso::PSOAlgorithm
    lbfgs::LBFGS
end

function HybridPSOLBFGS(; pso = PSOGPU.ParallelPSOKernel(100 ; global_update = false), lbfgs = LBFGS())
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

@kernel function lbfgs_run!(nlcache, x0, result)
    i = @index(Global, Linear)
    @set! nlcache.u = x0
    @show nlcache.u
    res = solve!(nlcache)
    @show res
    result[i] = res
end

function SciMLBase.__solve(cache::Optimization.OptimizationCache{F, RC, LB, UB, LC, UC, S, O, D, P, C}) where {F, RC, LB, UB, LC, UC, S, O <: HybridPSOLBFGS, D, P, C}
    t0 = time()
    psoalg = cache.opt.pso
    lbfgsalg = cache.opt.lbfgs
    @set! cache.opt = psoalg
    sol_pso = solve!(cache)

    x0s = sol_pso.original
    cache.lb = nothing
    cache.ub = nothing

    if cache.u0 isa SVector
        _g = (θ, _p = nothing) -> (G = KernelAbstractions.allocate(cache.opt.backend, eltype(cache.u0), size(cache.u0)); cache.f.grad(G, θ); return G) 
    else
        _g = (G, θ, _p=nothing) -> cache.f.grad(G, θ)
    end
    # @show cache.u0
    nlprob = NonlinearProblem(NonlinearFunction(_g), cache.u0)
    nlcache = init(nlprob, LimitedMemoryBroyden(; threshold = lbfgsalg.m, linesearch = LiFukushimaLineSearch()))
    @show nlcache
    backend = lbfgsalg.backend
    kernel = lbfgs_run!(backend)
    result = KernelAbstractions.allocate(lbfgsalg.backend, SciMLBase.NonlinearSolution, size(cache.u0))
    for x0 in x0s
        @show x0
        kernel(nlcache, x0, result; ndrange = (1,))
    end
    # @show result
    t1 = time()
    @show result
    sol_bfgs = [cache.f(θ, cache.p) for θ in getfield.(result, :u)]
    @show sol_bfgs
    minobj, ind = findmin(sol_bfgs)

    SciMLBase.build_solution(cache, cache.opt,
        result[ind].u, minobj)
end