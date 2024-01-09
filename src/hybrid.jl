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

@kernel function lbfgs_run!(nlcaches, x0s, result)
    i = @index(Global, Linear)
    # nlcache = reinit!(nlcaches[i], x0s[i])
    # @show nlcache.u
    res = solve!(nlcaches[i])
    # @show res
    # @show res.resid
    result[i] = res
end

@kernel function simplelbfgs_run!(nlprob, x0s, result)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    result[i] = solve(nlcache, SimpleLimitedMemoryBroyden(; threshold = 10))
end

@kernel function simplebfgs_run!(nlprob, x0s, result)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    result[i] = solve(nlcache, SimpleBroyden())
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
    @show length(x0s)
    if cache.u0 isa SVector
        G = KernelAbstractions.allocate(cache.opt.backend, eltype(cache.u0), size(cache.u0));
        _g = (θ, _p = nothing) -> (cache.f.grad(G, θ); return G)
    else
        _g = (G, θ, _p=nothing) -> cache.f.grad(G, θ)
    end
    # @show cache.u0
    # nlcaches = [init(NonlinearProblem(NonlinearFunction(_g), x0), LimitedMemoryBroyden(; threshold = lbfgsalg.m, linesearch = LiFukushimaLineSearch())) 
    #     for x0 in x0s
    # ]
    # @show nlcaches[1]
    # @show ismutable(nlcaches[1])
    backend = lbfgsalg.backend
    # kernel = lbfgs_run!(backend)
    # result = KernelAbstractions.allocate(lbfgsalg.backend, SciMLBase.NonlinearSolution, length(x0s))

    # kernel(nlcaches, x0s, result; ndrange = length(x0s))

    # kernel = simplelbfgs_run!(backend)
    # result = KernelAbstractions.allocate(backend, SciMLBase.NonlinearSolution, length(x0s))
    # nlprob = NonlinearProblem(NonlinearFunction(_g), cache.u0)
    # kernel(nlprob, x0s, result; ndrange = length(x0s))

    kernel = simplebfgs_run!(backend)
    result = KernelAbstractions.allocate(backend, SciMLBase.NonlinearSolution, length(x0s))
    nlprob = NonlinearProblem(NonlinearFunction(_g), cache.u0)
    kernel(nlprob, x0s, result; ndrange = length(x0s))

    # @show result
    t1 = time()
    @show result
    sol_bfgs = [cache.f(θ, cache.p) for θ in getfield.(result, Ref(:u))]
    @show sol_bfgs
    minobj, ind = findmin(x -> isnan(x) ? Inf : x ,sol_bfgs)

    SciMLBase.build_solution(cache, cache.opt,
        result[ind].u, minobj)
end