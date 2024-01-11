struct HybridPSOLBFGS
    pso::PSOAlgorithm
    lbfgs::LBFGS
end

function HybridPSOLBFGS(; pso = PSOGPU.ParallelPSOKernel(100 ; global_update = false), lbfgs = LBFGS())
    HybridPSOLBFGS(pso, lbfgs)
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

@kernel function simplebfgs_run!(nlprob, x0s, result, maxiters)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    result[i] = solve(nlcache, SimpleBroyden(; linesearch = Val(true)), maxiters = maxiters).u
end

function SciMLBase.__solve(prob::SciMLBase.OptimizationProblem, opt::HybridPSOLBFGS, args...; maxiters = 1000, kwargs...)
    t0 = time()
    psoalg = opt.pso
    lbfgsalg = opt.lbfgs
    
    sol_pso = solve(prob, psoalg, args...; maxiters, kwargs...)

    x0s = sol_pso.original
    @show prob.u0
    @show x0s
    prob = remake(prob, lb = nothing, ub = nothing)
    @show length(x0s)
    # f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, 0)

    _g = (θ, _p = nothing) -> ForwardDiff.gradient(x -> prob.f(x, prob.p), θ) 
    # @show prob.u0
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
    # nlprob = NonlinearProblem(NonlinearFunction(_g), prob.u0)
    # kernel(nlprob, x0s, result; ndrange = length(x0s))

    kernel = simplebfgs_run!(backend)
    result = KernelAbstractions.allocate(backend, eltype(x0s), length(x0s))
    nlprob = NonlinearProblem(NonlinearFunction(_g), prob.u0)
    kernel(nlprob, x0s, result, maxiters; ndrange = length(x0s))

    # @show result
    t1 = time()
    @show result
    sol_bfgs = [prob.f(θ, prob.p) for θ in getfield.(result, Ref(:u))]
    @show sol_bfgs

    minobj, ind = findmin(x -> isnan(x) ? Inf : x ,sol_bfgs)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p),
        result[ind].u, minobj)
end