@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    sol = solve(nlcache, opt; maxiters, abstol, reltol)
    result[i] = sol.u
end

function SciMLBase.__solve(prob::SciMLBase.OptimizationProblem,
        opt::HybridPSO{Backend, LocalOpt},
        args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100,
        local_maxiters = 10,
        kwargs...) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}
    t0 = time()
    psoalg = opt.pso
    local_opt = opt.local_opt
    backend = opt.backend

    sol_pso = solve(prob, psoalg, args...; maxiters, kwargs...)
    x0s = sol_pso.original
    prob = remake(prob, lb = nothing, ub = nothing)
    f = Base.Fix2(prob.f.f, prob.p)
    ∇f = instantiate_gradient(f, prob.f.adtype)

    kernel = simplebfgs_run!(backend)
    result = KernelAbstractions.allocate(backend, typeof(prob.u0), length(x0s))
    nlprob = NonlinearProblem{false}(∇f, prob.u0)

    nlalg = LocalOpt isa LBFGS ?
            SimpleLimitedMemoryBroyden(;
        threshold = local_opt.threshold,
        linesearch = Val(true)) : SimpleBroyden(; linesearch = Val(true))

    kernel(nlprob,
        x0s,
        result,
        nlalg,
        local_maxiters,
        abstol,
        reltol;
        ndrange = length(x0s))

    t1 = time()
    sol_bfgs = (x -> prob.f(x, prob.p)).(result)
    sol_bfgs = (x -> isnan(x) ? convert(eltype(prob.u0), Inf) : x).(sol_bfgs)

    minobj, ind = findmin(sol_bfgs)

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        view(result, ind), minobj)
end
