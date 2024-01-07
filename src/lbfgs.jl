struct LBFGS
    ϵ::Float64
    m::Int
end

function LBFGS(; ϵ = 1e-8, m = 10)
    LBFGS(ϵ, m)
end

SciMLBase.supports_opt_cache_interface(opt::LBFGS) = true

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt::LBFGS,
    data = Optimization.DEFAULT_DATA; save_best = true,
    callback = (args...) -> (false),
    progress = false, kwargs...)
    return Optimization.OptimizationCache(prob, opt, data; save_best, callback, progress,
        kwargs...)
end

function SciMLBase.__solve(cache::Optimization.OptimizationCache{
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O,
    D,
    P,
    C,
}) where {
    F,
    RC,
    LB,
    UB,
    LC,
    UC,
    S,
    O <:LBFGS,
    D,
    P,
    C,
}

    _g = (G, θ, _p=nothing) -> cache.f.grad(G, θ)
    @show cache.u0
    t0 = time()
    nlprob = NonlinearProblem(NonlinearFunction(_g), cache.u0)
    nlsol = solve(nlprob, LimitedMemoryBroyden(; threshold = cache.opt.m, linesearch = LiFukushimaLineSearch()))
    t1 = time()
    θ = nlsol.u
    @show nlsol.stats
    @show nlsol.resid

    SciMLBase.build_solution(cache, cache.opt, θ, cache.f(θ, cache.p), solve_time = t1 - t0)
end