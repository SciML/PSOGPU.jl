struct LBFGS{Backend}
    ϵ::Float64
    m::Int
    backend::Backend
end

function LBFGS(; ϵ = 1e-8, m = 10, backend = CPU())
    LBFGS(ϵ, m, backend)
end

function SciMLBase.__solve(prob::SciMLBase.OptimizationProblem, opt::LBFGS, args...; maxiters = 1000, kwargs...)
    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, 0)
    if prob.u0 isa SVector
        G = KernelAbstractions.allocate(opt.backend, eltype(prob.u0), size(prob.u0))
        _g = (θ, _p = nothing) -> f.grad(G, θ) 
    else
        _g = (G, θ, _p=nothing) -> f.grad(G, θ)
    end
    # @show cache.u0
    t0 = time()
    nlprob = NonlinearProblem(NonlinearFunction(_g), prob.u0)
    nlsol = solve(nlprob, SimpleLimitedMemoryBroyden(; threshold = opt.m, linesearch = Val(true)), maxiters = maxiters)
    t1 = time()
    θ = nlsol.u
    # @show nlsol.stats
    # @show nlsol.resid

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, θ, prob.f(θ, prob.p))
end