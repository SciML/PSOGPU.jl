struct LBFGS{Backend}
    ϵ::Float64
    m::Int
    backend::Backend
end

function LBFGS(; ϵ = 1e-8, m = 10, backend = CPU())
    LBFGS(ϵ, m, backend)
end

@kernel function solve_lbfgs(nlprob::NonlinearProblem, opt, result, maxiters = 1000)
    result .= SciMLBase.__solve(nlprob, opt; maxiters = maxiters).u
end

function SciMLBase.__solve(prob::SciMLBase.OptimizationProblem, opt::LBFGS, args...; maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f.f, prob.p)

    function _g(θ, _p = nothing) 
        return ForwardDiff.gradient(f , θ) 
    end

    kernel = solve_lbfgs(opt.backend) 
    # @show cache.u0
    t0 = time()
    result = KernelAbstractions.allocate(opt.backend, eltype(prob.u0), size(prob.u0))

    nlprob = NonlinearProblem{false}(_g, prob.u0)
    nlsol = kernel(nlprob, SimpleBroyden(; linesearch = Val(true)), result, maxiters; ndrange = (1,))
    t1 = time()
    θ = result
    # @show nlsol.stats
    # @show nlsol.resid

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt, θ, prob.f(θ, prob.p))
end