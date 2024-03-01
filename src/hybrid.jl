@kernel function simplebfgs_run!(nlprob, x0s, result, opt, maxiters, abstol, reltol)
    i = @index(Global, Linear)
    nlcache = remake(nlprob; u0 = x0s[i])
    sol = solve(nlcache, opt; maxiters, abstol, reltol)
    result[i] = sol.u
end

struct HybridPSOCache{TPc, TSp, TAlg}
    pso_cache::TPc
    start_points::TSp
    alg::TAlg
end

function SciMLBase.init(
        prob::OptimizationProblem, opt::HybridPSO{Backend, LocalOpt}, args...;
        kwargs...) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}
    psoalg = opt.pso
    backend = opt.backend

    pso_cache = init(prob, psoalg)

    start_points = KernelAbstractions.allocate(
        backend, typeof(prob.u0), opt.pso.num_particles)

    return HybridPSOCache{
        typeof(pso_cache), typeof(start_points), typeof(opt)}(pso_cache, start_points, opt)
end

function reinit_cache!(cache::HybridPSOCache,
        opt::HybridPSO{Backend, LocalOpt}) where {Backend, LocalOpt <: Union{LBFGS, BFGS}}
    reinit!(cache.pso_cache)
    fill!(cache.start_points, zero(eltype(cache.start_points)))
    # prob = cache.prob
    # backend = opt.backend
    # particles = cache.particles

    # kernel! = PSOGPU.gpu_init_particles!(backend)
    # kernel!(particles, prob, opt, typeof(prob.u0); ndrange = opt.num_particles)

    # best_particle = minimum(particles)
    # _init_gbest = SPSOGBest(best_particle.best_position, best_particle.best_cost)

    # copyto!(cache.gbest, [_init_gbest])

    return nothing
end

function Base.getproperty(cache::HybridPSOCache, name::Symbol)
    if name ∈ (:start_points, :pso_cache, :alg)
        return getfield(cache, name)
    else
        return getproperty(cache.pso_cache, name)
    end
end

function Base.setproperty!(cache::HybridPSOCache, name::Symbol, val)
    if name ∈ (:start_points, :pso_cache, :alg)
        return setfield!(cache, name, val)
    else
        return setproperty!(cache.pso_cache, name, val)
    end
end

function SciMLBase.solve!(
        cache::HybridPSOCache, opt::HybridPSO{Backend, LocalOpt}, args...;
        abstol = nothing,
        reltol = nothing,
        maxiters = 100, local_maxiters = 10, kwargs...) where {
        Backend, LocalOpt <: Union{LBFGS, BFGS}}

    pso_cache = cache.pso_cache

    sol_pso = solve!(pso_cache)
    x0s = sol_pso.original

    backend = opt.backend

    prob = remake(cache.prob, lb = nothing, ub = nothing)
    f = Base.Fix2(prob.f.f, prob.p)
    ∇f = instantiate_gradient(f, prob.f.adtype)

    kernel = simplebfgs_run!(backend)
    result = cache.start_points
    copyto!(result, x0s)
    nlprob = NonlinearProblem{false}(∇f, prob.u0)

    nlalg = LocalOpt isa LBFGS ?
            SimpleLimitedMemoryBroyden(;
        threshold = local_opt.threshold,
        linesearch = Val(true)) : SimpleBroyden(; linesearch = Val(true))

    t0 = time()
    kernel(nlprob,
        x0s,
        result,
        nlalg,
        local_maxiters,
        abstol,
        reltol;
        ndrange = length(x0s))

    sol_bfgs = (x -> prob.f(x, prob.p)).(result)
    sol_bfgs = (x -> isnan(x) ? convert(eltype(prob.u0), Inf) : x).(sol_bfgs)

    minobj, ind = findmin(sol_bfgs)
    sol_u, sol_obj = minobj > sol_pso.objective ? (sol_pso.u, sol_pso.objective) :
                     (view(result, ind), minobj)
    t1 = time()

    # @show sol_pso.stats.time

    solve_time = (t1 - t0) + sol_pso.stats.time

    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        sol_u, sol_obj,
        stats = Optimization.OptimizationStats(; time = solve_time))
end
