
function Newton_Spectral(nlp:: AbstractNLPModel;
                         ϵ :: Float64 = 1e-6,
                         maxiter :: Int = 200,
                         iter_LS :: Int = 50,
                         verbose :: Bool = false,
                         ∇f :: AbstractVector)
    α = nlp.meta.x0
    n = length(α)

    f = obj(nlp, α)

    ls = []
    iter = 0

    T = Float64
    if verbose > 0
        @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
        hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))

        @info log_row(Any[iter, f, norm(∇f)])
    end


    while (norm(∇f) > ϵ) && (iter < maxiter)
        i = 0
        H = hess(nlp, α)

        O = eigvecs(H)
        Δ = eigvals(H)

        D = Δ + max.(real.(1e-8 .- Δ), 0.0) .*ones(n)
        d = real.(- O*diagm(1.0 ./ D)*O'*∇f)

        τ₀ = 0.0005
        hp0 = ∇f'*d
        t=1.0

        ft = obj(nlp, α + t*d)
        #tries = 0
        while ft > (f + τ₀*t*hp0) #&& tries < 5
            t /= 2.0
            i += 1
            ft = obj(nlp, α + t*d)
            # if ft >= f
            #     tries += 1
            # end
        end

        append!(ls, i)

        α += t*d
        iter += 1

        ∇f = grad!(nlp, α, ∇f)
        f = ft

        if verbose > 0
            @info log_row(Any[iter, f, norm(∇f), t, hp0])
        end
    end

    if iter == maxiter @warn "Maximum d'itérations"
    end

    return iter, f, norm(∇f), α, ls
end

my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp,
                                                                 st,
                                                                 pnorm = Lp;
                                                                 kwargs...)

function Newton_Spectral_stp(nlp:: AbstractNLPModel,
                             stp_s :: NLPStopping,
                             iter_LS :: Int,
                             verbose :: Int)
    α = nlp.meta.x0
    n = length(α)

    f = obj(nlp, α)

    ls = []
    iter = 0

    ∇f = grad(nlp, nlp.meta.x0)

    T = Float64
    if verbose > 0
        @info log_header([:iter, :f, :norm, :step, :iterLS], [Int, T, T, T, Int],
                          hdr_override=Dict(:f=>"f(x)", :norm=>"‖∇f‖"))

        @info log_row(Any[iter, f, norm(∇f), 0.0, 0])
    end

    OK = update_and_start!(stp_s, x = α, fx = f, gx = ∇f)

    while !OK
        i = 0
        H = hess(nlp, α)

        O = eigvecs(H)
        Δ = eigvals(H)

        D = Δ + max.(real.(1e-8 .- Δ), 0.0) .*ones(n)
        d = real.(- O*diagm(1.0 ./ D)*O'*∇f)

        τ₀ = 0.0005
        hp0 = ∇f'*d
        t=1.0

        ft = obj(nlp, α + t*d)

        tries = 0
        while ft > (f + τ₀*t*hp0) && tries < iter_LS
            t /= 2.0
            i += 1
            ft = obj(nlp, α + t*d)
            tries += 1
        end

        append!(ls, i)

        α += t*d

        ∇f = grad!(nlp, α, ∇f)
        f = ft

        iter += 1

        if verbose > 0
            @info log_row(Any[iter, f, norm(∇f), t, tries])
        end

        OK = update_and_stop!(stp_s, x = α, fx = f, gx = ∇f)
    end

    return stp_s
end


function Newton_Solver(nlp:: AbstractNLPModel,
                        maxiter :: Int;
                        iter_LS :: Int = 40,
                        atol = 1e-8,
                        rtol = 1e-12,
                        maxtime = 2.0,
                        precision = 1.0,
                        verbose :: Int = 0
                      )

    stp_se = NLPStopping(nlp,
                        StoppingMeta(),
                        StopRemoteControl(domain_check = false),
                        NLPAtX(nlp.meta.x0)
                    )

    stp_se.meta.optimality_check = my_unconstrained_check
    stp_se.meta.max_iter = maxiter
    stp_se.meta.max_time = maxtime
    stp_se.meta.atol = atol*precision
    stp_se.meta.rtol = rtol*precision

    t = @timed stp_se = Newton_Spectral_stp(nlp, stp_se, iter_LS, verbose)

    iter = stp_se.meta.nb_of_stop

    xsol = stp_se.current_state.x
    fx = stp_se.current_state.fx
    gx = stp_se.current_state.gx

    status = getStatus(stp_se)

    return GenericExecutionStats(status, nlp,
                                 solution = xsol,
                                 iter = iter,
                                 dual_feas = stp_se.current_state.current_score,
                                 objective = fx,
                                 elapsed_time = t[2],
                                )
end

function getStatus(stp :: NLPStopping)
    status = :unknown
    # stopping properties status of the problem in Stopping
    m = stp.meta

    m.domainerror     && (status = :domainerror)
    m.unbounded       && (status = :unbounded)
    m.unbounded_pb    && (status = :unbounded_success)
    m.fail_sub_pb     && (status = :unknown)
    m.tired           && (status = :max_time)
    m.stalled         && (status = :stalled)
    m.iteration_limit && (status = :max_iter)
    m.resources       && (status = :max_time)
    m.optimal         && (status = :first_order)
    m.infeasible      && (status = :infeasible)
    m.main_pb         && (status = :unknown)
    m.suboptimal      && (status = :unknown)
    m.stopbyuser      && (status = :user)
    m.exception       && (status = :exception)

    return status
end