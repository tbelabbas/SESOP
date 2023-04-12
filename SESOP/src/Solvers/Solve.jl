function solve(nlp, maxiter, maxtime, atol, rtol,
               fn::Function, subspace_verbose::Int)

    # setup the Stopping object
    stp = NLPStopping(nlp,
                    StoppingMeta(),
                    StopRemoteControl(domain_check = false),
                    NLPAtX(nlp.meta.x0)  )


    my_unconstrained_check(nlp, st;
                            kwargs...) = unconstrained_check(nlp, st,
                                                                pnorm = Inf;
                                                                kwargs...
                                                            )

    stp.meta.optimality_check = my_unconstrained_check

    stp.meta.max_iter = maxiter
    stp.meta.max_time = maxtime
    stp.meta.atol = atol
    stp.meta.rtol = rtol

    reset!(nlp)
    reinit!(stp)

    global t = nothing

    #let t=t
    with_logger(NullLogger()) do
        t = @timed stp = fn(nlp, stp = stp, verbose = subspace_verbose)
    end
    #end

    iter = stp.meta.nb_of_stop

    xsol = stp.current_state.x
    fx = stp.current_state.fx
    gx = stp.current_state.gx

    status = getStatus(stp)

    return GenericExecutionStats(status, nlp,
                                    solution = xsol,
                                    iter = iter,
                                    dual_feas = stp.current_state.current_score,
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