
using BenchmarkTools
using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools
using TimerOutputs

include("../src/SESOP.jl")
include("../src/SESOP_pastiche.jl")

using DataFrames, Printf, Random
using SolverBenchmark

include("testFramework.jl")

my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st,
                                                                 pnorm = Lp;
                                                                 kwargs...)


function Compute_data(nlp, stp, pres, m, itSE)
    to = TimerOutput()

    df = Compute_SESOP(to, nlp, stp, pres, m, itSE)

    # df.SESOP_nt = test_sesop(to, SESOP_newton,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)
    # df.SESOP_lb = test_sesop(to, SESOP_lbfgs,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)
    # df.SESOP_b  = test_sesop(to, SESOP_bfgs,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)
    # df.SESOP_c  = test_sesop(to, SESOP_clbfgs,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)
    # df.SESOP_ch = test_sesop(to, SESOP_chlbfgs,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)
    # df.SESOP_tr = test_sesop(to, SESOP_trunk,
    #                             stp, nlp, m, verbose, atol, rtol, pres, itSE)

    reset!(nlp)

    results = @timeit to "LBFGS" lbfgs(nlp, verbose=verbose, max_time=maxtime,
                                        max_eval = maxiter,
                                        atol=atol, rtol=rtol, mem=sum(m))
    df.LBFGS = [results.iter, TimerOutputs.time(to["LBFGS"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective, results.status]

    reset!(nlp)

    results = @timeit to "Tron" tron(nlp, max_time=maxtime,
                                     max_eval = maxiter, atol=atol,
                                     rtol=rtol, verbose=verbose
                                        )
    df.Tron = [results.iter, TimerOutputs.time(to["Tron"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective, results.status]

    reset!(nlp)

    results = @timeit to "Trunk" trunk(nlp, max_time=maxtime,
                                        max_eval = maxiter, atol=atol, rtol=rtol)
    df.Trunk = [results.iter, TimerOutputs.time(to["Trunk"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective, results.status]


    println(df)

    return df
end

function test_sesop(to, fn :: Function, stp_i, nlp_i, m_i, v_i, atol_i, rtol_i,
                    pres_i, itSE_i)
    reset!(nlp_i)
    reinit!(stp_i)
    stp, iter, x, f = fn(
                            nlp_i, stp=stp_i, mem=m_i, verbose=v_i,
                            atol=atol_i, rtol=rtol_i,
                            max_iter_SE=itSE_i, precision=pres_i
                        )

    reset!(nlp_i)
    reinit!(stp_i)
    tm = sum(m_i)
    name = String(Symbol(fn))
    stp, iter, x, f = @timeit to "SESOP$(tm)_$(name)" fn(
                                                            nlp_i,
                                                            stp=stp_i,
                                                            mem=m_i,
                                                            verbose=v_i,
                                                            atol=atol_i,
                                                            rtol=rtol_i,
                                                            max_iter_SE=itSE_i,
                                                            precision=pres_i
                                                        )

    return [stp.meta.nb_of_stop,
            TimerOutputs.time(to["SESOP$(tm)_$(name)"])*10e-6,
            neval_obj(nlp_i), neval_grad(nlp_i), neval_hess(nlp_i),
            neval_hprod(nlp_i), norm(grad(nlp_i, x)), f, getStatus(stp)]
end


maxiter = 6000
maxtime = 10.0

atol = 1e-6
rtol = 1e-8

Lp = Inf

verbose = 0

dimensions = [5000]

for n in dimensions
    nlp_list = [    MathOptNLPModel(PureJuMP.palmer1c(n=n), name="palmer1c"),
                    MathOptNLPModel(PureJuMP.tquartic(n=n), name="tquartic"),
                    MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank"),
                    MathOptNLPModel(PureJuMP.dixmaang(n=n), name="dixmaang"),
                    MathOptNLPModel(PureJuMP.srosenbr(n=n), name="srosenbr"),
                    MathOptNLPModel(PureJuMP.woods(n=n), name="woods"),
                    MathOptNLPModel(PureJuMP.genrose(n=n), name="genrose")
                ]
    for nlp in nlp_list
        m = 5 # max(1, n ÷ 5)
        config_list = [ #[n, 0, 0],
                        #[0, n, 0],
                        #[n - (n ÷ 2), n ÷ 2, 0],
                        [m - (m ÷ 2), m ÷ 2, 0],
                        [1, m - 1, 0],
                        [m - 1, 1, 0],
                        #[0, m, 0],
                        #[m, 0, 0],
                      ]
        for mem in config_list
            precisions = [0.001, 0.4, 1]
            for pres in precisions
                max_itSEs = [10]
                for max_itSE in max_itSEs
                     # setup the Stopping object
                    stp = NLPStopping(nlp,
                    StoppingMeta(),
                    StopRemoteControl(domain_check = false),
                    NLPAtX(nlp.meta.x0))

                    stp.meta.optimality_check = my_unconstrained_check
                    stp.meta.max_iter = maxiter
                    stp.meta.max_time = maxtime
                    stp.meta.atol = atol
                    stp.meta.rtol = rtol

                    reset!(nlp)
                    reinit!(stp)
                    df = Compute_data(nlp, stp, pres, mem, max_itSE)
                    println("_____________________________________________\n")
                    println(df)
                    println("_____________________________________________\n")

                    open("results.txt","a") do io
                        println(io, "_____________________________________________\n")
                        println(io,"NLP = $(nlp.meta.name) | N = $(nlp.meta.nvar) | MEM = $(sum(mem)) - config : [$(mem[1]), $(mem[2])] | PRES = $pres | MAX_IT_SE = $(max_itSE)")
                        println(io, df)
                        println(io, "\n_____________________________________________")
                    end

                end
            end
        end
    end
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

;