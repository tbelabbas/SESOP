
using BenchmarkTools
using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools

include("../src/SESOP.jl")

using DataFrames, Printf, Random
using SolverBenchmark
using TimerOutputs

maxiter = 10000
itSE = 100
maxtime = 5.0
Lp = Inf

atol = 1e-8
rtol = 1e-10

n = 5000

m = [1, 4, 0]

verbose = 0

# nlp = MathOptNLPModel(PureJuMP.palmer1c(n=n), name="palmer1c")
# nlp = MathOptNLPModel(PureJuMP.tquartic(n=n), name="tquartic")
# nlp = MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank")
nlp = MathOptNLPModel(PureJuMP.dixmaang(n=n), name="dixmaang")
# nlp = MathOptNLPModel(PureJuMP.srosenbr(n=n), name="srosenbr")
# nlp = MathOptNLPModel(PureJuMP.woods(n=n), name="woods")
# nlp = MathOptNLPModel(PureJuMP.genrose(n=n), name="genrose")


my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st,
                                                                 pnorm = Lp;
                                                                 kwargs...)


function Compute_data(nlp, stp, pres)
    to = TimerOutput()
    df = DataFrame()
    df.Type = ["Nb_it", "T(ms)", "Nb_obj", "Nb_grad", "Nb_H", "Nb_hprod", "‖∇f‖", "f"]


    df.SESOP_nt = test_sesop(to, SESOP_newton,
                                stp, nlp, m, verbose, atol, rtol, pres, itSE)
    df.SESOP_lb = test_sesop(to, SESOP_lbfgs,
                                stp, nlp, m, verbose, atol, rtol, pres, itSE)
    df.SESOP_b  = test_sesop(to, SESOP_bfgs,
                                stp, nlp, m, verbose, atol, rtol, pres, itSE)
    df.SESOP_c  = test_sesop(to, SESOP_clbfgs,
                                stp, nlp, m, verbose, atol, rtol, pres, itSE)
    df.SESOP_ch = test_sesop(to, SESOP_chlbfgs,
                                stp, nlp, m, verbose, atol, rtol, pres, itSE)

    reset!(nlp)

    results = @timeit to "LBFGS" lbfgs(nlp, verbose=verbose,
                                        atol=atol, rtol=rtol)
    df.LBFGS = [results.iter, TimerOutputs.time(to["LBFGS"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective]

    reset!(nlp)

    results = @timeit to "Tron" tron(nlp, atol=atol, rtol=rtol, verbose=verbose)
    df.Tron = [results.iter, TimerOutputs.time(to["Tron"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective]

    reset!(nlp)

    results = @timeit to "Trunk" trunk(nlp, atol=atol, rtol=rtol)
    df.Trunk = [results.iter, TimerOutputs.time(to["Trunk"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), results.dual_feas, results.objective]


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
            neval_hprod(nlp_i), norm(grad(nlp_i, x)), f]
end


 # setup the Stopping object
 stp = NLPStopping(nlp,
 StoppingMeta(),
 StopRemoteControl(domain_check = false),
 NLPAtX(nlp.meta.x0)  )

stp.meta.optimality_check = my_unconstrained_check
stp.meta.max_iter = maxiter
stp.meta.max_time = maxtime
stp.meta.atol = atol
stp.meta.rtol = rtol

reset!(nlp)
reinit!(stp)

precision = 0.4
df = Compute_data(nlp, stp, precision)

SESOP_lbfgs(nlp, stp=stp, mem=m, atol=atol, rtol=rtol,
   max_iter_SE=itSE, precision=precision);

;