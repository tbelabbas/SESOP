
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

maxiter = 500
maxiter_SE = 5
maxtime = 1.0
Lp = Inf

atol = 1e-6
rtol = 1e-12

n = 50

m = [1, 2, 0]

verbose = 1

# nlp = MathOptNLPModel(PureJuMP.palmer1c(n=n), name="palmer1c")
# nlp = MathOptNLPModel(PureJuMP.tquartic(n=n), name="tquartic")
# nlp = MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank")
# nlp = MathOptNLPModel(PureJuMP.dixmaang(n=n), name="dixmaang")
# nlp = MathOptNLPModel(PureJuMP.srosenbr(n=n), name="srosenbr")
# nlp = MathOptNLPModel(PureJuMP.woods(n=n), name="woods")
nlp = MathOptNLPModel(PureJuMP.genrose(n=n), name="genrose")


function Compute_data(nlp, stp, pres)
    to = TimerOutput()
    df = DataFrame()
    df.Type = ["Nb_it", "T(ms)", "Nb_o", "Nb_g", "Nb_H", "Nb_hprd", "‖∇f‖", "f"]


    df.SESOP_nt = test_sesop(to, SESOP_bfgs,
                             stp, nlp, m, verbose,
                             atol, rtol, pres, maxiter_SE)

    return df
end

function test_sesop(to, fn :: Function, stp_i, nlp_i, m_i, v_i, atol_i, rtol_i,
                    pres_i, itSE_i)
    reset!(nlp_i)
    reinit!(stp_i)
    stp, iter, x, f = fn(
                            nlp_i, stp=stp_i, mem=m_i,
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
                                                            atol=atol_i,
                                                            rtol=rtol_i,
                                                            max_iter_SE=itSE_i,
                                                            precision=pres_i,
                                                            verbose=v_i,
                                                            subspace_verbose=v_i
                                                        )

    return  [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["SESOP$(tm)_$(name)"])*10e-6,
                neval_obj(nlp_i), neval_grad(nlp_i), neval_hess(nlp_i),
                neval_hprod(nlp_i), norm(grad(nlp_i, x)), f
            ]
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

println(df)

;