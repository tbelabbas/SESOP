using Pkg
Pkg.activate(".")

using TimerOutputs

using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels


include("../src/SESOP_pastiche.jl")
include("testFramework.jl")


using DataFrames


maxiter = 1000
maxiter_SE = 5
maxtime = 10.0
Lp = Inf

atol = 1e-5
rtol = 1e-8

n = 40

m = [2, 2, 0]

verbose = 0

# nlp = MathOptNLPModel(PureJuMP.palmer1c(n=n), name="palmer1c")
# nlp = MathOptNLPModel(PureJuMP.tquartic(n=n), name="tquartic")
# nlp = MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank")
# nlp = MathOptNLPModel(PureJuMP.dixmaang(n=n), name="dixmaang")
# nlp = MathOptNLPModel(PureJuMP.srosenbr(n=n), name="srosenbr")
# nlp = MathOptNLPModel(PureJuMP.woods(n=n), name="woods")
nlp = MathOptNLPModel(PureJuMP.genrose(n=n), name="genrose")


to = TimerOutput()

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


precision = 0.001

b0=3

using ProfileView
using TimerOutputs

to = TimerOutput()

df = DataFrame()
df.Type = ["Nb_it", "ms", "Nb_o", "Nb_g", "Nb_H", "Nb_hprd", "‖∇f‖", "f", "status"]


reset!(nlp)
reinit!(stp)
stp, iter, x, f = SESOP_pastiche(
                        nlp, stp=stp, mem=m,
                        atol=atol, rtol=rtol,
                        max_iter_SE=maxiter_SE, precision=precision,
                        matrix=b0
                    )

reset!(nlp)
reinit!(stp)
@profview SESOP_pastiche(
                        nlp, stp=stp, mem=m,
                        atol=atol, rtol=rtol,
                        max_iter_SE=maxiter_SE, precision=precision,
                        matrix=b0
                    )

reset!(nlp)
reinit!(stp)
stp, iter, x, f = @timeit to "SESOP"  SESOP_pastiche(
                        nlp, stp=stp, mem=m,
                        atol=atol, rtol=rtol,
                        max_iter_SE=maxiter_SE, precision=precision,
                        matrix=b0
                    )

df.SESOP =  [
                stp.meta.nb_of_stop, TimerOutputs.time(to["SESOP"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), norm(grad(nlp, x)), f,
                getStatus(stp)
            ]

# rename!(df, :SESOP => "SESOP_$(b0)")

reset!(nlp)

results = @timeit to "LBFGS" lbfgs(nlp, verbose=verbose, max_time=maxtime,
                                    max_eval = maxiter,
                                    atol=atol, rtol=rtol, mem=sum(m))
df.LBFGS = [results.iter, TimerOutputs.time(to["LBFGS"])*10e-6,
            neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
            neval_hprod(nlp), results.dual_feas, results.objective,
            results.status]


print(df)

;