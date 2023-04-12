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


maxiter = 3000
maxiter_SE = 10
maxtime = 10.0
Lp = Inf

atol = 1e-4
rtol = 1e-6

n = 50

m = [1, 49, 0]

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
df = Compute_SESOP(to, nlp, stp, precision, m, maxiter_SE)


reset!(nlp)

results = @timeit to "LBFGS" lbfgs(nlp, verbose=verbose, max_time=maxtime,
                                    max_eval = maxiter,
                                    atol=atol, rtol=rtol, mem=sum(m))
df.LBFGS = [results.iter, TimerOutputs.time(to["LBFGS"])*10e-6,
            neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
            neval_hprod(nlp), results.dual_feas, results.objective, results.status]

display(df)

;

