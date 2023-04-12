using TimerOutputs

using BenchmarkTools
using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools

include("../src/SESOP_pastiche.jl")

include("../../../../QuasiNewton/test/Framework.jl")

include("testFramework.jl")


using DataFrames, Printf, Random
using SolverBenchmark

maxiter = 1000
maxiter_SE = 5
maxtime = 10.0
Lp = Inf

atol = 1e-5
rtol = 1e-12

n = 400

m = [1, 2, 0]

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

precision = 0.4
df = Compute_SESOP(to, nlp, stp, precision, m, maxiter_SE)

reset!(nlp)
reinit!(stp)


n = nlp.meta.nvar
scaling = false
@show n

# @info "Version encapsulée opérateur, formule O(n^2)"
B = InverseBFGSOperator(Float64, n, scaling=scaling)
stp = bfgs_StopLS(nlp, stp=stp, B₀=B)
reset!(nlp)
reinit!(stp)
B = InverseBFGSOperator(Float64, n, scaling=scaling)
stp = @timeit to "BFGS" bfgs_StopLS(nlp, stp=stp, B₀=B)
# println("| Nb iter : $(stp.meta.nb_of_stop) | Grad norm : $(norm(stp.current_state.gx)) | Fct value : $(stp.current_state.fx) |")

df.BFGS = [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["BFGS"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), norm(stp.current_state.gx),
                stp.current_state.fx,
                getStatus(stp)
             ]

reset!(nlp)
reinit!(stp)
# @info "Version encapsulée opérateur, formule O(n^2)"
B = CompactInverseBFGSOperator(Float64, n, mem=sum(m), scaling=scaling)
stp = bfgs_StopLS(nlp, stp=stp, B₀=B)
reset!(nlp)
reinit!(stp)
B = CompactInverseBFGSOperator(Float64, n, mem=sum(m), scaling=scaling)
stp = @timeit to "Compact" bfgs_StopLS(nlp, stp=stp, B₀=B)
# println("| Nb iter : $(stp.meta.nb_of_stop) | Grad norm : $(norm(stp.current_state.gx)) | Fct value : $(stp.current_state.fx) |")

df.Compact = [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["Compact"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), norm(stp.current_state.gx),
                stp.current_state.fx,
                getStatus(stp)
             ]
reset!(nlp)
reinit!(stp)

# @info "Version encapsulée opérateur, formule O(n^2)"
B = InverseLBFGSOperator(Float64, n, mem=sum(m), scaling=scaling)
stp = bfgs_StopLS(nlp, stp=stp, B₀=B)
reset!(nlp)
reinit!(stp)
B = InverseLBFGSOperator(Float64, n, mem=sum(m), scaling=scaling)
stp = @timeit to "LBFGS" bfgs_StopLS(nlp, stp=stp, B₀=B)
# println("| Nb iter : $(stp.meta.nb_of_stop) | Grad norm : $(norm(stp.current_state.gx)) | Fct value : $(stp.current_state.fx) |")

df.LBFGS = [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["LBFGS"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), norm(stp.current_state.gx),
                stp.current_state.fx,
                getStatus(stp)
             ]

# @info "Version encapsulée opérateur, formule O(n^2)"
B = ChBFGSOperator(Float64, n, scaling=scaling)
stp = bfgs_StopLS(nlp, stp=stp, B₀=B)
reset!(nlp)
reinit!(stp)
B = ChBFGSOperator(Float64, n, scaling=scaling)
stp = @timeit to "Cholesky" bfgs_StopLS(nlp, stp=stp, B₀=B, verbose=0)
# println("| Nb iter : $(stp.meta.nb_of_stop) | Grad norm : $(norm(stp.current_state.gx)) | Fct value : $(stp.current_state.fx) |")

df.Cholesky = [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["Cholesky"])*10e-6,
                neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                neval_hprod(nlp), norm(stp.current_state.gx),
                stp.current_state.fx,
                getStatus(stp)
             ]


println(df)

# using ProfileView
# reset!(nlp)
# reinit!(stp)
# SESOP_pastiche(
#     nlp,
#     stp=stp,
#     mem=m,
#     atol=atol,
#     rtol=rtol,
#     max_iter_SE=maxiter_SE,
#     precision=precision,
#     verbose=verbose,
#     subspace_verbose=verbose,
#     matrix=3
# )
# reset!(nlp)
# reinit!(stp)
# @profview SESOP_pastiche(
#     nlp,
#     stp=stp,
#     mem=m,
#     atol=atol,
#     rtol=rtol,
#     max_iter_SE=maxiter_SE,
#     precision=precision,
#     verbose=verbose,
#     subspace_verbose=verbose,
#     matrix=3
# )
;