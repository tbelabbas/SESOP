using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools

include("../src/SESOP.jl")

using ProfileView
using DataFrames

n = 500
mSE = 300
precision = 0.6

# MathOptNLPModel
#mdl = MathProgNLPModel(rosenbrock(n))
#nlp = MathProgNLPModel(woods(n))
#mdl2 = MathProgNLPModel(penalty2(n))
nlp = MathProgNLPModel(dixmaane(n))
#mdl5 = MathProgNLPModel(dixmaani(n))


df = DataFrame()
df.Type = ["Nb_iter", "Nb_obj", "Nb_grad",  "Nb_hess", "Nb_hprod", "‖∇f‖"]


# println("LBFGS - premiere excecution")
# with_logger(NullLogger()) do
    # results = lbfgs(nlp, verbose=true, atol=1e-9, rtol=1e-9)
# end
# println("----------------")
# reset!(nlp)
# println("LBFGS - deuxieme excecution")
# #with_logger(NullLogger()) do
    # results = @profview lbfgs(nlp, verbose=true, atol=1e-9, rtol=1e-9)
    # df.LBFGS = [results.iter, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), results.dual_feas]
# #end
# println("----------------")
# reset!(nlp)

j
# println("Tron - premiere excecution")
# with_logger(NullLogger()) do
#     results = tron(nlp, atol=1e-9, rtol=1e-9)
# end
# println("----------------")
# reset!(nlp)

# println("Tron - deuxieme excecution")
# with_logger(NullLogger()) do
#     results = @profview tron(nlp, atol=1e-9, rtol=1e-9)
#     df.Tron = [results.iter, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), results.dual_feas]
# end
# println("----------------")
# reset!(nlp)

println("SESOP-06 - premiere excecution")
iter, x = @profview SESOP(nlp, mem=6, verbose=false, ϵ=1e-7, precision=0.1)
println("----------------")
reset!(nlp)
println("SESOP-06 - deuxieme excecution")
iter, x = @profview SESOP(nlp, mem=6, verbose=false, ϵ=1e-7, precision=0.1)
#df.SESOP6 = [iter, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), norm(grad(nlp, x))]
println("----------------")

# println("SESOP-06 BFGS - premiere excecution")
# iter, x = SESOPbfgs(nlp, mem=6, verbose=false, ϵ=1e-7, precision=0.1)
# println("----------------")
# reset!(nlp)
# println("SESOP-06 BFGS - deuxieme excecution")
# iter, x = @profview SESOPbfgs(nlp, mem=6, verbose=false, ϵ=1e-7, precision=0.1)
# df.SESOP6bfgs = [iter, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), norm(grad(nlp, x))]
# println(df)
# println("----------------")

#df

;


