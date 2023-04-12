using NLPModels
using NLPModelsJuMP
using OptimizationProblems

using Test

include("../../src/Models/SEModel.jl")


nlp = MathOptNLPModel(PureJuMP.woods(n=10), name="woods")

x = nlp.meta.x0
gradient = grad(nlp, x)

D = rand(length(x), 5)
SE = SEModel(nlp, x, D, gradient)

α = rand(length(SE.meta.x0))
v = rand(length(SE.meta.x0))

# By hand hprod
mhv = hess(SE, α) * v

# Implemented hprod
hv = hprod(SE, α, v)

@test mhv ≈ hv ;