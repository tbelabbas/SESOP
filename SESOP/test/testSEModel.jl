using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools

include("../src/Utils/SubspaceData.jl")

using DataFrames, Printf, Random
using SolverBenchmark
using TimerOutputs


n = 500

nlp = MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank")

to = TimerOutput()
df = DataFrame()
df.Type = ["Time_obj", "Time_grad",  "Time_grad_p", "Grad"]

x0 = nlp.meta.x0
gradient = grad(nlp, x0)

D = 1.0* Matrix(I, length(x0), length(x0))
SEMdl = SEModel(nlp, x, D, gradient)

α0 = SEMdl.meta.x0
@show size(α0), size(D*α0), size(x0 + D*α0)
obj(SEMdl, α0)
seobj = @timeit to "SEModel_obj" obj(SEMdl, α0)
# seobj = @profview obj(SEMdl, α0)
grad(SEMdl, α0)
segrad = @timeit to "SEModel_grad" grad(SEMdl, α0)
grad!(SEMdl, α0, segrad)
@timeit to "SEModel_grad_p" grad!(SEMdl, α0, segrad)
# @profview grad!(SEMdl, α0, segrad)

obj(nlp, x0)
f = @timeit to "nlp_obj" obj(nlp, x0)
grad(nlp, x0)
g = @timeit to "nlp_grad" grad(nlp, x0)
grad!(nlp, x0, g)
@timeit to "nlp_grad_p" grad!(nlp, x0, g)

df.SE = [TimerOutputs.time(to["SEModel_obj"])*10e-6, TimerOutputs.time(to["SEModel_grad"])*10e-6,
          TimerOutputs.time(to["SEModel_grad_p"])*10e-6, norm(segrad)]
df.Nlp = [TimerOutputs.time(to["nlp_obj"])*10e-6, TimerOutputs.time(to["nlp_grad"])*10e-6,
          TimerOutputs.time(to["nlp_grad_p"])*10e-6, norm(g)]

print(df)