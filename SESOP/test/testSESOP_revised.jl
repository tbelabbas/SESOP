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


n = 10000
memo = 4
itSE = 10
precision = 0.6
mit=200

dict = OrderedDict{Int, AbstractArray}()

# mdl = MathOptNLPModel(PureJuMP.palmer1c(n=n), name="palmer1c")
# mdl = MathOptNLPModel(PureJuMP.tquartic(n=n), name="tquartic")
# mdl = MathOptNLPModel(PureJuMP.dixmaank(n=n), name="dixmaank")
# mdl = MathOptNLPModel(PureJuMP.dixmaang(n=n), name="dixmaang")
# mdl = MathOptNLPModel(PureJuMP.srosenbr(n=n), name="srosenbr")
#mdl = MathOptNLPModel(PureJuMP.woods(n=n), name="woods")
mdl = MathOptNLPModel(PureJuMP.genrose(n=n), name="genrose")

function Compute_data(nlp, pres, newton)
    to = TimerOutput()
    df = DataFrame()
    df.Type = ["Nb_iter", "Time(ms)", "Nb_obj", "Nb_grad",  "Nb_hess", "Nb_hprod", "‖∇f‖"]

    reset!(nlp)
    stp, iter, x = SESOP(nlp, mem=[1, 3, 0], verbose=0, atol=1e-9, rtol=0.0, max_iter_SE=itSE, precision=pres)
    reset!(nlp)
    stp, iter, x = @timeit to "SESOP6" SESOP(nlp, mem=[2, 3, 0], verbose=0, atol=1e-9, rtol=0.0, max_iter_SE=itSE, precision=pres)

    # iter, x, it_SE = SESOPnewton(nlp, mem=memo, verbose=false, ϵ=1e-7, max_iter_SE=itSE, precision=pres, maxiter=mit)
    # reset!(nlp)
    # iter, x, it_SE = @timeit to "SESOP6" SESOPnewton(nlp, mem=memo, verbose=false, ϵ=1e-7, max_iter_SE=itSE, precision=pres, maxiter=mit)
    # println(it_SE)

    df.SESOP6 = [iter, TimerOutputs.time(to["SESOP6"])*10e-6, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), norm(grad(nlp, x))]
    #println("----------------")

    reset!(nlp)

    #println("JSOSolvers - lbfgs")
    with_logger(NullLogger()) do
        results = @timeit to "LBFGS" lbfgs(nlp, verbose=1, atol=1e-9, rtol=0.0, max_eval=mit)
        df.LBFGS = [results.iter, TimerOutputs.time(to["LBFGS"])*10e-6, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), results.dual_feas]
    end
    #println("----------------")

    reset!(nlp)

    #println("JSOSolvers - tron")
    with_logger(NullLogger()) do
        results = @timeit to "Tron" tron(nlp, atol=1e-9, rtol=0.0)
        df.Tron = [results.iter, TimerOutputs.time(to["Tron"])*10e-6, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), results.dual_feas]
    end
    #println("----------------")

    reset!(nlp)

    #println("JSOSolvers - trunk")
    with_logger(NullLogger()) do
        results = @timeit to "Trunk" trunk(nlp, atol=1e-9, rtol=0.0)
        df.Trunk = [results.iter, TimerOutputs.time(to["Trunk"])*10e-6, neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp), results.dual_feas]
    end
    #println("----------------")

    println(df)

    reset!(nlp)
    return df
end

function computeTests(mdl, precision, newton)
    #println("Woods")
    #df = Compute_data(mdl1, precision)
    #println("Penalty2")
    #df = Compute_data(mdl2, precision)
    #println("Penalty3")
    #df = Compute_data(mdl3, precision)
    #println("Dixmaane")
    df = Compute_data(mdl, precision, newton)
    #println("Dixmaani")
    #df = Compute_data(mdl5, precision)
    #println("Genrose")
    #df = Compute_data(mdl6, precision)
    #println("Dus2_1")
    #df = Compute_data(mdl7, precision)
end

# println("=============================")
# println("Precision = 0.1")
# precision = 0.1
# computeTests(mdl, precision)
# println("=============================")

# println("====================================")
# println("Precision = 0.4 ---- LBFGS subsolver")
# precision = 0.4
# computeTests(mdl, precision, false)
# println("====================================")

println("=====================================")
println("Precision = 0.4 ---- Newton subsolver")
precision = 0.4
computeTests(mdl, precision, true)
# using Profile
# @profile computeTests(mdl, precision, true)

#println("=====================================")

# println("=============================")
# println("Precision = 0.6")
# precision = 0.6
# computeTests(mdl, precision)
# println("=============================")

# println("=============================")
# println("Precision = 0.99")
# precision = 0.99
# computeTests(mdl, precision)
# println("=============================")

reset!(mdl)
iter, x, it_SE = SESOPnewton(mdl, mem=memo, verbose=false, ϵ=1e-7, max_iter_SE=itSE, precision=precision, maxiter=mit);
# using ProfileView
# @profview SESOPnewton(mdl, mem=memo, verbose=false, ϵ=1e-7, max_iter_SE=itSE, precision=precision, maxiter=mit);
# #println(it_SE)
println("\n[iter : $iter| f : $(obj(mdl,x))| norm_g : $(norm(grad(mdl, x)))")


;
