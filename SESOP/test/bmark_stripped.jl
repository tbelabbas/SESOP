using LinearAlgebra
using Logging

using CUTEst
#using AmplNLReader
using NLPModels
using SolverBenchmark
using SolverTools

#using Stopping

using JSOSolvers


include("../src/SESOP.jl")

shortlist = (CUTEstModel(probname) for probname in sort(CUTEst.select(min_var=200,
                                                                      max_var=200,
                                                                      max_con=0,
                                                                      only_free_var=true)))
problems = (shortlist)

function SE(prob)

    t = @timed iterations, xopt, optimal, ngx, fx, status = SESOP(prob,
                                                                    max_time = 1.0)

    res = GenericExecutionStats(status,
                                prob,
                                solution=xopt,
                                iter=iterations,
                                primal_feas=0.0,
                                dual_feas=ngx,
                                objective=fx,
                                elapsed_time=t[2],
                                )

    return res

end

solvers = Dict{Symbol,Function}(
    :SESOP => SE
)

stats = bmark_solvers(solvers, problems)


solved(df) = df.status .== :first_order
costnames = ["time",
             "objective evals",
             "gradient evals",
             "hessian-vector products",
             "obj + grad + hprod"]
costs = [df -> .!solved(df) .* Inf .+ df.elapsed_time,
         df -> .!solved(df) .* Inf .+ df.neval_obj,
         df -> .!solved(df) .* Inf .+ df.neval_grad,
         df -> .!solved(df) .* Inf .+ df.neval_hprod,
         df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hprod]

p = profile_solvers(stats, costs, costnames)

#Pour générer directement un pdf:

#Plots.pdf(p, "NomBidon.pdf")

