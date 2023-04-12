using LinearAlgebra
using Logging

using CUTEst
#using AmplNLReader
using NLPModels
using SolverBenchmark
using SolverTools

#using Stopping

using JSOSolvers
using Plots


include("../src/SESOP.jl")

# Sélection de problèmes spécifiques
#probnames = ["ROSENBR", "WOODS", "PENALTY1"]
#problems = (CUTEstModel(probname) for probname in probnames)

# Sélection avec l'outil de CUTEst
shortlist = (CUTEstModel(probname) for probname in sort(CUTEst.select(min_var=100,
                                                                      max_var=500,
                                                                      max_con=0,
                                                                      only_free_var=true)))
problems = (shortlist)

@show problems

# autres collections, comme AMPL, etc.......

#ampl_prob_dir = "/home/dussault/import/decoded_ampl_models/decoded_ampl_models-master/cute/"
#DIR = pwd()
#cd(ampl_prob_dir)
#probnames = ["rosenbr.nl", "vardim.nl", "penalty1.nl"]
#problems = (AmplModel(probname) for probname in probnames)
#cd(DIR)


for (fn, fnsimple) ∈ [(:lbfgs, :lb), (:trunk, :TR), (:SESOP, :SE)]  #(:SESOP, :SE), (:tron, :TRO)
    @eval begin

        function $fnsimple(prob)


            # on peut définir ici un éventuel stop = stopping(...) commun pour les comparaisons
            # stop = ...

            #nlp_at_x = NLPAtX(prob.meta.x0)
            #stp = NLPStopping(prob, unconstrained_check, nlp_at_x)
            #stp.meta.max_iter = 500
            #stp.meta.atol = 1e-7
            #stp.max_cntrs[:neval_sum] = 10000
            #stp.meta.max_time =1.0

            if $fn == SESOP
                t = @timed begin
                    iterations, xopt, optimal, ngx, fx, status = eval(SESOP)(prob, max_time = 1.0)
                end
                res = GenericExecutionStats(status,
                                            prob,
                                            solution=xopt,
                                            iter=iterations,
                                            primal_feas=0.0,
                                            dual_feas=ngx,
                                            objective=fx,
                                            elapsed_time=t[2],
                                           )
            else
                res = eval($fn)(prob, max_time = 15.0)
            end

            # On pourrait combiner des stoppings avec des non-stoppings.



            # dans les cas de stopping, on peut adapter quelque chose comme ça, mais il faut
            # adapter au stopping moderne, ceci marchait... mais ne marche plus

            #status = :unknown
            #stop.meta.optimal && (status = :first_order)
            #stop.meta.unbounded && (status = :unbounded)
            #stop.meta.stalled && (status = :stalled)
            #stop.meta.tired && (status = :max_eval)  # should be more specific
            #
            #
            #res = GenericExecutionStats(status,
            #                            prob,
            #                            solution = final_state.x,
            #                            iter = stop.meta.nb_of_stop,
            #                            primal_feas = 0.0,
            #                            dual_feas = norm(final_state.gx),
            #                            objective = final_state.fx,
            #                            elapsed_time = t[2],
            #                            )

            return res

        end

    end
end


solvers = Dict{Symbol,Function}(
    :LBFGS  => lb,
    :TRUNK  => TR,
    :TRON  => TO,
    :SESOP5_b => SE_b,
    :SESOP5_TR => SE_tr,
    :SESOP5_nt => SE_nt,
    :SESOP5_lb => SE_lb,
    :SESOP5_clb => SE_clb
)

stats = bmark_solvers(solvers, problems)



# plot performance profiles
#
# Lorsque plusieurs solveurs, affiche la comparaison totale, puis les solveurs deux à deux.

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


#pretty_latex_stats(df)

p = profile_solvers(stats, costs, costnames)

#Pour générer directement un pdf:

Plots.pdf(p, "NomBidon.pdf")

