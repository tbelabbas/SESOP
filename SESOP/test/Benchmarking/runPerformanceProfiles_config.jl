using Logging

using NLPModels
using SolverBenchmark

using Printf
using DataFrames
using SolverTools
using SolverCore

using Stopping
using OneDmin

include("../../src/SESOP_wrappers.jl")


maxiter = 10000
maxtime = 600.0
Lp = Inf

atol = 1e-5
rtol = 1e-8

precision = 0.01
mem_SE = 4
max_iter_SE = 10

verbose = 0
subspace_verbose = 0
h = 2

test_case = "_configuration"

######################################################################################
######################################################################################

##################################  Problems collections
#
# ##################################  CUTEst collection
# using CUTEst

# pbnames = sort(CUTEst.select(#min_var=20,
#                                  max_var=20000,
#                                  max_con=0,
#                                  only_free_var=true))

##################################   OptimizationModels
using OptimizationProblems

meta = OptimizationProblems.meta
pbnames = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (100 .<=  meta.nvar .<=100), :name]

@show length(pbnames)
# Comment out the selection for testing the whole set
probnames = pbnames#[1:5]  # for testing quickly on a subset
# add a dummy copy of the first problem
probnames = insert!(probnames, 1, probnames[1])
# ##########################

# #################################     ADNLPModels
# using ADNLPModels
# using OptimizationProblems.ADNLPProblems

# problems = (eval(Meta.parse(probname))() for probname ∈ probnames)

# ##################################    PureJuMP
using NLPModelsJuMP
using OptimizationProblems.PureJuMP

problems = (MathOptNLPModel(eval(Meta.parse(probname))(), name=probname) for probname ∈ probnames)
@show length(problems)
# ##################################  CUTEst
# problems = (CUTEstModel(probname) for probname ∈ probnames)


#test_case = "JuMP"

my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st, pnorm = Lp; kwargs...)
df = DataFrame()
df.Type = ["Nb_it", "T(ms)", "Nb_o", "Nb_g", "Nb_H", "Nb_hprd", "‖∇f‖", "f", "status"]
global solvn = 0
# (:L_bfgs_StopLS, :LbfgsSLS), (:C_bfgs_StopLS, :CLbfgsSLS), (:M_bfgs_StopLS, :BfgsLS),
#(:SESOP_g0, :SESOP_0),
#, (:SESOP_g5, :SESOP_5)
for (fn, fnsimple) ∈ [(:SESOP_g1, :SESOP_1), (:SESOP_g2, :SESOP_2), (:SESOP_g3, :SESOP_3), (:SESOP_g4, :SESOP_4)]

    @eval begin

        function $fnsimple(nlp)

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

            global config = []

            # Actual timing and execution
            #t = @timed    (stp,) = eval($fn)(nlp, stp = stp)
            t = @timed    (stp, normes, objectives) = eval($fn)(nlp, stp = stp)
            #@show $fn
            #@show typeof(stp)
            df.n = [
                    stp.meta.nb_of_stop,
                    t[2],
                    neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                    neval_hprod(nlp), norm(grad(nlp, stp.current_state.x)),
                    stp.current_state.fx, getStatus(stp)
                ]

            rename!(df, :n => "SESOP_$((config[1]))_$(nlp.meta.name)_$solvn")
            global solvn += 1

            d = DataFrame()
            d.Type = ["Nb_it", "T(ms)", "Nb_o", "Nb_g", "Nb_H", "Nb_hprd", "‖∇f‖", "f", "status"]
            d.results = [
                    stp.meta.nb_of_stop,
                    t[2],
                    neval_obj(nlp), neval_grad(nlp), neval_hess(nlp),
                    neval_hprod(nlp), norm(grad(nlp, stp.current_state.x)),
                    stp.current_state.fx, getStatus(stp)
                ]
            rename!(d, :results => "SESOP_$(solvn)_$(nlp.meta.name)")

            open("test/Benchmarking/confi/config_$(config[1]).txt","a") do io
                println(io, "_____________________________________________")
                println(io,"NLP = $(nlp.meta.name) | N = $(nlp.meta.nvar) | config : [$(config)]")
                println(io, d)
                println(io, "_____________________________________________\n")
            end

            open("test/Benchmarking/confi/config_normes$(config[1]).txt","a") do io
                println(io, "$(nlp.meta.name), ", normes)
            end

            open("test/Benchmarking/confi/config_objectives$(config[1]).txt","a") do io
                println(io, "$(nlp.meta.name), ", objectives)
            end

            iter = stp.meta.nb_of_stop#

            xsol = stp.current_state.x
            fx = stp.current_state.fx
            gx = stp.current_state.gx

            status = getStatus(stp)


            return GenericExecutionStats(status, nlp,
                                         solution = xsol,
                                         iter = iter,
                                         dual_feas = stp.current_state.current_score,
                                         objective = fx,
                                         elapsed_time = t[2],
                                         )
        end

    end
end

#Comment out the "Ordered" parts to preserve the solver order in the legends.
#Ordered collections does not work with JSO tools to manipulate tables of (sub) results from the data frames

using OrderedCollections
include("../../../../../BenchmarksLSDescent/my_profile_tools.jl")   # fournit aussi named_bark_solver
#include("../../../../BenchmarksLSDescent/my_profile_tools.jl")   # fournit aussi named_bark_solver





solvers = OrderedDict{Symbol,Function}(
    :G4_P1 => SESOP_4,
    :G3_P2 => SESOP_3,
    :G2_P3 => SESOP_2,
    :G1_P4 => SESOP_1
)



stats = ordered_bmark_solvers(solvers, problems,
                      colstats = [:name, :nvar, :status, :neval_obj, :neval_grad, :objective, :dual_feas, :elapsed_time])

# remove the dummy copy of the first problem for statistics
for s in solvers
    stats[s[1]] = stats[s[1]][2:end,:]
end


open("test/Benchmarking/config.txt","a") do io
    println(io, "_____________________________________________\n")
    println(io, df)
    println(io, "\n_____________________________________________")
end

using FileIO
using JLD2
#
# Save the stats produced above:
#

file = File{format"JLD2"}("Stats_config.jld2")
save(file, "stats", stats)

#include("../../../BenchmarksLSDescent/AllFail.jl")