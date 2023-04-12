
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
pbnames = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (2 .<=  meta.nvar .<=20000), :name]

@show length(pbnames)
# Comment out the selection for testing the whole set
probnames = pbnames #[10:15]  # for testing quickly on a subset
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

# (:L_bfgs_StopLS, :LbfgsSLS), (:C_bfgs_StopLS, :CLbfgsSLS), (:M_bfgs_StopLS, :BfgsLS),
for (fn, fnsimple) ∈ [(:SESOP_pastiche_1, :SESOP_id_B), (:SESOP_pastiche_3, :SESOP_DBD_B), (:SESOP_pastiche_4, :SESOP_hess_B), (:SESOP_lbfgs, :SESOPlb), (:SESOP_trunk, :SESOPtr), (:SESOP_newton, :SESOPnt)]

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

            # Actual timing and execution
            #t = @timed    (stp,) = eval($fn)(nlp, stp = stp)
            t = @timed    stp = eval($fn)(nlp, stp = stp)
            #@show $fn
            #@show typeof(stp)

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
include("../../../../BenchmarksLSDescent/my_profile_tools.jl")   # fournit aussi named_bark_solver





solvers = OrderedDict{Symbol,Function}(
#solvers = Dict{Symbol,Function}(
    #:CG_FR => FR,
    #:CG_PR => PR,
    #:CG_HS => HS,
    #:CG_HZ => HZ,
    # #:Lbfgs => LbfgsS,
    # :LbfgsLS => LbfgsSLS,
    # :CLbfgsLS => CLbfgsSLS,
    # :MbfgsLS  => BfgsLS,
    #
    :B0_id_bfgs => SESOP_id_B,
    :B0_DBD_bfgs => SESOP_DBD_B,
    :B0_hess_bfgs => SESOP_hess_B,
    #
    # :B0_id_lbfgs => SESOP_id_B,
    # :B0_DBD_lbfgs => SESOP_DBD_B,
    # :B0_hess_lbfgs => SESOP_hess_B,
    #
    # :B0_id_cbfgs => SESOP_id_B,
    # :B0_DBD_cbfgs => SESOP_DBD_B,
    # :B0_hess_cbfgs => SESOP_hess_B,
    #
    # :B0_id_newton => SESOPnt,
    # :B0_id_lbfgs => SESOPlb,
    # :B0_id_trunk => SESOPtr,
    #
    #:NwtLS   => NwtSLS,
    #:NwtCG   =>   NwtCG,
    #:Nwt   =>   NwtS,
)



stats = ordered_bmark_solvers(solvers, problems,
                      colstats = [:name, :nvar, :status, :neval_obj, :neval_grad, :objective, :dual_feas, :elapsed_time])

# remove the dummy copy of the first problem for statistics
for s in solvers
    stats[s[1]] = stats[s[1]][2:end,:]
end


using FileIO
using JLD2
#
# Save the stats produced above:
#

file = File{format"JLD2"}("Stats"*test_case*".jld2")
save(file, "stats", stats)

#include("../../../BenchmarksLSDescent/AllFail.jl")