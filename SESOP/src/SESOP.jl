using JSOSolvers
using SolverTools
using SolverCore
using NLPModels

using Stopping

using LinearAlgebra, Logging

include("Utils/SubspaceData.jl")
include("Models/SEModel.jl")

include("Solvers/Solvers.jl")


function SESOP(nlp :: AbstractNLPModel;
               stp :: NLPStopping = NLPStopping(nlp,
                                                NLPAtX(nlp.meta.x0)),
               atol = 1e-8,
               rtol = 1e-5,
               precision = 0.4,
               maxtime = 10.0,
               mem_SE :: Int = 4,
               max_iter_SE = 10,
               mem :: Union{Int, Vector{Int}, LMC} = [1, 3, 0],
               verbose :: Int = 0,
               subspace_verbose :: Int = 0,
               subspace_solver :: Function = bfgs_StopLS
              )
    iter = 0
    x = nlp.meta.x0
    n = length(x)

    subdata = SubspaceData(n, m = mem)

    f = obj(nlp, x)
    gradient = grad(nlp, x)
    norm_g = norm(gradient)

    D = ones(n, 1).*gradient
    SE = SEModel(nlp, x, D, gradient)

    ############################################################################
    T = Float64
    verbose == 1 && @info log_header([:iter, :dual, :callsObj, :callsGrad],
                                [Int, T, T, T, T],
    hdr_override=Dict(:dual=>"‖∇f‖"))
    verbose == 1 && @info log_row(Any[iter, norm_g])
    ############################################################################

    OK = update_and_start!(stp, x = x, fx = f, gx = gradient)


    while !OK

        results = solve(SE, max_iter_SE, maxtime/2.0,
                        0.0,
                        precision,
                        subspace_solver, subspace_verbose)

        x_prime = x + D * results.solution

        gradient = SE.∇f
        updateGradients!(subdata, gradient)

        newdir = x_prime - x
        updatePreviousDirs!(subdata, newdir)

        x = x_prime

        nobj = neval_obj(nlp)
        ngrad = neval_grad(nlp)
        norm_g = norm(gradient)

        #######################################################################
        verbose == 1 && @info log_row(Any[iter, norm_g, nobj, ngrad])
        #######################################################################

        updateNormes!(subdata, norm_g)
        updateObjectives!(subdata, results.objective)

        D = getSubspace(subdata)
        F = qr!(D)
        D = Matrix(F.Q)
        SE = SEModel(nlp, x, D, gradient)

        iter+=1

        OK = update_and_stop!(stp, x = x, gx = gradient, fx = results.objective)
    end

    return stp, subdata.Normes, subdata.Objectives#, iter, x, stp.current_state.fx
end

"""
    SESOP solvers as wrapper of the general SESOP function
"""
# BFGS as subspace solver
function SESOP_bfgs(nlp :: AbstractNLPModel; kwargs...)
    return SESOP(nlp; subspace_solver=bfgs_StopLS,
                 subspace_verbose=verbose,
                 kwargs...)
end

# LBFGS as subspace solver
function SESOP_lbfgs(nlp :: AbstractNLPModel; kwargs...)
    return SESOP(nlp; subspace_solver=L_bfgs_StopLS, kwargs...)
end

# Compact LBFGS as subspace solver
function SESOP_clbfgs(nlp :: AbstractNLPModel; kwargs...)
    return SESOP(nlp; subspace_solver=C_bfgs_StopLS, kwargs...)
end

# Cholesky BFGS as subspace solver
function SESOP_chlbfgs(nlp :: AbstractNLPModel; kwargs...)
    return SESOP(nlp; subspace_solver=Ch_bfgs_StopLS, kwargs...)
end

# Newton spectral as subspace solver
function SESOP_newton(nlp :: AbstractNLPModel;
                      stp :: NLPStopping = NLPStopping(nlp,
                                                       NLPAtX(nlp.meta.x0)),
                      atol = 1e-8,
                      rtol = 1e-5,
                      precision = 0.4,
                      maxtime = 10.0,
                      mem_SE :: Int = 4,
                      max_iter_SE = 50,
                      mem :: Union{Int, Vector{Int}, LMC} = [1, 3, 0],
                      verbose :: Int = 0,
                      subspace_verbose :: Int = 0,
                     )

    iter = 0
    x = nlp.meta.x0
    n = length(x)

    subdata = SubspaceData(n, m = mem)

    f = obj(nlp, x)
    gradient = grad(nlp, x)
    norm_g = norm(gradient)

    D = ones(n, 1).*gradient
    SE = SEModel(nlp, x, D, gradient)

    ############################################################################
    T = Float64
    verbose == 1 && @info log_header([:iter, :dual, :callsObj, :callsGrad],
                                [Int, T, T, T, T],
    hdr_override=Dict(:dual=>"‖∇f‖"))
    verbose == 1 && @info log_row(Any[iter, norm_g])
    ############################################################################

    OK = update_and_start!(stp, x = x, fx = f, gx = gradient)


    while !OK

        results = Newton_Solver(SE, max_iter_SE,
                                verbose=subspace_verbose,
                                precision=precision*norm(gradient))

        x_prime = x + D * results.solution

        gradient = SE.∇f
        updateGradients!(subdata, LinearAlgebra.normalize(gradient))

        newdir = x_prime - x
        updatePreviousDirs!(subdata, newdir)

        x = x_prime

        nobj = neval_obj(nlp)
        ngrad = neval_grad(nlp)
        norm_g = norm(gradient)

        ########################################################################
        verbose == 1 && println("_____________________________________________")
        verbose == 1 && @info log_row(Any[iter, norm_g, nobj, ngrad])
        ########################################################################
        updateWeights!(subdata, norm_g)
        D = getSubspace(subdata)
        SE = SEModel(nlp, x, D, gradient)

        OK = update_and_stop!(stp, x = x, gx = gradient, fx = results.objective)
    end

    return stp#, iter, x, stp.current_state.fx
end


# Trunk as subspace solver
function SESOP_trunk( nlp :: AbstractNLPModel;
                      stp :: NLPStopping = NLPStopping(nlp,
                                                       NLPAtX(nlp.meta.x0)),
                      atol = 1e-8,
                      rtol = 1e-5,
                      precision = 0.4,
                      maxtime = 10.0,
                      mem_SE :: Int = 4,
                      max_iter_SE = 50,
                      mem :: Union{Int, Vector{Int}, LMC} = [1, 3, 0],
                      verbose :: Int = 0,
                      subspace_verbose :: Int = 0,
                     )

    iter = 0
    x = nlp.meta.x0
    n = length(x)

    subdata = SubspaceData(n, m = mem)

    f = obj(nlp, x)
    gradient = grad(nlp, x)
    norm_g = norm(gradient)

    D = ones(n, 1).*gradient
    SE = SEModel(nlp, x, D, gradient)

    ############################################################################
    T = Float64
    verbose == 1 && @info log_header([:iter, :dual, :callsObj, :callsGrad],
                                [Int, T, T, T, T],
    hdr_override=Dict(:dual=>"‖∇f‖"))
    verbose == 1 && @info log_row(Any[iter, norm_g])
    ############################################################################

    OK = update_and_start!(stp, x = x, fx = f, gx = gradient)


    while !OK

        results = trunk(SE; verbose=subspace_verbose,
                        atol = atol*precision*norm(gradient),
                        rtol = rtol*precision*norm(gradient),
                        max_eval = max_iter_SE, max_time=maxtime)

        x_prime = x + D * results.solution

        gradient = SE.∇f
        updateGradients!(subdata, LinearAlgebra.normalize(gradient))

        newdir = x_prime - x
        updatePreviousDirs!(subdata, newdir)

        x = x_prime

        nobj = neval_obj(nlp)
        ngrad = neval_grad(nlp)
        norm_g = norm(gradient)

        ########################################################################
        verbose == 1 && println("_____________________________________________")
        verbose == 1 && @info log_row(Any[iter, norm_g, nobj, ngrad])
        ########################################################################
        updateWeights!(subdata, norm_g)
        D = getSubspace(subdata)
        SE = SEModel(nlp, x, D, gradient)

        OK = update_and_stop!(stp, x = x, gx = gradient, fx = results.objective)
    end

    return stp#, iter, x, stp.current_state.fx
end