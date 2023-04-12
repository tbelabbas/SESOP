using JSOSolvers
using SolverTools
using SolverCore
using NLPModels

using Stopping

using LinearAlgebra, Logging

include("Utils/SubspaceData.jl")
include("Models/SEModel.jl")

include("Solvers/Solvers.jl")

function SESOP_pastiche_1(nlp :: AbstractNLPModel;
                            stp :: NLPStopping = NLPStopping(nlp,
                                                            NLPAtX(nlp.meta.x0)),
                            atol = 1e-5,
                            rtol = 1e-7,
                            precision = 0.001,
                            maxtime = 10.0,
                            mem_SE :: Int = 4,
                            max_iter_SE = 10,
                            mem :: Union{Int, Vector{Int}, LMC} = [1, 4, 0],
                            verbose :: Int = 0,
                            subspace_verbose :: Int = 0,
                            subspace_solver :: Function = bfgs_StopLS
                        )
    return SESOP_pastiche(  nlp; stp=stp, atol = atol, rtol = rtol,
                            precision = precision,
                            maxtime = maxtime,
                            mem_SE = mem_SE,
                            max_iter_SE = max_iter_SE,
                            mem = mem,
                            verbose = verbose,
                            subspace_verbose = subspace_verbose,
                            subspace_solver = subspace_solver,
                            matrix = 1
                        )
end

function SESOP_pastiche_3(nlp :: AbstractNLPModel;
                            stp :: NLPStopping = NLPStopping(nlp,
                                                            NLPAtX(nlp.meta.x0)),
                            atol = 1e-5,
                            rtol = 1e-7,
                            precision = 0.001,
                            maxtime = 10.0,
                            mem_SE :: Int = 4,
                            max_iter_SE = 10,
                            mem :: Union{Int, Vector{Int}, LMC} = [1, 4, 0],
                            verbose :: Int = 0,
                            subspace_verbose :: Int = 0,
                            subspace_solver :: Function = bfgs_StopLS
                        )
    return SESOP_pastiche(  nlp; stp=stp, atol = atol, rtol = rtol,
                            precision = precision,
                            maxtime = maxtime,
                            mem_SE = mem_SE,
                            max_iter_SE = max_iter_SE,
                            mem = mem,
                            verbose = verbose,
                            subspace_verbose = subspace_verbose,
                            subspace_solver = subspace_solver,
                            matrix = 3
                        )
end

function SESOP_pastiche_4(nlp :: AbstractNLPModel;
                            stp :: NLPStopping = NLPStopping(nlp,
                                                            NLPAtX(nlp.meta.x0)),
                            atol = 1e-5,
                            rtol = 1e-7,
                            precision = 0.001,
                            maxtime = 10.0,
                            mem_SE :: Int = 4,
                            max_iter_SE = 10,
                            mem :: Union{Int, Vector{Int}, LMC} = [1, 4, 0],
                            verbose :: Int = 0,
                            subspace_verbose :: Int = 0,
                            subspace_solver :: Function = bfgs_StopLS
                        )
    return SESOP_pastiche(  nlp; stp=stp, atol = atol, rtol = rtol,
                            precision = precision,
                            maxtime = maxtime,
                            mem_SE = mem_SE,
                            max_iter_SE = max_iter_SE,
                            mem = mem,
                            verbose = verbose,
                            subspace_verbose = subspace_verbose,
                            subspace_solver = subspace_solver,
                            matrix = 4
                        )
end


function grad_descent(mdl, maxiter, maxtime, atol, rtol)
    x = mdl.meta.x0
    f = obj(mdl, x)
    g = grad(mdl, x)

    reset!(mdl)

    # setup the Stopping object
    stp = NLPStopping(mdl,
                    StoppingMeta(),
                    StopRemoteControl(domain_check = false),
                    NLPAtX(mdl.meta.x0)  )


    my_unconstrained_check(mdl, st; kwargs...) = unconstrained_check(mdl, st,
                                                                     pnorm = Inf;
                                                                     kwargs...
                                                                     )

    stp.meta.optimality_check = my_unconstrained_check

    stp.meta.max_iter = maxiter
    stp.meta.max_time = maxtime
    stp.meta.atol = atol
    stp.meta.rtol = rtol

    @show f


    OK = update_and_start!(stp, x = x, fx = f, gx = g)

    # start gradient descent algorithm
    while !OK
        α = f ./ g
        # update x until objective funtion value is minimized
        #@show norm(α)
        if norm(x - α) > norm(x)
            OK = true
        else
            x = x - α

            # gradient of objective function
            g = grad(mdl, x)
            f = obj(mdl, x)

            @show f
            OK = update_and_stop!(stp, x = x, gx = g, fx = f)
        end
    end

    iter = stp.meta.nb_of_stop

    xsol = stp.current_state.x
    fx = stp.current_state.fx
    gx = stp.current_state.gx

    status = getStatus(stp)



    @show iter, status
    @show length(x)
    return x
end


function SESOP_pastiche(nlp :: AbstractNLPModel;
                        stp :: NLPStopping = NLPStopping(nlp,
                                                         NLPAtX(nlp.meta.x0)),
                        atol = 1e-8,
                        rtol = 1e-5,
                        precision = 0.4,
                        maxtime = 5000.0,
                        mem_SE :: Int = 4,
                        max_iter_SE = 10,
                        mem :: Union{Int, Vector{Int}, LMC} = [1, 3, 0],
                        verbose :: Int = 0,
                        subspace_verbose :: Int = 0,
                        subspace_solver :: Function = bfgs_StopLS,
                        h = 2,
                        matrix::Int = 0
                       )

    iter = 0
    x₀ = nlp.meta.x0
    x₀ = ones(length(nlp.meta.x0))
    x = x₀
    n = length(x)

    subdata = SubspaceData(n, m = mem)

    f = obj(nlp, x)
    gradient = grad(nlp, x)
    norm_g = norm(gradient)

    D = ones(n, 1).*normalize(gradient)

    # display(D)


    SE = SEModel(nlp, x, D, gradient)
    α₀ = zeros(1)

    #B_3 = CompactInverseBFGSOperator(Float64, n, mem=sum(mem))
    #B_2 = InverseBFGSOperator(Float64, n, scaling=true)
    B = InverseLBFGSOperator(Float64, n, mem=sum(mem))

    # @show (norm(Matrix(B)-Matrix(B_2)))
    # @show (norm(Matrix(B)-Matrix(B_3)))
    # @show (norm(Matrix(B_2)-Matrix(B_3)))
    ############################################################################
    T = Float64
    verbose == 1 && @info log_header([:iter, :dual, :callsObj, :callsGrad],
                                [Int, T, T, T, T],
    hdr_override=Dict(:dual=>"‖∇f‖"))
    verbose == 1 && @info log_row(Any[iter, norm_g])
    ############################################################################

    OK = update_and_start!(stp, x = x, fx = f, gx = gradient)

    m = SE.meta.nvar
    B_proj = Matrix{Float64}(I, m, m)

    ω = 1.0

    while !OK
        #println("ITERATION $iter")
        m = SE.meta.nvar

        #Identité
        B0 = Matrix{Float64}(I, m, m)

        if matrix == 2
            #QN calculé dans le SE précédent
            println("B0")
            display(B0)
            println("B_proj")
            display(Matrix(B_proj))



            B_proj = Matrix(B_proj)
            view(B0, 1:size(B_proj,1), 1:size(B_proj,2)) .= B_proj

            println("B0")
            display(B0)

            Bk = D'*Matrix(B*D)
            B0[:, m] = Bk[:,m]
            B0[m, :] = Bk[m,:]

            if m-1 > 0
                B0[:, m-1] = Bk[:,m-1]
                B0[m-1, :] = Bk[m-1,:]
            end

            # B0[:, m] = zeros(m)
            # B0[m, :] = zeros(m)
            # B0[m, m] = 1.0
            println("B0")
            display(B0)

            B0 = T(0.5)*(B0 + B0')

            println("B0")
            display(B0)
            # B0[m, m] = 1.0
        elseif matrix == 3
            #QN global projeté dans le nouveau SE
            #println("proj_B")
            # B0 = @timeit to "proj_B" D'*Matrix(B*D)
            # B0 = @timeit to "symm" T(0.5)*(B0 + B0')
            B0 = D'*Matrix(B*D)
            B0 = T(0.5)*(B0 + B0')
            #println("proj_B done")
        elseif matrix == 4
            H = hess(SE, SE.meta.x0)

            O = eigvecs(H)
            Δ = eigvals(H)

            Diag = Δ + max.(real.(1e-8 .- Δ), 0.0) .*ones(length(SE.meta.x0))

            B0 = O*diagm(1.0 ./ Diag)*O'
        end


        subspace_verbose = 1
        B_proj, results = solve_pastiche(SE, max_iter_SE, maxtime/2.0,
                                         0.0,
                                         0.05,
                                        #  atol,
                                        #  rtol,
                                         subspace_solver, subspace_verbose,
                                         B0)

        α = D * results.solution
        f = results.objective

        # β =  D * a
        β = α

        B = push!(B, β, (SE.∇f - gradient))

        gradient = SE.∇f
        x += β

        nobj = neval_obj(nlp)
        ngrad = neval_grad(nlp)
        norm_g = norm(gradient)

        #######################################################################
        verbose == 1 && @info log_row(Any[iter, norm_g, nobj, ngrad])
        #######################################################################

        updateGradients!(subdata, normalize(gradient))
        updatePreviousDirs!(subdata, normalize(β))
        # updateNemirovskiDirs!(subdata, ω*gradient)
        updateNemirovskiDirs!(subdata, normalize(x - x₀))
        D = getSubspace(subdata)

        F = qr!(D)
        D = Matrix(F.Q)

        SE = SEModel(nlp, x, D, gradient)

        iter+=1

        OK = update_and_stop!(stp, x = x, gx = gradient, fx = results.objective)

        l=Int(sqrt(nlp.meta.nvar))
        nr=l
        nc=l
        Xs = reshape(stp.current_state.x,nr,nc)
        Bimg = map(clamp01nan, Xs)
        name = "SESOP/testSESOP_" * "$(iter).png"
        open("SESOP/sesop.txt", "a") do io
            println(io, stp.current_state.current_time - stp.meta.start_time , " ", stp.current_state.fx, " ", norm(stp.current_state.gx))
        end;

        ImageFiles && save(name,  Bimg)

    end


    return stp, iter, x, stp.current_state.fx#, TimerOutputs.time(to2["mat"])*10e-6
end

function SESOP_pastiche_pcc(nlp :: AbstractNLPModel;
                        stp :: NLPStopping = NLPStopping(nlp,
                                                         NLPAtX(nlp.meta.x0)),
                        atol = 1e-8,
                        rtol = 1e-5,
                        precision = 0.4,
                        maxtime = 5000.0,
                        mem_SE :: Int = 4,
                        max_iter_SE = 10,
                        mem :: Union{Int, Vector{Int}, LMC} = [1, 3, 0],
                        verbose :: Int = 0,
                        subspace_verbose :: Int = 0,
                        subspace_solver :: Function = bfgs_StopLS,
                        h = 2,
                        matrix::Int = 0
                       )

    iter = 0
    x₀ = ones(length(nlp.meta.x0))
    x = x₀
    n = length(x)

    subdata = SubspaceData(n, m = mem)

    f = obj(nlp, x)
    gradient = grad(nlp, x)
    norm_g = norm(gradient)

    D = ones(n, 1).*normalize(gradient)


    update_values!(nlp.m1, x, D)
    SE = SEModel_pcc(nlp, x, D, gradient)
    α₀ = zeros(1)

    #B_3 = CompactInverseBFGSOperator(Float64, n, mem=sum(mem))
    #B_2 = InverseBFGSOperator(Float64, n, scaling=true)
    B = InverseLBFGSOperator(Float64, n, mem=sum(mem))

    ############################################################################
    T = Float64
    verbose == 1 && @info log_header([:iter, :dual, :callsObj, :callsGrad],
                                [Int, T, T, T, T],
    hdr_override=Dict(:dual=>"‖∇f‖"))
    verbose == 1 && @info log_row(Any[iter, norm_g])
    ############################################################################

    OK = update_and_start!(stp, x = x, fx = f, gx = gradient)

    m = SE.meta.nvar
    B_proj = Matrix{Float64}(I, m, m)

    ω = 1.0

    while !OK

        m = SE.meta.nvar
        #Identité
        B0 = Matrix{Float64}(I, m, m)

        subspace_verbose = 1
        B_proj, results = solve_pastiche(SE, max_iter_SE, maxtime/2.0,
                                         0.0,
                                         0.05,
                                        #  atol,
                                        #  rtol,
                                         subspace_solver, subspace_verbose,
                                         B0)

        α = D * results.solution
        f = results.objective


        β = α

        B = push!(B, β, (SE.∇f - gradient))

        gradient = SE.∇f

        x += β

        nobj = neval_obj(nlp)
        ngrad = neval_grad(nlp)
        norm_g = norm(gradient)

        #######################################################################
        verbose == 1 && @info log_row(Any[iter, norm_g, nobj, ngrad])
        #######################################################################

        updateGradients!(subdata, normalize(gradient))
        updatePreviousDirs!(subdata, normalize(β))
        D = getSubspace(subdata)
        F = qr!(D)
        D = Matrix(F.Q)
        update_values!(nlp.m1, x, D)
        SE = SEModel_pcc(nlp, x, D, gradient)

        iter+=1

        OK = update_and_stop!(stp, x = x, gx = gradient, fx = results.objective)

        l=Int(sqrt(nlp.meta.nvar))
        nr=l
        nc=l
        Xs = reshape(stp.current_state.x,nr,nc)
        Bimg = map(clamp01nan, Xs)
        name = "SESOP_pcc/testSESOP_pcc_" * "$(iter).png"
        open("SESOP_pcc/sesop.txt", "a") do io
            println(io, stp.current_state.current_time - stp.meta.start_time , " ", stp.current_state.fx, " ", norm(stp.current_state.gx))
        end;
        ImageFiles && save(name,  Bimg)
    end


    return stp, iter, x, stp.current_state.fx#, TimerOutputs.time(to2["mat"])*10e-6
end

function solve_pastiche(nlp, maxiter, maxtime, atol, rtol,
               fn::Function, subspace_verbose::Int,
               B₀)

    # setup the Stopping object
    stp = NLPStopping(nlp,
                    StoppingMeta(),
                    StopRemoteControl(domain_check = false),
                    NLPAtX(nlp.meta.x0)  )


    my_unconstrained_check(nlp, st;
                            kwargs...) = unconstrained_check(nlp, st,
                                                                pnorm = Inf;
                                                                kwargs...
                                                            )

    stp.meta.optimality_check = my_unconstrained_check

    stp.meta.max_iter = maxiter
    stp.meta.max_time = maxtime
    stp.meta.atol = atol
    stp.meta.rtol = rtol

    reset!(nlp)
    reinit!(stp)

    global t = nothing

    #let t=t
    with_logger(NullLogger()) do
        t = @timed stp = fn(nlp, stp = stp,# x = α₀, #m=3,
                            verbose = 1, B₀=B₀)
    end
    #end

    iter = stp.meta.nb_of_stop

    xsol = stp.current_state.x
    fx = stp.current_state.fx
    gx = stp.current_state.gx

    # if m.fail_sub_pb
        # FAIRE un pas dans la direction du gradient
    # end

    status = getStatus(stp)

    if stp.meta.fail_sub_pb
        @warn "Fail_sup_pb"
    end

    return stp.stopping_user_struct["BFGS"], GenericExecutionStats(status, nlp,
                                    solution = xsol,
                                    iter = iter,
                                    dual_feas = stp.current_state.current_score,
                                    objective = fx,
                                    elapsed_time = t[2],
                                )
end

function getStatus(stp :: NLPStopping)
    status = :unknown
    # stopping properties status of the problem in Stopping
    m = stp.meta

    m.domainerror     && (status = :domainerror)
    m.unbounded       && (status = :unbounded)
    m.unbounded_pb    && (status = :unbounded_success)
    m.fail_sub_pb     && (status = :unknown)
    m.tired           && (status = :max_time)
    m.stalled         && (status = :stalled)
    m.iteration_limit && (status = :max_iter)
    m.resources       && (status = :max_time)
    m.optimal         && (status = :first_order)
    m.infeasible      && (status = :infeasible)
    m.main_pb         && (status = :unknown)
    m.suboptimal      && (status = :unknown)
    m.stopbyuser      && (status = :user)
    m.exception       && (status = :exception)

    return status
end