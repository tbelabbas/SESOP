using LinearAlgebra

using OptimizationProblems
using NLPModelsJuMP
using NLPModels

using SolverTools
using Krylov

using LinearOperators

using Hadamard

#include("../src/NLPQuad.jl")
include("../src/BPNDModel.jl")
include("../src/SESOP.jl")
#include("../SESOP/GC_linCol.jl")

function generateSignal(γ, n)
    x = zeros(n)
    for i = 1 : n
        a = rand()
        if a < γ
            s = rand()
            if s < 0.5
                x[i] = -1
            else
                x[i] = 1
            end
        end
    end
    return x
end

function awgn(X,SNR)
    #Assumes X to be a matrix and SNR a signal-to-noise ratio specified in decibel (dB)
    #Implented by author, inspired by https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    N=length(X) #Number of elements in X
    signalPower = sum(X[:].^2)/N
    linearSNR = 10^(SNR/10)
    a=size(X)
    noiseMat = randn((a)).*√(signalPower/linearSNR) #Random gaussian noise, scaled according to the signal power of the entire matrix (!) and specified SNR

    return solution = X + noiseMat
end

function compareCGandSESOP(N :: Int = 1024, L :: Int = 24)

    K = L*N
    mmA = [hadamard(N) Matrix{Float64}(I, N, K - N)]
    z0 = generateSignal(0.3, K)
    mA = mmA'*mmA
    x0 = mA * z0
    b = awgn(x0, 25)

    A = LinearOperator(mA)

    nlp = BPNDModel(A, b, w=0.0)

    T = Float64
    @info log_header([:model, :nbiter, :f, :dual, :nObj, :nGrad, :nprod, :ncprod], [String, Int, T, T, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖"))

    reset!(nlp)
    reset!(A)

    results = nothing
    # lbfgs
    with_logger(lLogger()) do
        results = lbfgs(nlp, verbose = false)
        @show results.iter
    end

    @info log_row(Any["LBFGS", results.iter, results.objective, results.dual_feas, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])

    reset!(nlp)
    reset!(A)

    #  trunk

    with_logger(lLogger()) do
        try
            results = trunk(nlp)
        catch e
            println("Outside trust region")
        end
    end

    @info log_row(Any["Trunk", results.iter, results.objective, results.dual_feas, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])

    reset!(nlp)
    reset!(A)

    #  tron

    with_logger(lLogger()) do
        try
            results = tron(nlp)
        catch e
            println("Outside trust region")
        end
    end

    @info log_row(Any["Tron", results.iter, results.objective, results.dual_feas, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])


    reset!(nlp)
    reset!(A)

    # SESOP-01

    iter, x_opt, optimal, ng, fx = SESOP(nlp, verbose=false, mem=1, exact=false)
    @info log_row(Any["SESOP-01", iter, fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])

    reset!(nlp)
    reset!(A)

    # SESOP-03

    iter, x_opt, optimal, ng, fx = SESOP(nlp, verbose=false, mem=3, exact=false)
    @info log_row(Any["SESOP-03", iter,  fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])

    reset!(nlp)
    reset!(A)

    # SESOP-08

    iter, x_opt, optimal, ng, fx = SESOP(nlp, verbose=false, mem=8, exact=false)
    @info log_row(Any["SESOP-08", iter,  fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])

    reset!(nlp)
    reset!(A)

    ## SESOP-01 Opt
#
    #iter, x_opt, optimal, ng, fx = optSESOP(A, b, verbose=false, mem=1, exact=false)
    #@info log_row(Any["SESOP-01_opt", iter,  fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])
    #
    #reset!(nlp)
    #reset!(A)
#
    ## SESOP-03 Opt
#
    #iter, x_opt, optimal, ng, fx = optSESOP(A, b, verbose=false, mem=3, exact=false)
    #@info log_row(Any["SESOP-03_opt", iter,  fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])
    #
    #reset!(nlp)
    #reset!(A)
#
    ## SESOP-08 Opt
#
    #reset!(nlp)
    #reset!(A)
#
    #iter, x_opt, optimal, ng, fx = optSESOP(A, b, verbose=false, mem=8, exact=false)
    #@info log_row(Any["SESOP-08_opt", iter,  fx, ng, neval_obj(nlp), neval_grad(nlp), A.nprod, A.nctprod])


    return A, b;
end


A, b = compareCGandSESOP()

;