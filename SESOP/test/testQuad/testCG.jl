using LinearAlgebra

using OptimizationProblems
using JuMP
using NLPModelsJuMP
using NLPModels
#using SparseArrays

using SolverTools  
using Krylov
using Random
using QuadraticModels
using LinearOperators


include("../../src/NLPQuad.jl")
#include("../../src/GC_lin.jl")
include("../../src/SESOP.jl")

function objFunc(Q :: Matrix, c :: AbstractVector, x :: AbstractVector)
    return (0.5 * x' * Q * x + c' * x)'
end

function gradFunc(Q :: Matrix, c :: AbstractVector, x :: AbstractVector)
    return Q * x + c
end



function compareCGandSESOP(nlp; n :: Int = 30)
    
    #quadNLP = QuadraticModel(nlp, nlp.meta.x0)
    #hop = hess_op(quadNLP, quadNLP.meta.x0) 

    #@show isposdef(Matrix(hop))

    #Λ = [1.0; 5.0; 5.0; 100.0; 100.0; 100.0]
    #Λ = [100.0; 5.0; 5.0; 100.0; 100.0; 100.0]
    Λ = [0.00001; 5.0; 30.0; 55.0; 5.0; 10000000; 150.0; 80.0; 100.0]
    n = length(Λ)
    M=rand(n,n)
    O,R = qr(M)
    Q = O*diagm(Λ)*O'
    
    Q = 0.5*(Q+Q')
    
    c = rand(n)

    @show size(Q)
    @show size(c)

    T = Float64
    @info log_header([:model, :nbiter, :f, :dual], [String, Int, T, T],
    hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖"))

    # Krylov CG - quad 
    (x_opt, stats, iter) = cg(Q, -c, itmax=30,verbose=true)
    @info log_row(Any["quadGC_Kryv", iter, objFunc(Q, c, x_opt), norm(gradFunc(Q, c, x_opt))])

    # reset!(nlp)    
    
    # SESOP-01 - exact
    iter, x_opt, optimal = quadSESOP3(Q, c, verbose=true, maxiter=30)
    @info log_row(Any["SESOP-1-exact", iter, objFunc(Q, c, x_opt), norm(gradFunc(Q, c, x_opt))])
    #@info log_row(Any["SESOP-1-lbfgs", iter, obj(quadNLP, x_opt), norm(grad(quadNLP, x_opt))])

    # SESOP-01 - lbfgs
    quadNLP = NLPQuad(Q,-c)
    iter, x_opt, optimal = SESOP(quadNLP, verbose=false, mem=1, maxiter=30)
    @info log_row(Any["SESOP-1-lbfgs", iter, objFunc(Q, c, x_opt), norm(gradFunc(Q, c, x_opt))])


    
    # xBFGS = BFGS_lin(mse)mse = SEModel(Z, mq, x0)
    # println("SE avec Z rand")
    
    # println("Gradient conjugué.")
    # xGC = GC_lin(mse)
    
    # include("BFGS_QuadNLP.jl")
    # println("BFGS.")
    #reset!(quadNLP)    
    
    # # SESOP-01 - quad
    # iter, x_opt, optimal = quadSESOP2(quadNLP, verbose=true)
    # @info log_row(Any["quadSESOP-1-ext", iter, obj(quadNLP,x_opt), norm(grad(quadNLP,x_opt)), neval_obj(quadNLP), neval_grad(quadNLP)])

    # # # Krylov CG
    # hop = hess_op(nlp, nlp.meta.x0) 
    # (x_opt, stats, iter) = cg(hop, ones(length(nlp.meta.x0)), itmax=50, atol=1e-15, rtol=1e-15)
    # @info log_row(Any["GC_Krylov", iter, obj(nlp,x_opt), norm(grad(nlp,x_opt)), neval_obj(nlp), neval_grad(nlp)])

    # reset!(nlp)

    # # # SESOP-01 - exact
    # iter, x_opt, optimal = quadSESOP(nlp, verbose=false)
    # @info log_row(Any["SESOP-1-exact", iter, obj(nlp,x_opt), norm(grad(nlp,x_opt)), neval_obj(nlp), neval_grad(nlp)])

    # reset!(quadNLP)    

    # # Descent Methods GC
    # nlpatx = NLPAtX(quadNLP.meta.x0)
    # nlpstop = NLPStopping(quadNLP, unconstrained_check, nlpatx)
    # final_nlp_at_x, optimal, iter = CG_generic(quadNLP, nlpstop)
    # @info log_row(Any["GC", iter, final_nlp_at_x.current_state.fx, norm(final_nlp_at_x.current_state.gx), neval_obj(quadNLP), neval_grad(quadNLP)])

    # reset!(quadNLP)

    # # lbfgs
    # results = lbfgs(quadNLP) 
    # @info log_row(Any["LBFGS", results.iter, results.objective, results.dual_feas, neval_obj(quadNLP), neval_grad(quadNLP)])

end

n = 4

nlp = MathOptNLPModel(genrose(n))
#nlp = MathOptNLPModel(dixmaane(n))
#nlp = MathOptNLPModel(rosenbrock(n))

#nlp = MathOptNLPModel(dixmaani(n))

compareCGandSESOP(nlp, n=n)

;