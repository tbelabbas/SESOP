using NLPModels
using LinearAlgebra

"""
    Exact solver for SESOP quadratic problem 
"""
function SEsolver(nlp :: AbstractNLPModel, x, D)

    #@show D
    #H = Matrix(hess(nlp, nlp.meta.x0))
    Q = transpose(D) * hess(nlp, x) * D
    c = transpose(grad(nlp, x)) * D 

    return - inv(transpose(Q)) * transpose(c)
end

function SEsolver2(nlp :: AbstractNLPModel, x, D)

    #@show D
    #H = Matrix(hess(nlp, nlp.meta.x0))
    H = hess(nlp, x)
    Q = transpose(D) * H * D
    c = - (transpose(nlp.data.c) * D + 0.5 * (transpose(x) * H * D))
    
    return transpose(c * inv(Q))
end

function SEsolver3(Q :: Matrix, c :: AbstractVector, D, x :: AbstractVector)

    @assert isposdef(Q)
    return pinv(D'*Q*D) * -(D'*c + D'*Q*x)
end
