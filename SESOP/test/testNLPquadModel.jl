using LinearAlgebra
using test

include("../src/NLPQuad.jl")

function objFunc(Q :: Matrix, c :: AbstractVector, x :: AbstractVector)
    return (0.5 * x' * Q * x + c' * x)'
end

function gradFunc(Q :: Matrix, c :: AbstractVector, x :: AbstractVector)
    return Q * x + c
end


#Λ = [1.0; 5.0; 5.0; 100.0; 100.0; 100.0]
#Λ = [100.0; 5.0; 5.0; 100.0; 100.0; 100.0]
Λ = [100.0; 5.0; 30.0; 55.0; 5.0; 200.0; 150.0; 80.0; 100.0]
n = length(Λ)
M=rand(n,n)
O,R = qr(M)
Q = O*diagm(Λ)*O'

Q = 0.5*(Q+Q')

c = rand(n)

nlp = NLPQuad(Q,c)

x = rand(n)

@test obj(nlp, x) ≈ objFunc(Q, c, x) atol=1e-15
@test grad(nlp, x) ≈ gradFunc(Q, c, x) atol=1e-15

