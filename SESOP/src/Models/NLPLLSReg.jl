import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod!, hess

export obj, grad, grad!, objgrad!, hess
export NLPQuad

using LinearAlgebra

"""
"""
mutable struct NLPQuad <: AbstractNLPModel
    
    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta     

    b :: AbstractVector
    A :: Matrix

end

function NLPQuad( A  :: Matrix,
                  b  :: AbstractVector,
                  x  :: AbstractVector,)
    
    nr, nc = size(A)
    meta = NLPModelMeta(nc, x0=x)             

    return NLPQuad(Counters(), meta, b, A)
end

function NLPQuad( A  :: Matrix,
                  b  :: AbstractVector)
    
    nr, nc = size(A)
    meta = NLPModelMeta(nc, x0=zeros(nc))             

    return NLPQuad(Counters(), meta, b, A)
end

import NLPModels: reset!

function reset!(QD :: NLPQuad)
    reset!(QD.counters)
end

"""
Fonction objectif 
  f(x) = 0.5 * xᵀQx + cx
"""
function obj(QD :: NLPQuad, x :: AbstractVector)
  increment!(QD, :neval_obj)
  return 0.5 * x' * QD.A * x + QD.b' * x
end

"""
Gradient de la fonction f
    ∇f(x) :=  Qx + c
"""
function grad(QD :: NLPQuad, x :: AbstractVector)
  increment!(QD, :neval_grad)
  gx = similar(QD.meta.x0)
  return grad!(QD, x, gx)
end

"""
Gradient de la fonction f, avec gradient préalloué
"""
function grad!(QD :: NLPQuad, x :: AbstractVector, g :: AbstractVector)
  increment!(QD, :neval_grad)

  g = QD.A * x +  QD.b
  return g
end

function objgrad!(QD :: NLPQuad, x :: AbstractVector, g :: AbstractVector)

  return obj(QD, x), grad!(QD, x, g)
end

 """
 Hessien
 """
 function hess(QD :: NLPQuad, t :: AbstractFloat)
   NLPModels.increment!(QD, :neval_hess)
   return QD.A' * QD.A
 end
