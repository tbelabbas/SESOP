export LLS_Op


"""
    lls = LLS_Op(A, b)

    Creates a simple lls model ||Ax-b||²  .
"""
mutable struct LLS_Op{T, S} <: AbstractNLPModel{T, S}
    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta{T, S}

    A :: AbstractLinearOperator
    b :: S
end

function LLS_Op(A :: AbstractLinearOperator,
                b :: S,
                x0 :: S
                ) where {T, S}
    n = length(x0)
    meta = NLPModelMeta(n, x0=x0)
    return LLS_Op(Counters(), meta, A, b)
end



import NLPModels: reset!

function reset!(llsm :: LLS_Op)
    reset!(llsm.counters)

    return llsm
end


import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod!, hess, hess_structure!, hess_coord!

function obj(llsm :: LLS_Op, x :: AbstractVector)
    increment!(llsm, :neval_obj)
    residual = llsm.A*x - llsm.b
    valobj = 1/2 * (residual'*residual)
    return valobj
end

function grad(llsm :: LLS_Op, x :: AbstractVector)
  g = zeros(llsm.meta.nvar)
  return grad!(llsm, x, g)
end

function grad!(llsm :: LLS_Op, x :: AbstractVector, g :: AbstractVector)
    increment!(llsm, :neval_grad)
    residual = llsm.A*x - llsm.b
    g = (llsm.A'*residual)

    return g
end

function objgrad!(llsm :: LLS_Op, x :: AbstractVector, g :: AbstractVector)

  return obj(llsm,x), grad!(llsm, x, g)
end

function NLPModels.hprod!(
                           llsm :: LLS_Op,
                           x::AbstractVector,
                           v::AbstractVector,
                           Hv::AbstractVector;
                           obj_weight = one(T),
                         ) where {T, S}
    increment!(llsm, :neval_hprod)
    Av = llsm.A*v
    Hv = transpose(llsm.A)*Av

    Hv .*= obj_weight
    return Hv
end
