export SumModel


"""
    comb = SumModel(m1, m2)

Creates a combined model m1 + m2.
"""
mutable struct SumModel{T, S} <: AbstractNLPModel{T, S}
    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta{T, S}

    m1 :: AbstractNLPModel{T, S}
    m2 :: AbstractNLPModel{T, S}

    # pour graphiques de convergence
    objGraph :: AbstractVector
    gradGraph :: AbstractVector
end

function SumModel(m1 :: AbstractNLPModel{T, S},
                  m2 :: AbstractNLPModel{T, S},
                  x0 :: S,
                  lb :: S,
                  ub :: S#,
                  #x  :: AbstractVector # ground truth
                  ) where {T, S}
    n = length(x0)
    meta = NLPModelMeta(n, x0=x0, lvar = lb, uvar = ub)
    return SumModel(Counters(), meta, m1, m2, zeros(0),zeros(0)) #, x, zeros(0),zeros(0),Inf,zeros(0),  zeros(0))
end

function SumModel(m1 :: AbstractNLPModel{T, S},
                  m2 :: AbstractNLPModel{T, S},
                  x0 :: S#,
                  #x  :: AbstractVector # ground truth
                  ) where {T, S}
    n = length(x0)
    meta = NLPModelMeta(n, x0=x0)
    return SumModel(Counters(), meta, m1, m2, zeros(0),zeros(0)) #, x, zeros(0),zeros(0),Inf,zeros(0), zeros(0))
end



import NLPModels: reset!

function reset!(sm :: SumModel)
    reset!(sm.counters)
    reset!(sm.m1)
    reset!(sm.m2)

    #sm.graph = zeros(0)
    sm.gradGraph = zeros(0)
    sm.objGraph = zeros(0)
    #sm.best = zeros(0)
    #sm.bestVal = Inf
    return sm
end


import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod!, hess, hess_structure!, hess_coord!

function obj(sm :: SumModel, x :: AbstractVector)
    increment!(sm, :neval_obj)
    #CurrVal = norm(x-sm.x)
    #append!(sm.graph, CurrVal)
    #if CurrVal < sm.bestVal
    #    sm.bestVal = CurrVal
    #    sm.best = copy(x)
    #end
    valobj = obj(sm.m1,x) + obj(sm.m2,x)
    append!(sm.objGraph, valobj)

    return valobj
end

function grad(sm :: SumModel, x :: AbstractVector)
  g = zeros(sm.meta.nvar)
  return grad!(sm, x, g)
end

proj(ub :: Vector, lb :: Vector, x :: Vector) = max.(min.(x,ub),lb)

gradproj(ub :: Vector, lb :: Vector, g::Vector, x :: Vector) =  x - proj(ub, lb, x-g)


function grad!(sm :: SumModel, x :: AbstractVector, g :: AbstractVector)
    increment!(sm, :neval_grad)
    g = grad!(sm.m1, x, g)
    g2 = similar(g)
    g2 = grad!(sm.m2, x, g2)
    append!(sm.gradGraph, norm(gradproj(sm.meta.uvar,sm.meta.lvar,x,g+g2),Inf))
    return g + g2
end

function objgrad!(sm :: SumModel, x :: AbstractVector, g :: AbstractVector)

  return obj(sm,x), grad!(sm, x, g)
end

function hess(sm :: SumModel, x :: AbstractVector; obj_weight ::
              Float64 = 1.0, y :: AbstractVector = Float64[])
    increment!(sm, :neval_hess)



    return hess(sm.m1,x) + hess(sm.m2,x)
end


function hess_coord!(sm :: SumModel, x :: AbstractVector, vals :: AbstractVector; obj_weight ::
                    Float64 = 1.0, y :: AbstractVector = Float64[])
    Hx = hess(sm, x, obj_weight=obj_weight)
    vals = findnz(Hx)
    return vals
end



function hess_coord(sm :: SumModel, x :: AbstractVector; obj_weight ::
                    Float64 = 1.0, y :: AbstractVector = Float64[])
    Hx = hess(sm, x, obj_weight=obj_weight)
    return findnz(Hx)
end

function hprod(sm :: SumModel, x :: AbstractVector, v ::
               AbstractVector; obj_weight :: Float64 = 1.0)
    Hv = zeros(length(x))
    return hprod!(sm, x, v, Hv, obj_weight=obj_weight)
end

function hprod!(sm :: SumModel, x :: AbstractVector, v ::
                AbstractVector, Hv :: AbstractVector; obj_weight =
                1.0)
    increment!(sm, :neval_hprod)

    Hv = hprod!(sm.m1, x, v, Hv, obj_weight = obj_weight)
    Hv2 = similar(Hv)
    Hv2 = hprod!(sm.m2, x, v, Hv2, obj_weight = obj_weight)
    Hv += Hv2


    return obj_weight * Hv
end


function NLPModels.hess_structure!(nlp :: SumModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  n = nlp.meta.nvar
  Ind = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(Ind, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(Ind, 2)
  return rows, cols
end
