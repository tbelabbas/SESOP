import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod, hprod!, hess
import SolverTools: redirect!

export obj, grad, grad!, objgrad!, hess, redirect!, hprod, hprod!
export SEModel, SEModel_pcc


"""
    Soit une fonction f contenue dans un NLPModel.

    ϕ = SEModel(nlp, x, D)

    Où x est le point courant et D un ensemble de directions
    alors :

    ϕ(t) := f(x + Dt)

    Avec
    x ∈ Rⁿ
    D ∈
    t ∈
"""
mutable struct SEModel{T, S} <: AbstractNLPModel{T, S}

    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta{T, S}
    nlp :: AbstractNLPModel   # modèle contenant f(x)

    x :: AbstractVector
    D :: AbstractArray
    ∇f :: AbstractVector
end

function SEModel( ms_nlp :: AbstractNLPModel,
                    x    :: AbstractVector,
                    D    :: AbstractArray,
                    ∇f   :: AbstractVector
                )
        m = size(D, 2)
        meta = NLPModelMeta(m, x0=zeros(m))

        return SEModel(Counters(), meta, ms_nlp, x, D, ∇f)
end

import NLPModels: reset!

function reset!(se :: SEModel)
        reset!(se.counters)
end

"""
    Fonction objectif ϕ
    ϕ(t) := f(x_k + D * t)
"""
function obj(SE :: SEModel, t :: AbstractVector)
    increment!(SE, :neval_obj)
    return obj(SE.nlp, SE.x + SE.D * t)
end

"""
    Gradient de la fonction ϕ
        ∇ϕ(t) :=  ∇f(x_k + D * t)ᵀ D
"""
function grad(SE :: SEModel, t :: AbstractVector)
    increment!(SE, :neval_grad)
    g = similar(SE.meta.x0)
    return grad!(SE, t, g)
end

"""
    Gradient de la fonction ϕ, avec gradient préalloué
    ∇ϕ(t) := Dᵀ * ∇f(x_k + D * t )
"""
function grad!(SE :: SEModel, t, g)
    increment!(SE, :neval_grad)
    gx = zeros(length(SE.x))
    gx = grad!(SE.nlp, SE.x + SE.D * t, gx)

    #@show gx

    SE.∇f = gx

    g[:] .= SE.D'*gx
    return g
end

function objgrad!(SE :: SEModel, t :: AbstractVector, g :: AbstractVector)

    return obj(SE, t), grad!(SE, t, g)
end

function hess(SE :: SEModel, t :: AbstractVector;
                obj_weight :: Float64 = 1.0,
                y :: AbstractVector = Float64[])
    increment!(SE, :neval_hess)


    H = hess_op(SE.nlp, SE.x + (SE.D * t))
    HD = Matrix(H*SE.D)

    DHD = SE.D' * HD
    return 0.5*(DHD+DHD')
end


# function hess_coord(m :: SEModel, α :: AbstractVector;
#                         obj_weight :: Float64 = 1.0, y :: AbstractVector = Float64[])

#     Hα = hess(m, α, obj_weight=obj_weight)

#     return findnz(Hα)
# end

function hprod(m :: SEModel, α :: AbstractVector,
               v :: AbstractVector; obj_weight :: Float64 = 1.0)
    Hv = zeros(length(α))
    return hprod!(m, α, v, Hv, obj_weight=obj_weight)
end

function hprod!(m :: SEModel, α :: AbstractVector,
                vα :: AbstractVector, Hv :: AbstractVector;
                obj_weight = 1.0)

    increment!(m, :neval_hprod)

    x = m.x + m.D * α
    Hvx = hprod(m.nlp, x, m.D * vα)
    Hv[:] = obj_weight * m.D' * Hvx

    return Hv
end

"""
    Soit une fonction f contenue dans un NLPModel.

    ϕ = SEModel_pcc(nlp, x, D)

    Où x est le point courant et D un ensemble de directions
    alors :

    ϕ(t) := f(x + Dt)

    Avec
    x ∈ Rⁿ
    D ∈
    t ∈
"""
mutable struct SEModel_pcc{T, S} <: AbstractNLPModel{T, S}

    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta{T, S}
    nlp :: AbstractNLPModel   # modèle contenant f(x)

    x :: AbstractVector
    D :: AbstractArray
    ∇f :: AbstractVector
end

function SEModel_pcc( ms_nlp :: AbstractNLPModel,
                    x    :: AbstractVector,
                    D    :: AbstractArray,
                    ∇f   :: AbstractVector
                )
        m = size(D, 2)
        meta = NLPModelMeta(m, x0=zeros(m))

        return SEModel_pcc(Counters(), meta, ms_nlp, x, D, ∇f)
end

import NLPModels: reset!

function reset!(se :: SEModel_pcc)
        reset!(se.counters)
end

"""
    Fonction objectif ϕ
    ϕ(t) := f(x_k + D * t)
"""
function obj(SE :: SEModel_pcc, t :: AbstractVector)
    increment!(SE, :neval_obj)
    quad_r = SE.nlp.m1.residual + SE.nlp.m1.AD*t
    quad_obj = 1/2 * (quad_r'*quad_r)
    return quad_obj + obj(SE.nlp.m2, SE.x + SE.D * t)
end

"""
    Gradient de la fonction ϕ
        ∇ϕ(t) :=  ∇f(x_k + D * t)ᵀ D
"""
function grad(SE :: SEModel_pcc, t :: AbstractVector)
    increment!(SE, :neval_grad)
    g = similar(SE.meta.x0)
    return grad!(SE, t, g)
end

"""
    Gradient de la fonction ϕ, avec gradient préalloué
    ∇ϕ(t) := Dᵀ * ∇f(x_k + D * t )
"""
function grad!(SE :: SEModel_pcc, t, g)
    increment!(SE, :neval_grad)
    gx = zeros(length(SE.x))
    g_r = SE.nlp.m1.residual + SE.nlp.m1.AD*t
    g_quad = SE.nlp.m1.A' * g_r
    SE.∇f = g_quad + grad!(SE.nlp.m2, SE.x + SE.D * t, gx)

    g[:] .= (SE.D)'*SE.∇f
    return g
end

function objgrad!(SE :: SEModel_pcc, t :: AbstractVector, g :: AbstractVector)

    return obj(SE, t), grad!(SE, t, g)
end

function hess(SE :: SEModel_pcc, t :: AbstractVector;
                obj_weight :: Float64 = 1.0,
                y :: AbstractVector = Float64[])
    increment!(SE, :neval_hess)


    H = hess_op(SE.nlp, SE.x + (SE.D * t))
    HD = Matrix(H*SE.D)

    DHD = SE.D' * HD
    return 0.5*(DHD+DHD')
end


# function hess_coord(m :: SEModel_pcc, α :: AbstractVector;
#                         obj_weight :: Float64 = 1.0, y :: AbstractVector = Float64[])

#     Hα = hess(m, α, obj_weight=obj_weight)

#     return findnz(Hα)
# end

function hprod(m :: SEModel_pcc, α :: AbstractVector,
               v :: AbstractVector; obj_weight :: Float64 = 1.0)
    Hv = zeros(length(α))
    return hprod!(m, α, v, Hv, obj_weight=obj_weight)
end

function hprod!(m :: SEModel_pcc, α :: AbstractVector,
                vα :: AbstractVector, Hv :: AbstractVector;
                obj_weight = 1.0)

    increment!(m, :neval_hprod)

    x = m.x + m.D * α
    Hvx = hprod(m.nlp, x, m.D * vα)
    Hv[:] = obj_weight * m.D' * Hvx

    return Hv
end
