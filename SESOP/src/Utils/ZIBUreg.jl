import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod!, hess, hess_structure!, hess_coord!
import SolverTools: redirect!

export obj, grad, grad!, objgrad!, hess, redirect!, hprod!, hprod
export ZIBUreg

include("SEData.jl")

"""
Soit une régularisation ψ(x)

"""
mutable struct ZIBUreg <: AbstractNLPModel
    
    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta     
    
    w  :: Float64
    s  :: Float64
  end

function ZIBUreg( x :: AbstractVector ;
                  w  :: Float64 = 1e-2,
                  s  :: Float64 = 0.0002)

    meta = NLPModelMeta(length(x), x0=x)             

    return ZIBUreg(Counters(), meta, w, s)
end

import NLPModels: reset!

function reset!(se :: ZIBUreg)
reset!(se.counters)
end

"""
Fonction ϕ(z)
"""
function ϕ(s :: Float64, z :: Float64)
return abs(z) - s * log(1 + (abs(z) / s))
end

function ϕgrad(s :: Float64, z :: Float64)
return (abs(z) * sign(z)) / (s + abs(z))
end

function ϕhess(s :: Float64, z :: Float64)
return s / (s + abs(z))^2
end

"""
Fonction objectif 
f(z) := ||Az - b||² + Sum(w * ϕ(z))
"""
function obj(ZR :: ZIBUreg, z :: AbstractVector)
    increment!(ZR, :neval_obj)

    n = length(z)
    b = 0.0

    for i = 1 : n
    b += ZR.w * ϕ(ZR.s, z[i])
    end

    return b
end

"""
Gradient de la fonction 
f'(z) := Sum(w * ϕ'(z))
"""
function grad(ZR :: ZIBUreg, t :: AbstractVector)
    g = similar(ZR.meta.x0)
    return grad!(ZR, t, g)
end

"""
Gradient de la fonction ϕ, avec gradient préalloué  
"""
function grad!(ZR :: ZIBUreg, z :: AbstractVector, g :: AbstractVector)
    increment!(ZR, :neval_grad)

    n = length(z)

    for i = 1 : n
        g[i] = ZR.w * ϕgrad(ZR.s, z[i])
    end

    return g
end

function objgrad!(ZR :: ZIBUreg, t :: AbstractVector, g :: AbstractVector)
    return obj(ZR, t), grad!(ZR, t, g)
end

function hess(ZR :: ZIBUreg, z :: AbstractVector)
    increment!(ZR, :neval_grad)

    Hx = zeros(n,n)

    n = length(z)
    for i = 1 : n
        Hx[i,i] += ZR.w * ϕhess(ZR.s, z[i])
    end

    return 0.5*(Hx+Hx')
end

function hprod(ZR :: ZIBUreg, α :: AbstractVector, 
               v :: AbstractVector; obj_weight :: Float64 = 1.0,
               y :: AbstractVector = Float64[])
               Hv = zeros(α)

    return hprod!(ZR, α, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(ZR :: ZIBUreg, α :: AbstractVector, 
                v :: AbstractVector, Hv :: AbstractVector; 
                obj_weight = 1.0, y :: AbstractVector = Float64[])

    increment!(ZR, :neval_hprod)

    Hv = hprod!(ZR.nlp, α, v, Hv, obj_weight = obj_weight)
    Hv2 = similar(Hv)

    n = length(α)
    for i = 1 : n
    Hv2[n] = ZR.w * ϕhess(ZR.s, α[i])
    end

    Hv += Hv2

    return Hv
end
