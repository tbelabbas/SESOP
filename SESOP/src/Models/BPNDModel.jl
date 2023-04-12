import NLPModels: obj, increment!, grad!, grad, objgrad!, hprod!, hess, hess_structure!, hess_coord!
import SolverTools: redirect!

export obj, grad, grad!, objgrad!, hess, redirect!, hprod!, hprod
export BPNDModel

"""
Soit une fonction f contenue dans un NLPModel. 

  f(z) = BPNDModel(nlp, x, D)

"""
mutable struct BPNDModel <: AbstractNLPModel
    
    counters :: Counters      # Evaluation counters.
    meta :: NLPModelMeta     
    nlp :: AbstractNLPModel   # modèle contenant f(x) 


    w  :: Float64
    s  :: Float64
  end

function BPNDModel( A  :: LinearOperator{Float64},
                    b  :: AbstractVector;
                    w  :: Float64 = 1e-2,
                    s  :: Float64 = 0.0002)
    
    N, K = size(A)

    nlp = LLSModel(A, b)
    meta = NLPModelMeta(N, x0=zeros(N))             

    return BPNDModel(Counters(), meta, nlp, w, s)
end

import NLPModels: reset!

function reset!(se :: BPNDModel)
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
function obj(BP :: BPNDModel, z :: AbstractVector)
  increment!(BP, :neval_obj)

  n = length(z)
  b = 0.0

  for i = 1 : n
    b += BP.w * ϕ(BP.s, z[i])
  end

  return obj(BP.nlp, z) + b
end

"""
Gradient de la fonction 
   f'(z) := ∇LLS + Sum(w * ϕ'(z))
"""
function grad(BP :: BPNDModel, t :: AbstractVector)
  increment!(BP, :neval_grad)
  g = similar(BP.meta.x0)
  return grad!(BP, t, g)
end

"""
Gradient de la fonction ϕ, avec gradient préalloué
    ∇ϕ(t) := Dᵀ * ∇f(x_k + D * t )    
"""
function grad!(BP :: BPNDModel, z :: AbstractVector, g :: AbstractVector)
  increment!(BP, :neval_grad)

  increment!(BP, :neval_obj)

  n = length(z)

  for i = 1 : n
    g[i] = BP.w * ϕgrad(BP.s, z[i])
  end

  g += grad(BP.nlp, z)

  return g
end

function objgrad!(BP :: BPNDModel, t :: AbstractVector, g :: AbstractVector)

  return obj(BP, t), grad!(BP, t, g)
end

function hess(BP :: BPNDModel, z :: AbstractVector)
    increment!(BP, :neval_grad)
    
    Hx = hess(BP.nlp, z)

    n = length(z)
    for i = 1 : n
        Hx[n,n] += BP.w * ϕhess(BP.s, z[i])
    end

    return 0.5*(Hx+Hx')
end

function hprod(m :: BPNDModel, α :: AbstractVector, 
               v :: AbstractVector; obj_weight :: Float64 = 1.0,
               y :: AbstractVector = Float64[])
  Hv = zeros(α)
  return hprod!(m, α, v, Hv, obj_weight=obj_weight, y=y)
end

function hprod!(m :: BPNDModel, α :: AbstractVector, 
                v :: AbstractVector, Hv :: AbstractVector; 
                obj_weight = 1.0, y :: AbstractVector = Float64[])

  increment!(m, :neval_hprod)

  Hv = hprod!(m.nlp, α, v, Hv, obj_weight = obj_weight)
  Hv2 = similar(Hv)

  n = length(α)
  for i = 1 : n
      Hv2[n] = m.w * ϕhess(m.s, α[i])
  end

  Hv += Hv2
  
  return Hv
end
