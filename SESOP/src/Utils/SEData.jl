export SEData

using DataStructures

"""
Struct contenant les directions du sous-espace
"""
mutable struct SEData

    G  :: AbstractVector
    Gx :: Pair{Int, Matrix}  #previous gradients matrix
    P  :: Pair{Int, Matrix}  #previous directions matrix
    N  :: Pair{Int, Matrix}  #Nemirovski directions matrix
    weights :: AbstractVector

    AD :: Pair{Int, Matrix}  #
    ADt:: Matrix  #
    Ax ::  Matrix   #
    Axt:: Matrix

    dict_se :: OrderedDict{Int, AbstractArray}

end

function SEData(x0 :: AbstractVector)

  a = zeros(length(x0),0)
  m = 0=>a
  return SEData(x0, m, m, m, [1.0], m, a, a, a, OrderedDict{Int, AbstractArray}())

end

function update_subspace!(M :: Pair{Int, Matrix}, dir, mem)

  nr, ncD = size(M.second)
  matrix = M.second


  nc = 0

  if isa(dir, AbstractVector)
    nc = 1
  elseif isa(dir, AbstractArray)
    nr, nc = size(dir)
  end

  if ncD == 0
    M = (M.first+nc) => [M.second dir]
  else
    for i = 1 : nc
        index = ( M.first % mem) + 1

        if ncD < index
          matrix = [M.second dir[:, i]]
        else
          matrix[:, index] = dir[:, i]
        end

    end
    M = (M.first+nc) => matrix
  end

  return M
end

