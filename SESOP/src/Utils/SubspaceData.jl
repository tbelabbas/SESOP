export SubspaceData
using DataStructures
include("LimitedMemoryConfig.jl")

"""
    Struct contenant les directions du sous-espace
"""
mutable struct SubspaceData{T}

    Gradients :: DirectionDict{T} # Previous gradients matrix
    PreviousDirs :: DirectionDict{T} # Previous directions matrix
    NemirovskiDirs :: DirectionDict{T} # Nemirovski directions matrix

    Normes :: AbstractArray{T}
    Objectives :: AbstractArray{T}

    # Subspace :: Matrix{T} # Allocate memory for full subspace -- TODO

    memory :: LMC # Keeps track of the total memory of subspace
    dict_se :: OrderedDict{Int, AbstractArray} # Keeps info on subspace opt
end

"""
    SubspaceData constructor.
        Parameter m is the number of directions we want to keep
                    for each category of directions. If only an
                    overall int is provided, a default LMC is used.
        Parameter n is the dimension of a direction.
"""
function SubspaceData(T::DataType,
                      n :: Int;
                      m :: Union{Int, Vector{Int}, LMC} = LMC₂₂₀
                     )
    if !isa(m, LMC)
        m = LMC(m)
    end
    return SubspaceData{T}(DirectionDict(T; n = n, m = m.nbGradient),
                           DirectionDict(T; n = n, m = m.nbDirection),
                           DirectionDict(T; n = n, m = m.nbNemirovski),
                           [-1], [-1],
                           m, OrderedDict{Int, AbstractArray}()
                          )
end

SubspaceData(n::Int; kwargs...) = SubspaceData(Float64, n; kwargs...)

"""
    SubspaceData getters.
        For direction : calls getDirections on specified dictionnary.
        For memory : get total memory allocated (Full subspace dimension).
"""
# Gradients
function getGradients(seData :: SubspaceData)
    return getDirections(seData.Gradients)
end
# PreviousDirs
function getPreviousDirs(seData :: SubspaceData)
    return getDirections(seData.PreviousDirs)
end
# NemirovskiDirs
function getNemirovskiDirs(seData :: SubspaceData)
    return getDirections(seData.NemirovskiDirs)
end
# Memory
function getAllocMemory(seData :: SubspaceData)
    return size(seData.Gradients.matrix, 2) +
           size(seData.PreviousDirs.matrix, 2) +
           size(seData.NemirovskiDirs.matrix, 2)
end

"""
    Full subspace getter
"""
function getSubspace(se :: SubspaceData)
    # println("getGradients")
    # display(getGradients(se))
    # println("getPreviousDirs")
    # display(getPreviousDirs(se))
    # println("getNemirovskiDirs")
    # display(getNemirovskiDirs(se))
    return [getGradients(se) getPreviousDirs(se) getNemirovskiDirs(se)]
end

"""
    SubspaceData setters.
        Calls update! on specified dictionnary.
"""
# Gradients
function updateGradients!(seData :: SubspaceData, x)
    return update!(seData.Gradients, x)
end
# PreviousDirs
function updatePreviousDirs!(seData :: SubspaceData, x)
    return update!(seData.PreviousDirs, x )
end
# NemirovskiDirs
function updateNemirovskiDirs!(seData :: SubspaceData, x)
    return update!(seData.NemirovskiDirs, x)
end

function updateNormes!(seData :: SubspaceData, x)
    append!(seData.Normes, x)
end

function updateObjectives!(seData :: SubspaceData, x)
    append!(seData.Objectives, x)
end