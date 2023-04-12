

export LMC, LMC₂₂₀ , LMC₃₃₀ , LMC₂₂₂


################################################################################

"""
    Configure the subspace shape for SESOP.
        LMC stands for Limited-memory configuration.
"""
mutable struct LMC
    nbGradient::Int   # Number of previous gradients kept in the subspace
    nbDirection::Int  # Number of previous directions kept in the subspace
    nbNemirovski::Int # Number of Nemirovski directions kept in the subspace
end

# Some usable default configurations
LMC₂₂₀ = LMC(2, 2, 0)
LMC₃₃₀ = LMC(3, 3, 0)
LMC₂₂₂ = LMC(2, 2, 2)

"""
    Constructor for a subspace configuration according to m.
        m can be either an integer representing the total number of directions
        to keep or it can be a list of 3 integers representing the memory to
        allocate for each different type of direction.
"""
function LMC(m :: Union{Int, Vector{Int}})

    if isa(m, Vector{Int})
        if any(x->x<0, m)
            @warn "The input memory vector contains negatives. Using absolute
            value as allocated memory."
            m = abs.(m)
        end
        if length(m) != 3
            if length(m) == 2
                return LMC(m[1], m[2], 0)
            else
                m = sum(m)
                @warn "The input memory vector is not of length 3. Redistributing
                total memory ($m) amongst gradients and previous directions."
            end
        else
            return LMC(m[1], m[2], m[3])
        end

    elseif m < 0
        @warn "The input memory is negative. Using absolute value as allocated
        memory."
        m = abs(m)
    end

    return LMC(m - (m ÷ 2), m ÷ 2, 0)
end

"""
    Function to get total memory allocated from LMC.
"""
function getMemory(m :: LMC)
    return m.nbDirection + m.nbGradient + m.nbNemirovski
end

################################################################################

"""
    Construct a type detailing the configuration of a certain type of directions
    within a subspace.
"""
mutable struct DirectionDict{T}
    full   :: Bool  # Bool indicating if at least m directions have been stored
    index  :: Int   # Integer pointing to oldest direction kept in matrix
    matrix :: AbstractArray{T}    # Matrix of *m* or less directions. Each column
                               # is a direction
end

"""
    DirectionDict constructor.
        Parameter m is the number of directions we want to keep.
        Parameter n is the dimension of a direction.
"""
function DirectionDict(T :: DataType; n :: Int, m :: Int)
    return DirectionDict{T}(false, 0, zeros(n, m))
end

DirectionDict(m :: Int, n :: Int) = DirectionDict(Float64; m, n)


"""
    Returns a view of rightfully shaped matrix of directions.
"""
function getDirections(dirs :: DirectionDict)
    (;full, index, matrix) = dirs

    m = size(matrix, 2)

    # If this directions categories has no memory allocated, return
    if m == 0
        return matrix
    end

    # If nothing has been inserted into the matrix yet, return identity
    if index < 1
        return ones(n, m)
    else
        # If we have been through m directions at least once
        if full
            # Get the columns in order of oldest direction to newest
            indices = collect(index:m)
            append!(indices, collect(1:(index-1)))
            # Return a view of directions matrix in the obtained order
            return view(matrix, :, indices)
        else
            i = index-1
            if i < 1
                i = m
            end

            # If we haven't been through m directions at least once
            # return a view of a matrix size n by index - 1
            # (since the "oldest" direction slot pointed at by *index*
            # is empty)
            return view(matrix, :, collect(1:i))
        end
    end
end

"""
    Updates the direction matrix in a circular manner.
"""
function update!(dirs :: DirectionDict, x :: Vector)
    (;full, index, matrix) = dirs

    # Store the matrix size for later use
    m = size(matrix, 2)

    if m == 0
        return matrix
    end

    # Insert in position of oldest direction. If index is 0, then no insertion
    # has yet been done. Increment index to insert at the right column.
    if index == 0
        index += 1
        if index == m
            full = true
        end
    end

    matrix[:, index] .= x

    # Increment index to point at oldest direction. If we haven't been through
    # m directions at least once, index points at the next empty slot to fill.
    index += 1

    # If we now have been through m direction when we hadn't before,
    # update the full boolean.
    # if !full && index == m
    #     full = true
    # end
    # Clip index value
    if index > m
        full = true
        index = 1
    end

    # Update struct data
    dirs.full = full
    dirs.index = index
    dirs.matrix = matrix
end

################################################################################
