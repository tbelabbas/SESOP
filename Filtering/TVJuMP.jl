export TVJuMP

# returns a JuMP model with TV regularization for images of shape X, weight λ and smoothing ϵ

function TVJuMP(X, λ; ϵ :: Float64=0.0, iso :: Bool=true, kwargs...)

    n, m  = size(X)     #image size

    nlp = Model()

    x0 = spzeros(Float64, m, n)

    @variable(nlp, x[i=1:m, j=1:n], start = x0[i, j])


    if iso  #  Isotropic
        @NLobjective(
            nlp,
            Min,
            λ * (
                (sum(
                    sum( sqrt((x[i, j] - x[i+1, j])^2 +
                              (x[i, j] - x[i, j + 1])^2 + ϵ^2) for j = 1:n-1)
                    for i = 1:m-1)
                 + sum(sqrt((x[i, n] - x[i+1, n])^2  + ϵ^2) for i = 1:m-1)
                 + sum(sqrt((x[m, j] - x[m, j+1])^2  + ϵ^2) for j = 1:n-1)
                 )
                -(n * m - 1) * ϵ
            )
        )
    else   # non isotropic
        @NLobjective(
            nlp,
            Min,
            λ*sum(
                sum(
                    (abs(x[i, j] - x[i+1, j]) + abs(x[i, j] - x[i, j+1])) for i = 1:m-1
                ) for j = 1:n-1
            ) + sum(abs(x[i, n] - x[i+1, n]) for i = 1:m-1)+sum(abs(x[m, j] - x[m, j+1]) for j = 1:n-1)
        )
    end#if
    
    return nlp
    
end#function

