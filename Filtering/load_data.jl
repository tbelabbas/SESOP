function square_circle(m::Int ,n:: Int; noise::Real = 1.0, blur::Int = 18)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% function [B, Ac, Ar, X] = challenge1(m, n, noise, Blur)
#%
#% This function generates a true image X, a blurred
#% image B, and two blurring matrices Ac and Ar so that
#%   B = Ac * X * Ar' + random noise .
#% The noise has mean 0 and standard deviation "noise".
#%
#% Ac, Ar, B, and X are all m x n arrays.
#%
#% from Chapter 1 of the text by Hansen, Nagy, and O'Leary
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    X = zeros(m,n)

    # Generate square
    I = round(Int, m/5):round(Int, 3*m/5)
    J = round(Int, n/5):round(Int, 3*n/5)
    X[I,J] .= 0.5

    # Generate circle
    for i=1:m
        for j=1:n
            if (i-round(Int, 3*m/5))^2+(j-round(Int, 5*n/8))^2 < round(Int, max(m,n)/5)^2
                X[i,j] = 1.0
            end
        end
    end
    X = 1 .- X

    cmin = findmin(X)
    cmax = findmax(X)
    Simg = map(scaleminmax(cmin, cmax), X)
    save("Results/true.png",  Simg)

    @assert(blur < min(m,n//2))

    c = zeros(m)
    blurH = (blur+1)*blur/2

    c[1:blur] = collect(blur:-1:1)/blurH

    #Ac = matrixdepot("toeplitz",c)
    Ac = SymmetricToeplitz(c)
    c = zeros(n)
    c[1:blur] = collect(blur:-1:1)/blurH
    r = zeros(n)
    r[1:2*blur] = collect(blur:-.5:.5)/blurH
    #Ar = matrixdepot("toeplitz",c,r)
    Ar = Toeplitz(c,r)

    Bbrut = (Ac * X) * (Ar')
    B = Bbrut + (noise .* randn(m,n))

    return B, Ac, Ar, X
end

function test_image(name::String; noise::Real = 1.0, blur::Int = 18)
    img = Float32.(testimage(name))

    m, n = size(img)
    @assert(blur < min(m,n//2))

    c = zeros(m)
    blurH = (blur+1)*blur/2

    c[1:blur] = collect(blur:-1:1)/blurH

    #Ac = matrixdepot("toeplitz",c)
    Ac = SymmetricToeplitz(c)
    c = zeros(n)
    c[1:blur] = collect(blur:-1:1)/blurH
    r = zeros(n)
    r[1:2*blur] = collect(blur:-.5:.5)/blurH
    #Ar = matrixdepot("toeplitz",c,r)
    Ar = Toeplitz(c,r)

    Bbrut = (Ac * img) * (Ar')

    B = Bbrut + (noise .* randn(m,n))

    return B, Ac, Ar, img
end