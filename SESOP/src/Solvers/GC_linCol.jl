using SolverTools
using LinearAlgebra

function GC_lin(Q,c,x₀;  tol :: Real = 1e-8, MaxIter = length(x₀) + 100, method = :GC )
    # résout le système linéaire Q*x+c=0 par le gradient conjugué linéaire.

    # Entrée:
    # une matrice Q symétrique définie positive, un vecteur colonne c
    # x₀, un point initial
    #
    # Sortie:
    #
    #   x : solution optimale
    @assert isposdef(Q)

    ngc=0;
    x = x₀
    Qx = Q*x
    β = 0.0
    d = 0.0*x₀
    g = Qx + c
    
    @info log_header([:iter, :residu, :step], [Int, Float64, Float64],
                     hdr_override=Dict( :residu=>"‖∇q‖"))
    
    @info log_row(Any[ngc, norm(g)])

    iter = 0
    while  (norm(g) > tol) && (ngc < MaxIter)
        ngc=ngc+1
        d = -g + β*d
        Qd = Q*d
        dQd = d'*Qd

        θ = (-g'*d / (dQd))         #theta=-nablaq*p/(pQp)
        x = x + θ*d
        gold = copy(g)
        g = g + θ*Qd;

        β = ((g'*g) / (gold'*gold))   #Beta = pQ*nablaq'/pQp
        if !(method == :GC) β = 0.0 end  # pour tester le gradient simpliste.
        @info log_row(Any[ngc, norm(g), θ])

        iter +=1
    end

    return iter, x
end



# Matrice aléatoire avec 3 paquets de valeurs propres distinctes de valeurs 1, 5 et 100
# La théorie nous informe que le GC convergera en 3 itérations.
# println("\n valeurs propresL 1, 5 et 100\n")
# Λ = [1.0; 5.0; 5.0; 100.0; 100.0; 100.0]
# M=rand(6,6)
# O,R = qr(M)
# Q = O*diagm(Λ)*O'

# Q = 0.5*(Q+Q')

# c = rand(6)

# GC_lin(Q,c,zeros(length(c)))

# # Matrice aléatoire avec 2 paquets de valeurs propres distinctes de valeurs  5 et 100
# # La théorie nous informe que le GC convergera en 2 itérations.
# println("\n valeurs propresL 5 et 100\n")
# Λ = [100.0; 5.0; 5.0; 100.0; 100.0; 100.0]
# M=rand(6,6)
# O,R = qr(M)
# Q = O*diagm(Λ)*O'

# Q = 0.5*(Q+Q')

# c = rand(6)

# GC_lin(Q,c,zeros(length(c)))

# # Matrice aléatoire avec 2 paquets de valeurs propres distinctes de valeurs  -5 et 100
# # La matrice n'est pas définie positive, on teste ainsi que la fonction échoue..
# println("\n valeurs propresL 55 et 100\n")
# Λ = [100.0; -5.0; -5.0; 100.0; 100.0; 100.0]
# M=rand(6,6)
# O,R = qr(M)
# Q = O*diagm(Λ)*O'

# Q = 0.5*(Q+Q')

# c = rand(6)

# GC_lin(Q,c,zeros(length(c)))

;
