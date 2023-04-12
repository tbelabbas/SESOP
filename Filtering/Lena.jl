# using Pkg
# Pkg.activate(".")

using JuMP, NLPModels, NLPModelsJuMP
using JSOSolvers
using SparseArrays

using LinearAlgebra
using LinearOperators

using ToeplitzMatrices
using FFTW

using FileIO
using Images
using ImageView
using TestImages

using Stopping

using LBFGSB

include("LLS_Op.jl")
include("TVJuMP.jl")
include("wrapper.jl")
include("SumModelG1.jl")
include("../Recherche/repo/Revised SESOP/src/SESOP_pastiche.jl")

TV = TVJuMP


ImageFiles = true    # Production d'images png
ImageDisplay = false  # Affichage des images



##################################################
# Un cercle et un carré.                         #
# X est la vérité terrain, img l'image brouillée.#
# Ac et Ar les opérateurs de brouillage.         #
##################################################
include("load_data.jl")

# USE ONLY SQUARE IMAGES FOR NOW

#nr=200;nc=200   # taille
blur = 5        # degré de brouillage
noise = 0.6      # niveau de bruit
img, Ac, Ar, X = test_image("cameraman", noise=noise, blur = blur)
#################################


#################################
######
# Image avec texte illisible
# Similaire à ci haut, mais pas de vérité terrain
######
#using MAT
#vars = matread("challenge2.mat")
#img = vars["B"]
#Ac = vars["Ac"]
#Ar = vars["Ar"]
#################################


nr, nc = size(img)

################################
# Afficher les images sous étude
######
#img = clamp.(img,0,Inf)
cmin, = findmin(img)
cmax, = findmax(img)


# Fermer toute fenêtre restantes d'une précédente exécution
ImageView.closeall()

#save("GroundTruth.png",  X)
#imshow(X, name = "Vérité terrain")

Simg = map(scaleminmax(cmin, cmax), img)
ImageFiles && save("Results/BlurredImg.png",  Simg)
ImageDisplay && imshow(Simg, name = "Rescaled in [0,1]")
sleep(0.001)


Cimg = map(clamp01nan, img)
ImageFiles && save("Results/BlurredImgC.png",  Cimg)
ImageDisplay && imshow(Cimg, name = "Clamped in [0,1]")
sleep(0.001)


###############################
# TV
#######
λ = 1.0
ϵ = 0.01
m = TV(img, λ, ϵ = ϵ)
m2 = MathOptNLPModel(m)


##############################
# Adéquation aux données
#######
Vimg = img[:]   # Représentation de l'image sous forme vecteur

MA(X) = V=Ac*X*Ar'
VA(x) = v=vec(MA(reshape(x,nr,nc)))
MAt(X) = V=Ac'*X*Ar
VAt(x) = v=vec(MAt(reshape(x,nr,nc)))

dim = nc*nr

# Encapsulation des fonctions pour l'efficacité de LinearOperators avec mul5
function muVA!(res, v, α, β::T) where T
  if β == zero(T)
    res .= α .* VA(v)
  else
    res .= α .* VA(v) .+ β .* res
  end
end

function muVAt!(res, w, α, β::T) where T
  if β == zero(T)
    res .= α .* VAt(w)
  else
    res .= α .* VAt(w) .+ β .* res
  end
end

Aop = LinearOperator(Float64, dim, dim, false, false, muVA!, nothing, muVAt!)




let Aop=Aop, Vimg=Vimg, m2=m2
    for i in 50:50
        m1 = LLS_Op(Aop, Vimg, zeros(length(Vimg)))

        ##############################
        # Somme des deux modèles
        #######
        ub = ones(size(Vimg))
        lb = 0*copy(Vimg)
        nlp = SumModel(m1, m2, zeros(size(Vimg)), lb, ub)


        nlp_at_x = NLPAtX(nlp.meta.x0)
        stp = NLPStopping(nlp, nlp_at_x)

        ##############################
        # conditions d'arrêt
        ######
        stp.meta.optimality_check = optim_check_bounded
        stp.meta.max_iter = i
        stp.meta.atol = 1e-2
        stp.meta.max_time = 5000.0


        ############################
        # Deux exécutions avec et sans bornes
        #######

        # println("\n\n---------- With bounds")
        # println("\n\n L-BFGS-B_$i")


        mem = 5
        NLPlbfgsbS = L_BFGS_B(nr*nc, mem)

        # stp = NLPlbfgsbS(nlp, nlp.meta.x0, stp = stp, m=mem);
        # @show getStatus(stp)
        # @show stp.meta.nb_of_stop
        # @show sum_counters(stp.pb)

        # Xs = reshape(stp.current_state.x,nr,nc)
        # Bimg = map(clamp01nan, Xs)
        # name = "Results/testLbfgs_bounded_" * "$i.png"
        # ImageFiles && save(name,  Bimg)
        # ImageDisplay && imshow(Bimg, name = "L-BFGS-B deblurred")

        #############################################################
        println("\n\n---------- Without bounds")
        println("\n\n SESOP_$i")

        m1 = LLS_Op(Aop, Vimg, zeros(length(Vimg)))
        reset!(m2)

        nlp = SumModel(m1, m2, zeros(length(Vimg)))

        reinit!(stp)
        stp.pb = nlp
        stp.meta.optimality_check = unconstrained_check

        stp, iter, x, fx = SESOP_pastiche(nlp;stp=stp);
        @show getStatus(stp)
        @show stp.meta.nb_of_stop
        @show sum_counters(stp.pb)

        XsU = reshape(stp.current_state.x,nr,nc)
        Uimg = map(clamp01nan, XsU)
        name = "Results/testSESOP_no_prc_" * "$i.png"
        ImageFiles && save(name, Uimg)
        ImageDisplay && imshow(Uimg, name = "SESOP without bounds")
        sleep(0.001)

        #############################################################
        println("\n\n---------- Without bounds")
        println("\n\n L-BFGS-U_$i")


        m1 = LLS_Op(Aop, Vimg, zeros(length(Vimg)))
        reset!(m2)

        ub = Inf*ones(size(Vimg))
        lb = -Inf*ones(size(Vimg))
        nlp = SumModel(m1, m2, zeros(size(Vimg)), lb, ub)
        @show neval_obj(nlp), neval_grad(nlp)

        reinit!(stp)
        stp.pb = nlp
        stp.meta.optimality_check = unconstrained_check

        stp = NLPlbfgsbS(nlp, nlp.meta.x0, stp = stp, m=mem);
        @show getStatus(stp)
        @show stp.meta.nb_of_stop
        @show sum_counters(stp.pb)

        XsU = reshape(stp.current_state.x,nr,nc)
        Uimg = map(clamp01nan, XsU)
        name = "Results/testLbfgs_unbounded_" * "$i.png"
        ImageFiles && save(name,  Uimg)
        ImageDisplay && imshow(Uimg, name = "L-BFGS without bounds")

    end
end

;
