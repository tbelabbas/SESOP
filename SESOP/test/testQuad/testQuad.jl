import cmpt_lbfgs


println("\n valeur propre < : 1, valeur propre > : 100\n")

Λ = [1.0; 5.0; 15.0; 100.0; 20.0; 100.0]
M=rand(6,6)
O,R = qr(M)
Q = O*diagm(Λ)*O'

Q = 0.5*(Q+Q')

c = rand(1,6)

cmpt_lbfgs(Q,c,zeros(length(c)))


println("\n valeur propre < : 5, valeur propre > : 1000\n")
Λ = [200.0; 15.0; 5.0; 100.0; 500.0; 1000.0]
M=rand(6,6)
O,R = qr(M)
Q = O*diagm(Λ)*O'

Q = 0.5*(Q+Q')

c = rand(1,6)

cmpt_lbfgs(Q,c,zeros(length(c)))


# La matrice non définie positive
println("\n valeurs propres -5 et 100\n")
Λ = [100.0; -5.0; -5.0; 100.0; 100.0; 100.0]
M=rand(6,6)
O,R = qr(M)
Q = O*diagm(Λ)*O'

Q = 0.5*(Q+Q')

c = rand(1,6)

cmpt_lbfgs(Q,c,zeros(length(c)))

;
