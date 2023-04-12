using Stopping
using NLPModels
using ADNLPModels
using LBFGSB
using Test

include("wrapper.jl")

NLPlbfgsbS = L_BFGS_B(1024, 17)

#using CUTEstModel

function objFunc(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
x₀ = [0.0; 0.0];

display("Test interface to NLPModels and Stopping")
display("        first without bounds")

nlp = ADNLPModel(objFunc, x₀)
nlp_at_x = NLPAtX(nlp.meta.x0)
stp = NLPStopping(nlp, nlp_at_x)
stp.meta.optimality_check = optim_check_bounded
stp.meta.max_iter = 55
stp.meta.atol = 1e-7
stp.meta.max_eval = 300
stp.meta.max_time = 15.0


stp  = NLPlbfgsbS(nlp, nlp.meta.x0, stp = stp, m=5)

x = stp.current_state.x
f = stp.current_state.fx

@show stp.meta.nb_of_stop


@test abs(f) < 0.00001
@test abs(x[1]-1) < 0.00001
@test abs(x[2]-1) < 0.00001
@show x

reinit!(stp)
reset!(nlp)
display("        next with ∞ bounds")

nlp = ADNLPModel(objFunc, x₀,  [-100.0; -100.0],  [100.0; 100.0])

stp  = NLPlbfgsbS(nlp, x₀)

x = stp.current_state.x
f = stp.current_state.fx

@show stp.meta.nb_of_stop

@test abs(f) < 0.00001
@test abs(x[1]-1) < 0.00001
@test abs(x[2]-1) < 0.00001


@show x

reinit!(stp)
reset!(nlp)
display("        next with finite bounds")

nlp = ADNLPModel(objFunc, x₀, [0.0; 0.0], [0.5; 100.0])
stp  = NLPlbfgsbS(nlp, x₀)

x = stp.current_state.x
f = stp.current_state.fx

@show stp.meta.nb_of_stop

@test abs(f-0.25) < 0.00001
@test abs(x[1]-0.5) < 0.00001
@test abs(x[2]-0.25) < 0.00001


@show x


@test 1 == 1
