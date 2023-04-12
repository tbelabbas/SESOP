using LinearOperators

using NLPModels
using NLPModelsJuMP
using OptimizationProblems

using Test
using TimerOutputs

include("../../src/Models/SEModel.jl")


function prod_OP_D(op :: AbstractLinearOperator, mat)
    res = ones(op.ncol, size(mat, 2))
    i=1
    for col in eachcol(mat)
        res[:, i] = op*col
        i+=1
    end
    return res
end


n = 500
m = 5

@show n,m

#nlp = MathOptNLPModel(PureJuMP.woods(n=n), name="woods")

x = nlp.meta.x0
gradient = grad(nlp, x)

D = rand(n, m)
SE = SEModel(nlp, x, D, gradient)

α = rand(length(SE.meta.x0))
v = rand(length(SE.meta.x0))

B_1 = InverseBFGSOperator(Float64, n)
B_2 = InverseLBFGSOperator(Float64, n, mem=m)
B_3 = CompactInverseBFGSOperator(Float64, n, mem=m)

B = [B_1, B_2, B_3]
Bnames = ["BFGS", "LBFGS", "CBFGS"]

for i in 1:m
    for B_op in B
        push!(B_op, rand(n), rand(n))
    end
end

index=1
let index = index
    for B_op in B
        to = TimerOutput()

        a = @timeit to "Mat(OP_D)_$(Bnames[index])" Matrix(B_op*D)
        b = @timeit to "Mat(OP)_D_$(Bnames[index])" Matrix(B_op)*D
        c = @timeit to "Prod_OP_D_$(Bnames[index])" prod_OP_D(B_op, D)


        @test a ≈ b ;
        @test a ≈ c ;

        @show to
        index += 1
    end

end

;
