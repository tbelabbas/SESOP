using Test

using NLPModels
using NLPModelsJuMP
using OptimizationProblems


"""
    Test the ability to play with and modify and NLP when it's in another
    NLP as a parameter.
"""
mutable struct testModel{T, S} <: AbstractNLPModel{T, S}

    counters :: Counters
    meta :: NLPModelMeta{T, S}

    ms_nlp :: AbstractNLPModel   # Main space model
end

function testModel(x₀ :: AbstractVector, ms_nlp :: AbstractNLPModel)

        meta = NLPModelMeta(length(x₀), x0=x₀)

        return testModel(Counters(), meta, ms_nlp)
end

"""
    Test the changing of "in" NLPModel dimension.
"""
in_nlp = MathOptNLPModel(PureJuMP.woods(n=50), name="woods")
n = in_nlp.meta.nvar

out_nlp = testModel(zeros(n), in_nlp)

@test out_nlp.ms_nlp.meta.nvar == n

# Cannot do this
# in_nlp.meta.nvar = 4