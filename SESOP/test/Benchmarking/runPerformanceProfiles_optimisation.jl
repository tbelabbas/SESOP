using Logging

using NLPModels
using SolverBenchmark

using Printf
using DataFrames
using SolverTools
using SolverCore

using Stopping
using OneDmin


include("../src/SESOP_benchmarking.jl")
include("../src/SESOP_b.jl")


maxiter = 1000000000
maxtime = 600.0
Lp = Inf

atol = 1e-5
rtol = 1e-8

precision = 0.01
mem_SE = 4
max_iter_SE = 10

nb_g = 1
nb_p = 4
nb_n = 0
mem = [nb_g, nb_p, nb_n]

verbose = 0
subspace_verbose = 0
h = 2

test_case = "p_$(precision)_mem_[$(nb_g)_$(nb_p)_$(nb_n)]_mt_$(maxtime)_mi_$(maxiter)_atol_$(atol)_rtol_$(rtol)"

include("PerformanceProfile.jl")
