include("SESOP.jl")

function SESOP_p0(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/2
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_p1(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/10
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_p2(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/100
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_p3(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/1000
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_p4(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/500
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_p5(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global pres = 1/50
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = pres,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end

function SESOP_g0(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [0, 5]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [0, 5],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_g1(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [1, 4]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [1, 4],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_g2(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [2, 3]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [2, 3],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_g3(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [3, 2]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [3, 2],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_g4(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [4, 1]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [4, 1],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_g5(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = [5, 0]
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = [5, 0],
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end

function SESOP_m0(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = 100
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = config,
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_m1(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = 50
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = config,
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_m2(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = 20
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = config,
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end


function SESOP_m3(
                    nlp :: AbstractNLPModel;
                    stp :: NLPStopping = NLPStopping(nlp, NLPAtX(nlp.meta.x0)),
                    atol = 1e-5,
                    rtol = 1e-7,
                    precision = 0.001,
                    maxtime = 10.0,
                    mem_SE :: Int = 4,
                    max_iter_SE = 300,
                    verbose :: Int = 0,
                    subspace_verbose :: Int = 0,
                    subspace_solver :: Function = bfgs_StopLS
                  )
   global config = 5
    return SESOP(
                    nlp; stp=stp, atol = atol, rtol = rtol,
                    precision = precision,
                    maxtime = maxtime,
                    mem_SE = mem_SE,
                    max_iter_SE = max_iter_SE,
                    mem = config,
                    verbose = verbose,
                    subspace_verbose = subspace_verbose,
                    subspace_solver = subspace_solver
                )
end

