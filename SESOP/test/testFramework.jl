
names = ["id", "prec", "proj", "hess"]



using ProfileView


function Compute_SESOP( to, nlp, stp, pres, m, maxiter_SE; nb_ts = 1:3)
    df = DataFrame()
    df.Type = ["Nb_it", "T(ms)", "Nb_o", "Nb_g", "Nb_H", "Nb_hprd", "‖∇f‖", "f", "status"]
    #df.SESOP_pastiche_1 =


    for i in [1]
        df.new_column = test_sesop_c(to, SESOP_pastiche, i,
                                   stp, nlp, m, verbose,
                                   atol, rtol, pres, maxiter_SE)
        rename!(df, :new_column => "SESOP_$(names[i])")
    end

    return df
end

function test_sesop_c(to, fn :: Function, mat, stp_i, nlp_i, m_i, v_i,
                    atol_i, rtol_i,
                    pres_i, itSE_i)

    reset!(nlp_i)
    reinit!(stp_i)
    stp, iter, x, f = fn(
                            nlp_i, stp=stp_i, mem=m_i,
                            atol=atol_i, rtol=rtol_i,
                            max_iter_SE=itSE_i, precision=pres_i,
                            matrix=mat
                        )

    reset!(nlp_i)
    reinit!(stp_i)
    tm = sum(m_i)
    name = String(Symbol(fn))
    stp, iter, x, f= @timeit to "$(name)_$mat" fn(
                                                            nlp_i,
                                                            stp=stp_i,
                                                            mem=m_i,
                                                            atol=atol_i,
                                                            rtol=rtol_i,
                                                            max_iter_SE=itSE_i,
                                                            precision=pres_i,
                                                            verbose=v_i,
                                                            subspace_verbose=v_i,
                                                            matrix=mat
                                                        )

    return  [
                stp.meta.nb_of_stop,
                TimerOutputs.time(to["$(name)_$mat"])*10e-6,
                neval_obj(nlp_i), neval_grad(nlp_i), neval_hess(nlp_i),
                neval_hprod(nlp_i), norm(grad(nlp_i, x)), f,
                getStatus(stp)
            ]
end