function bfgs(nlp:: AbstractNLPModel;
              ϵ :: Float64 = 1e-6,
              maxiter :: Int = 200,
              B₀ = nothing)

    x = nlp.meta.x0
    iter = 0

    T = Float64
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f = obj(nlp,x)
    g = grad(nlp, x)

    xt = similar(x)
    gt = similar(g)

    if B₀ == nothing
        B = I
    else
        B₀ = B
    end

    τ₀ = 0.0005
    τ₁ = 0.9999

    @info log_row(Any[iter, f, norm(g)])

    while (norm(g, Inf) > ϵ) && (iter <= maxiter)


        d = - B*g

        hp0 = g'*d
        t=1.0
        # Simple Wolfe forward tracking
        gp = grad(nlp,x+t*d)
        hp = gp'*d
        ft = obj(nlp, x + t*d)
        #  while  ~wolfe & armijo
        while (hp <= τ₁ * hp0) && (ft <= ( f + τ₀*t*hp0))
            t *= 5
            gp = grad(nlp,x+t*d)
            hp = gp'*d
            ft = obj(nlp, x + t*d)
            @info "W", ft
        end
        tw = t

        # Simple Armijo backtracking
        ft = obj(nlp, x + t*d)
        while ft > ( f + τ₀*t*hp0)
            t *= 0.7
            ft = obj(nlp, x + t*d)
            @info "A", ft
        end

        xt = copy(x)
        x += t*d

        gt = copy(g)
        if t==tw  g = gp else g = grad(nlp, x) end
        # Update BFGS approximation.
        sk = t*d
        yk = g - gt

        nsk2 =sk'*sk
        denom = yk'*sk
        if (denom > 1.0e-20)# && (mod(iter,n+1)!=0)  #  ϵ * ∇ftNorm^α * nsk2  #  Li & Fukushima
            M = I - sk*yk'/denom
            B = M*B*M' + sk*sk'/denom
            B = 0.5*(B+B')  # make sure B is symmetric
        else
            @warn "No update, B=I, denom = ", denom
            B=Matrix(1.0I,n,n)
        end

        f = ft
        iter += 1

        @info log_row(Any[iter, f, norm(g), t, hp0])
    end
    if iter > maxiter @warn "Maximum d'itérations"
    end

    return iter, f, norm(g), x
end
