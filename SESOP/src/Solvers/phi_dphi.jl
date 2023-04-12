export phi_dphi
"""
If f : Rⁿ → R then we have h(θ) = f(x + θ*d).
We can use φ(θ) = h(θ) - h(0) - τ₀*h'(0).
Inputs are a line model and a LSAtT structure.
This function returns  φ(θ) and φ'(θ).
"""
function phi_dphi(h :: LineModel,  state :: LSAtT;  τ₀ :: Float64 = 0.01)
    ht, dht = objgrad(h, state.x)
    update!(state, ht = ht, gt = dht)

    if state.x == 0.0
        φt = 0.0                      # known that φ(0) = 0.0
        dφt = (1.0 - τ₀) * state.g₀   # known that φ'(0) = (1.0 - τ₀) * h'(0)
    else
        φt = state.ht - state.h₀ - τ₀ * state.x* state.g₀
        dφt = state.gt - τ₀ * state.g₀
    end

    return φt, dφt
end
