# ============================================================
# ETAS-SPDE MCMC sampler
# ============================================================

using Distributions
using LinearAlgebra
using StatsBase

# ============================================================
# 1. Numerical constants
# ============================================================

const EPS_RHO = 1e-12
const EPS_SIG = 1e-12

# ============================================================
# 2. SPDE priors
# ============================================================

function logprior_spde(
    ρ,
    σ,
    μ;
    mρ = log(4.1),
    sρ = 4.9,
    mσ = log(6.1),
    sσ = 5.30,
    mμ = -1.0,
    sμ = 5.8,
    debug::Bool = false
)
    if ρ <= 0 || σ <= 0 || !isfinite(ρ) || !isfinite(σ) || !isfinite(μ)
        return -Inf
    end

    lpρ = logpdf(LogNormal(mρ, sρ), ρ)
    lpσ = logpdf(LogNormal(mσ, sσ), σ)
    lpμ = logpdf(Normal(mμ, sμ), μ)

    if debug
        @info "logprior_spde" ρ=ρ σ=σ μ=μ lpρ=lpρ lpσ=lpσ lpμ=lpμ
    end

    return lpρ + lpσ + lpμ
end

function logprior_offspring_lognormal(θ::Vector{Float64}, bounds)
    for i in eachindex(θ)
        if θ[i] < bounds[i][1] || θ[i] > bounds[i][2]
            return -Inf
        end
    end

    μlog = log.([0.01, 1.1, 0.01, 1.9, 1.6, 0.1, 0.5])
    σlog = [1.0, 2.5, 1.0, 0.8, 0.4, 3.0, 0.7]

    lp = 0.0

    for i in eachindex(θ)
        lp += logpdf(LogNormal(μlog[i], σlog[i]), θ[i])
    end

    return lp
end

# ============================================================
# 3. Bounded transformations
# ============================================================

function safe_logit_forward(x::Real, a::Real, b::Real; eps::Real = EPS_RHO)
    if !(a < b)
        throw(ArgumentError("safe_logit_forward: lower bound must be smaller than upper bound"))
    end

    if !(a < x < b)
        @warn "Value outside bounds in safe_logit_forward; clamping" value=x a=a b=b
        x = clamp(x, a + eps, b - eps)
    end

    return log((x - a) / (b - x))
end

function logit_inverse(η::Real, a::Real, b::Real)
    t = 1 / (1 + exp(-η))
    return a + (b - a) * t
end

function dlogit_inverse(η::Real, a::Real, b::Real)
    t = 1 / (1 + exp(-η))
    return (b - a) * t * (1 - t)
end

# ============================================================
# 4. ETAS offspring likelihood
# ============================================================

function etas_log_likelihood(
    K,
    α,
    c,
    p,
    q,
    D,
    γ,
    x::Vector{Int},
    catalog,
    Tspan
)
    cx = counts(x, length(x) + 1)
    nS = @views cx[2:end]

    loglik = 0.0

    for i in 1:length(catalog)
        parent_idx = x[i] - 1

        ΔM_i = catalog.ΔM[i]
        t_i = catalog.t[i]
        lon_i = catalog.lon[i]
        lat_i = catalog.lat[i]

        κm = κ(ΔM_i, K, α)

        loglik -= κm * H(Tspan, t_i, c, p)
        loglik += nS[i] * log(κm)

        if parent_idx != 0
            t_parent = catalog.t[parent_idx]
            lon_parent = catalog.lon[parent_idx]
            lat_parent = catalog.lat[parent_idx]
            ΔM_parent = catalog.ΔM[parent_idx]

            loglik += log(h(t_i, t_parent, c, p))
            loglik += log(
                v(
                    lon_i,
                    lon_parent,
                    lat_i,
                    lat_parent,
                    ΔM_parent + catalog.M₀,
                    q,
                    D,
                    γ
                )
            )
        end
    end

    return loglik
end

# ============================================================
# 5. Main ETAS-SPDE MCMC sampler
# ============================================================

function etas_spde_mcmc_full(
    catalog::Catalog{T},
    M,
    di,
    ν,
    S,
    C,
    pts,
    nsteps::Int;
    K0 = 0.01,
    α0 = 1.9,
    c0 = 0.01,
    p0 = 1.1,
    q0 = 1.8,
    D0 = 0.10,
    γ0 = 0.5,
    ρ0 = 2.0,
    σ0 = 0.5,
    μ0 = -2.0
) where {T<:Real}

    n = length(catalog.t)
    Tmax = maximum(catalog.t)

    # --------------------------------------------------------
    # Parameter order:
    # θ = [c, p, K, α, q, D, γ]
    # φ = [ρ, σ, μ_spde]
    # --------------------------------------------------------

    θ = [c0, p0, K0, α0, q0, D0, γ0]
    c, p, K, α, q, D, γ = θ

    ρ = ρ0
    σ = σ0
    μ_spde = μ0
    φ = [ρ, σ, μ_spde]

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    bounds_offspring = [
        (0.0005, 10.5),   # c
        (1.01, 10.0),     # p
        (0.005, 10.0),    # K
        (0.6, 10.0),      # α
        (1.01, 10.0),     # q
        (0.0001, 10.0),   # D
        (0.01, 10.0)      # γ
    ]

    bounds_spde = [
        (0.0001, 5.0),    # ρ
        (0.00001, 8.0),   # σ
        (-10.0, 10.0)     # μ_spde
    ]

    # --------------------------------------------------------
    # Safe initial values
    # --------------------------------------------------------

    ρ = clamp(ρ, bounds_spde[1][1] + EPS_RHO, bounds_spde[1][2] - EPS_RHO)
    σ = clamp(σ, bounds_spde[2][1] + EPS_SIG, bounds_spde[2][2] - EPS_SIG)
    μ_spde = clamp(μ_spde, bounds_spde[3]...)

    φ .= [ρ, σ, μ_spde]

    aρ, bρ = bounds_spde[1]
    aσ, bσ = bounds_spde[2]

    ηρ = safe_logit_forward(ρ, aρ, bρ; eps = EPS_RHO)
    ησ = safe_logit_forward(σ, aσ, bσ; eps = EPS_SIG)

    # --------------------------------------------------------
    # Branching process initialization
    # --------------------------------------------------------

    b = BranchingProcess(T, n)

    xp_init = Product([
        Categorical(inv(i) .* ones(i))
        for i in 1:length(catalog.t)
    ])

    x = rand(xp_init)

    μvec = zeros(Float64, n)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    chain_K = Float64[]
    chain_α = Float64[]
    chain_c = Float64[]
    chain_p = Float64[]
    chain_q = Float64[]
    chain_D = Float64[]
    chain_γ = Float64[]

    chain_μ = Vector{Vector{Float64}}()
    chain_ρ = Float64[]
    chain_σ = Float64[]
    chain_μspde = Float64[]

    chain_intensity = Vector{Vector{Float64}}()
    chain_nbg = Int[]

    # --------------------------------------------------------
    # Proposal scales
    # --------------------------------------------------------

    σ_off_vec = fill(0.01, 7)
    σ_spde_vec = [2.5, 0.5, 0.2]

    # ========================================================
    # MCMC loop
    # ========================================================

    for s in 1:nsteps

        # ----------------------------------------------------
        # 1. Update branching structure
        # ----------------------------------------------------

        intensity, l_spde, random_field_mode, He, Q =
            compute_intensity(
                M,
                smoothclamp(ρ, 0.01, 5.0),
                σ,
                μ_spde,
                x,
                di,
                ν,
                s
            )

        μvec = S * intensity

        update_weights!(
            b,
            catalog,
            μvec,
            K,
            α,
            c,
            p,
            q,
            D,
            γ
        )

        StatsBase.sample!(b, x)

        # ----------------------------------------------------
        # 2. Metropolis-Hastings for ETAS offspring parameters
        # ----------------------------------------------------

        Nsamples_off = s <= 100 ? 100 : 10

        loglik_current = etas_log_likelihood(
            K,
            α,
            c,
            p,
            q,
            D,
            γ,
            x,
            catalog,
            Tmax
        )

        logprior_current = logprior_offspring_lognormal(
            θ,
            bounds_offspring
        )

        acc_off = 0

        for _ in 1:Nsamples_off
            logθ = log.(θ)
            logθ_star = similar(logθ)

            for i in eachindex(logθ)
                logθ_star[i] = rand(Normal(logθ[i], σ_off_vec[i]))
            end

            θ_star = exp.(logθ_star)

            if any(
                θ_star[i] < bounds_offspring[i][1] ||
                θ_star[i] > bounds_offspring[i][2]
                for i in eachindex(θ_star)
            )
                continue
            end

            loglik_star = etas_log_likelihood(
                θ_star[3],
                θ_star[4],
                θ_star[1],
                θ_star[2],
                θ_star[5],
                θ_star[6],
                θ_star[7],
                x,
                catalog,
                Tmax
            )

            if !isfinite(loglik_star)
                loglik_star = -1e12
            end

            logprior_star = logprior_offspring_lognormal(
                θ_star,
                bounds_offspring
            )

            log_ratio =
                (loglik_star + logprior_star) -
                (loglik_current + logprior_current)

            if rand() < min(1.0, exp(log_ratio))
                θ .= θ_star
                c, p, K, α, q, D, γ = θ

                loglik_current = loglik_star
                logprior_current = logprior_star

                acc_off += 1
            end
        end

        acc_rate_off = acc_off / Nsamples_off

        if s <= 800
            if acc_rate_off < 0.2
                σ_off_vec .*= 0.8
            elseif acc_rate_off > 0.6
                σ_off_vec .*= 1.2
            end
        end

        σ_off_vec = clamp.(σ_off_vec, 1e-4, 2.0)

        # ----------------------------------------------------
        # 3. Metropolis-Hastings for SPDE parameters
        # ----------------------------------------------------

        Nsamples_spde = 20
        acc_spde = 0

        d_field = length(random_field_mode)
        z_common = randn(d_field)

        for _ in 1:Nsamples_spde
            ηρ_star = rand(Normal(ηρ, σ_spde_vec[1]))
            ησ_star = rand(Normal(ησ, σ_spde_vec[2]))
            μ_star = rand(Normal(μ_spde, σ_spde_vec[3]))

            ρ_star = logit_inverse(ηρ_star, aρ, bρ)
            σ_star = logit_inverse(ησ_star, aσ, bσ)

            intensity_prop,
            l_spde_prop,
            random_field_mode_prop,
            H_prop,
            Q_prop = compute_intensity(
                M,
                smoothclamp(ρ_star, 0.01, 5.0),
                σ_star,
                μ_star,
                x,
                di,
                ν,
                s
            )

            u_old = random_field_mode + He.PtL' \ z_common
            u_new = random_field_mode_prop + H_prop.PtL' \ z_common

            μvec_old = S * exp.(μ_spde .+ u_old)
            μvec_new = S * exp.(μ_star .+ u_new)

            intensity_old = exp.(μ_spde .+ u_old)
            intensity_new = exp.(μ_star .+ u_new)

            log_jac_old =
                log(abs(dlogit_inverse(ηρ, aρ, bρ))) +
                log(abs(dlogit_inverse(ησ, aσ, bσ)))

            log_jac_new =
                log(abs(dlogit_inverse(ηρ_star, aρ, bρ))) +
                log(abs(dlogit_inverse(ησ_star, aσ, bσ)))

            prior_old =
                logprior_spde(
                    smoothclamp(ρ, 0.01, 5.0),
                    σ,
                    μ_spde
                ) +
                log_jac_old +
                0.5 * logdet(Q) -
                0.5 * dot(u_old, Q, u_old) -
                0.5 * logdet(He)

            prior_new =
                logprior_spde(
                    smoothclamp(ρ_star, 0.01, 5.0),
                    σ_star,
                    μ_star
                ) +
                log_jac_new +
                0.5 * logdet(Q_prop) -
                0.5 * dot(u_new, Q_prop, u_new) -
                0.5 * logdet(H_prop)

            logpost_old =
                logsum1(μvec_old, x) +
                prior_old -
                sum(di.w .* intensity_old)

            logpost_new =
                logsum1(μvec_new, x) +
                prior_new -
                sum(di.w .* intensity_new)

            if rand() < min(1.0, exp(logpost_new - logpost_old))
                ρ = ρ_star
                σ = σ_star
                μ_spde = μ_star
                φ .= [ρ, σ, μ_spde]

                ηρ = ηρ_star
                ησ = ησ_star

                μvec = μvec_new
                l_spde = l_spde_prop
                random_field_mode = random_field_mode_prop
                He = H_prop
                Q = Q_prop
                intensity = intensity_new

                acc_spde += 1
            end
        end

        acc_rate_spde = acc_spde / Nsamples_spde

        if acc_rate_spde < 0.2
            σ_spde_vec .*= 0.6
        elseif acc_rate_spde > 0.6
            σ_spde_vec .*= 1.2
        end

        σ_spde_vec = clamp.(σ_spde_vec, 0.05, 3.0)

        # ----------------------------------------------------
        # 4. Store chains
        # ----------------------------------------------------

        push!(chain_K, K)
        push!(chain_α, α)
        push!(chain_c, c)
        push!(chain_p, p)
        push!(chain_q, q)
        push!(chain_D, D)
        push!(chain_γ, γ)

        push!(chain_μ, copy(μvec))
        push!(chain_ρ, ρ)
        push!(chain_σ, σ)
        push!(chain_μspde, μ_spde)

        push!(chain_intensity, copy(intensity))
        push!(chain_nbg, count(x .== 1))

        # ----------------------------------------------------
        # 5. Console diagnostics
        # ----------------------------------------------------

        println(
            "Step $s done ",
            "(acc_off=$(round(acc_rate_off, digits=3)), ",
            "acc_spde=$(round(acc_rate_spde, digits=3)))"
        )

        println("Current φ: ρ=$ρ, σ=$σ, μ_spde=$μ_spde")
        println("θ = ", θ)
        println("number of background events: ", chain_nbg[end])
    end

    return (
        chain_K,
        chain_α,
        chain_c,
        chain_p,
        chain_q,
        chain_D,
        chain_γ,
        chain_μ,
        chain_ρ,
        chain_σ,
        chain_μspde,
        chain_intensity,
        chain_nbg
    )
end
