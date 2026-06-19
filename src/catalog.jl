abstract type AbstractCatalog end

Base.length(cat :: AbstractCatalog) = Base.length(cat.t)
Base.maximum(cat :: AbstractCatalog) = Base.maximum(cat.M)
Base.minimum(cat :: AbstractCatalog) = Base.minimum(cat.M)

struct Catalog{T<:Real} <: AbstractCatalog
    t::Vector{T}
    M::Vector{T}
    ΔM::Vector{T}
    M₀::T
    lon::Vector{T}
    lat::Vector{T}
    start_time::Union{DateTime, Missing}
    elapsed_time::T
    px::Vector{T}
    py::Vector{T}
    function Catalog{T}(t, M, M₀,lon,lat, start_time, elapsed_time,px,py) where {T<:Real}
        @assert length(t) == length(M)
        return new(t, M, M.-M₀, M₀,lon,lat, start_time, elapsed_time,px,py)
    end
end

Catalog(t::Vector{T}, M::Vector{T}, M₀::T,lon::Vector{T},lat::Vector{T}, start_time::Union{DateTime, Missing}, elapsed_time::T,px::Vector{T},py::Vector{T}) where {T<:Real} = Catalog{T}(t, M, M₀,lon,lat, start_time, elapsed_time,px,py)

struct ETASInhomogeneousPP{T <: Real}
    μ :: Function
    K::T
    α::T
    c::T
    p::T
    ETASInhomogeneousPP(μ, K::T, α::T, c::T, p::T) where T = p < 1 ? error("p must be >= 1") : new{T}(μ, K, α, c, p)
end

function simulate_Poisson(μ::T, t_end::T, b::T, rng = MersenneTwister(43771120)) where {T <: Real}
    t = T[]
    M = T[]
    t_now = zero(T)
    β = b * convert(T, log(10))
    scale = one(T) / β
    while t_now < t_end
        u = rand(rng, T)
        D = rand(rng, T) # extra call to rand to synchronize the RNG with simulate_ETAS so that the catalogs are identical if K = 0

        w = -log(u)/μ
        t_now += w
        if t_now > t_end
            break
        else
            push!(M, rand(rng, Exponential(scale)))
            push!(t, t_now)
        end        
    end
    return Catalog(t, M, zero(T), missing, t_end) 
end

function simulate_ETAS(model::ETASInhomogeneousPP{T}, t_end::T, b::T, rng = MersenneTwister(43771120)) where {T <: Real}
    # Simulates an ETAS IPP catalog using Ogata's thinning algorithm
    t = Array{T,1}()
    M = Array{T,1}() 
    t_now = zero(T)
    β = b * convert(T, log(10))
    scale = one(T) / β
    μ = model.μ
    K = model.K
    α = model.α
    c = model.c
    p = model.p
    while t_now < t_end
        if length(t) > 0
            ν = sum(κ.(M, K, α) .* h.(t_now, t, c, p))
        else
            ν = zero(T)
        end
        λ̄ = μ(t_now) + ν
        u = rand(rng, T)
        w = -log(u)/λ̄
        t_now = t_now  + w
        D = rand(rng, T)
        if length(t) > 0
            ν = sum(κ.(M, K, α) .* h.(t_now, t, c, p))
        else
            ν = zero(T)
        end
        λ = μ(t_now) + ν 
        if D*λ̄ <= λ
            if t_now > t_end
                break
            else
                push!(M, rand(rng, Exponential(scale)))
                push!(t, t_now)
            end
        end
    end
    return Catalog(t, M, zero(T), missing, t_end) 
end