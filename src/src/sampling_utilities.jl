
# ============================================================
# SPDE / LGCP utilities
# ============================================================

using LinearAlgebra
using SparseArrays
using Optim
using LambertW
using NearestNeighbors
using TriangleMesh
using GeometryBasics

# ============================================================
# 1. Data structures
# ============================================================

struct Gridmesh
    mesh
    w
    S
    imat
end

struct LogGaussianCoxProcess
    M
    range
    sd
    mean
    xp
end

struct LaplaceApproximation{SM,F,O,YT,XT,WT}
    size
    Q::SM
    lik::F
    obj::O
    y::YT
    x::XT
    w::WT
    b
end

# ============================================================
# 2. Small utilities
# ============================================================

function logsum1(μ::Vector{T}, x) where {T}
    s = zero(T)

    @inbounds for i in eachindex(μ)
        s += x[i] == 1 ? log(μ[i]) : zero(T)
    end

    return s
end

function smoothclamp(x, low, high)
    r = high - low
    y = clamp((x - low) / r, 0, 1)
    y = y * y * (3 - 2 * y)
    return y * r + low
end

# ============================================================
# 3. Mesh matrices
# ============================================================

function component_matrices(mesh)
    n_edge = size(mesh.edge, 2)

    ii = zeros(Int, n_edge)
    jj = zeros(Int, n_edge)

    for (ei, edge) in enumerate(eachcol(mesh.edge))
        ii[ei], jj[ei] = edge
    end

    ii = [ii; jj]
    jj = [jj; ii[1:n_edge]]

    G = sparse(ii, jj, zeros(2 * n_edge))
    C = sparse(ii, jj, zeros(2 * n_edge))
    C_tilde = spdiagm(0 => zeros(mesh.n_point))

    for triangle in eachcol(mesh.cell)
        i, j, k = triangle

        vi, vj, vk = [mesh.point[:, ind] for ind in (i, j, k)]

        ei = vk - vj
        ej = vi - vk
        ek = vj - vi

        edges = Dict(i => ei, j => ej, k => ek)

        cosθ = dot(ei, ej) / (norm(ei) * norm(ej))
        θk = acos(clamp(cosθ, -1.0, 1.0))

        area = 0.5 * norm(ei) * norm(ej) * sin(θk)

        for m in (i, j, k)
            C_tilde[m, m] += area / 3
        end

        for m in (i, j, k), n in (i, j, k)
            C[m, n] += m == n ? area / 6 : area / 12
            G[m, n] += dot(edges[m], edges[n]) / (4 * area)
        end
    end

    return C, C_tilde, G
end

function inverse_distance_weights(dists::AbstractVector)
    w = [d > 0 ? 1 / d : 1 / eps() for d in dists]
    return w ./ sum(w)
end

function observation_matrix(mesh::TriMesh, points::AbstractMatrix)
    npoint = size(points, 2)
    nmesh = size(mesh.point, 2)

    tree = KDTree(mesh.point)
    idx, dists = knn(tree, points, 3)

    ii = repeat(1:npoint, inner = 3)
    jj = reduce(vcat, idx)
    ww = mapreduce(inverse_distance_weights, vcat, dists)

    return sparse(ii, jj, ww, npoint, nmesh)
end

function polygon_area(poly)
    n = length(poly)

    return 0.5 * abs(
        sum(
            poly[i][1] * poly[mod1(i + 1, n)][2] -
            poly[mod1(i + 1, n)][1] * poly[i][2]
            for i in 1:n
        )
    )
end

function intersected_point_area(mesh, domain)
    point_areas = zeros(Float64, size(mesh.point, 2))

    xs = [p[1] for p in domain]
    ys = [p[2] for p in domain]

    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    inside(p) = xmin <= p[1] <= xmax && ymin <= p[2] <= ymax

    for triangle in eachcol(mesh.cell)
        i, j, k = triangle

        vi, vj, vk = [Tuple(mesh.point[:, idx]) for idx in (i, j, k)]

        if !any(inside(p) for p in (vi, vj, vk))
            continue
        end

        area = polygon_area([vi, vj, vk])

        point_areas[i] += area / 3
        point_areas[j] += area / 3
        point_areas[k] += area / 3
    end

    return point_areas
end

# ============================================================
# 4. Optim / Cholesky wrapper
# ============================================================

struct OptimCholWrapper{T} <: SparseArrays.AbstractSparseMatrix{Float64,Int}
    data::T
end

function Base.similar(::OptimCholWrapper, ::Type{T}, n::Int, m::Int) where {T}
    @assert n == 0 == m
    return zeros(0, 0)
end

Base.:\(x::OptimCholWrapper, y::Vector{T}) where {T} = x.data \ y
Base.copy(x::OptimCholWrapper) = x
Base.show(io::IO, x::OptimCholWrapper) = show(io, x.data)
LinearAlgebra.logdet(x::OptimCholWrapper) = logdet(x.data)
Base._all(::typeof(isfinite), itr::OptimCholWrapper, ::Colon) = true

# ============================================================
# 5. Laplace approximation utilities
# ============================================================

function lambertwexp(x)
    if x < 700
        return LambertW.lambertw(exp(x))
    else
        return x - log(x)
    end
end

function weightstonodes(locationweight)
    return weightstonodes!(zeros(size(locationweight, 2)), locationweight)
end

function weightstonodes!(A, locationweight, b = 1)
    for j in 1:size(locationweight, 2)
        A[j] += b * sum(locationweight[:, j])
    end

    return A
end

function initial_guess(fixed_field, locationweight, q)
    aa = weightstonodes!(zero(fixed_field), locationweight)

    @. aa = aa / q - lambertwexp(fixed_field - log(q) + aa / q)

    return aa
end

function background_locationweight(S, x)
    idx_bg = findall(x .== 1)
    return S[idx_bg, :], idx_bg
end

# ============================================================
# 6. Log-Gaussian Cox process
# ============================================================

function logpdff(d::LogGaussianCoxProcess, di::Gridmesh, x, s)
    sd = d.sd
    scale = di.w
    range = d.range
    fixedfield = d.mean

    prec = 1 / sd^2
    ν = 1.0

    locationweight, idx_bg = background_locationweight(di.S, x)

    (l, _, pz), random_field_mode, H, Q =
        poisson_gp(
            d.M,
            locationweight,
            fixedfield,
            prec,
            range,
            scale,
            di.mesh.n_point,
            ν,
            x
        )

    H = H.data

    intensity = exp.(fixedfield .+ random_field_mode)

    l1 = sum(log.(locationweight * intensity))
    l2 = sum(scale .* intensity)

    return l1 - l2, random_field_mode, H, Q
end

function poisson_gp(
    M,
    locationweight,
    fixed_field,
    prec,
    range,
    integration_weights,
    gridsize,
    ν,
    x
)
    Q = M(range, prec, ν)
    chol = cholesky(Q)
    Qlogdet = logdet(chol)

    la = LaplaceApproximation(
        gridsize,
        locationweight,
        integration_weights,
        Q,
        chol,
        x
    )

    return poisson_gp(la, fixed_field, Q, Qlogdet)
end

function poisson_gp(la::LaplaceApproximation, fixed_field, Q, Qlogdet)
    zhat, H = poisson_hatZ(la, fixed_field, Q)

    l = LA_factors(zhat, H, Q, Qlogdet)

    return l, reshape(zhat, la.size), H, Q
end

function LA_factors(zhat, H, Q, Qlogdet)
    d = length(zhat)

    pz = 0.5 * Qlogdet -
         d / 2 * log(2π) -
         0.5 * dot(zhat, Q, zhat)

    m = d / 2 * log(2π) -
        0.5 * logdet(H)

    a = m + pz

    return ifelse(isnan(a), -Inf, a), m, pz
end

function poisson_hatZ(la::LaplaceApproximation, fixed_field::Number, Q)
    fill!(la.x, fixed_field)
    return poisson_hatZ(la, Q)
end

function poisson_hatZ(la::LaplaceApproximation, fixed_field, Q)
    copy!(la.x, fixed_field)
    return poisson_hatZ(la, Q)
end

function poisson_hatZ(la::LaplaceApproximation, Q)
    copy!(la.Q, Q)

    initial = vec(initial_guess(la.x, la.y, Q[1]))
    obj = la.obj

    opt = Optim.optimize(obj, initial)

    x = Optim.minimizer(opt)
    H = Optim.hessian!(obj, x)

    return x, H
end

function LaplaceApproximation(
    gridsize,
    locationweight,
    integration_weights,
    Q,
    chol,
    b
)
    H = copy(Q)

    y = locationweight
    x = zeros(gridsize)
    w = integration_weights

    obj = LA_poisson_objective_chol(
        y,
        x,
        w,
        Q,
        H,
        chol,
        b
    )

    return LaplaceApproximation(
        gridsize,
        Q,
        nothing,
        obj,
        y,
        x,
        w,
        b
    )
end

function LA_poisson_objective_chol(
    locationweight,
    fixed_field,
    integration_weights,
    Q,
    H,
    cholQ,
    b
)
    function fgh!(F, D, chol, x)
        random_field = x

        if chol !== nothing
            copy!(H, Q)
        end

        if D !== nothing || F !== nothing
            Qx = Q * x

            if D !== nothing
                D .= Qx
                weightstonodes!(D, locationweight, -1)
            end

            if F !== nothing
                F = 0.5 * x' * Qx - sum(locationweight * random_field)
            end
        end

        for i in eachindex(random_field)
            a = integration_weights[i] *
                exp(fixed_field[i] + random_field[i])

            if !isfinite(a)
                return -Inf
            end

            F !== nothing && (F += a)
            D !== nothing && (D[i] += a)
            chol !== nothing && (H[i, i] += a)
        end

        if chol !== nothing
            cholesky!(chol.data, Symmetric(H))
        end

        if F !== nothing
            return F
        end
    end

    initial = vec(zero(fixed_field))

    return Optim.TwiceDifferentiable(
        Optim.only_fgh!(fgh!),
        initial,
        0.0,
        copy(initial),
        OptimCholWrapper(cholQ)
    )
end

# ============================================================
# 7. Intensity computation
# ============================================================

function compute_intensity(M, ρ, σ, μ, x, di, ν, s)
    l, random_field, H, Q =
        logpdff(
            LogGaussianCoxProcess(M, ρ, σ, μ, x),
            di,
            x,
            s
        )

    intensity = exp.(μ .+ random_field)

    return intensity, l, random_field, H, Q
end

# ============================================================
# 8. Dual mesh
# ============================================================

function dual_mesh(mesh)
    centroids = [
        mean(mesh.point[:, mesh.cell[:, i]], dims = 2)[:]
        for i in 1:size(mesh.cell, 2)
    ]

    polygons = Vector{Union{Polygon{2,Float64},Nothing}}(
        undef,
        mesh.n_point
    )

    edges = vcat(
        mesh.cell[[1, 2], :]',
        mesh.cell[[2, 3], :]',
        mesh.cell[[3, 1], :]'
    )

    edges = [sort(edge) for edge in eachrow(edges)]

    edge_counts = Dict{Tuple{Int,Int},Int}()

    for edge in edges
        edge_counts[Tuple(edge)] = get(edge_counts, Tuple(edge), 0) + 1
    end

    border_edges = [edge for (edge, count) in edge_counts if count == 1]
    border_points = unique(vcat(border_edges...))

    for i in 1:mesh.n_point
        plist = []

        for k in 1:3
            adjacent = findall(x -> x == i, mesh.cell[k, :])

            for tri_idx in adjacent
                tri = mesh.cell[:, tri_idx]

                push!(plist, centroids[tri_idx])

                other_idx = setdiff(1:3, k)

                push!(
                    plist,
                    (mesh.point[:, tri[k]] +
                     mesh.point[:, tri[other_idx[1]]]) / 2
                )

                push!(
                    plist,
                    (mesh.point[:, tri[k]] +
                     mesh.point[:, tri[other_idx[2]]]) / 2
                )
            end
        end

        if i in border_points
            b_edges = [edge for edge in border_edges if i in edge]

            for edge in b_edges
                other = edge[1] == i ? edge[2] : edge[1]

                push!(
                    plist,
                    (mesh.point[:, i] + mesh.point[:, other]) / 2
                )
            end
        end

        if isempty(plist)
            x, y = mesh.point[:, i]
            δ = 0.01

            polygons[i] = Polygon([
                Point(x - δ, y - δ),
                Point(x + δ, y - δ),
                Point(x + δ, y + δ),
                Point(x - δ, y + δ)
            ])
        else
            pmat = hcat(plist...)'
            center = mean(pmat, dims = 1)

            angles = atan.(
                pmat[:, 2] .- center[1, 2],
                pmat[:, 1] .- center[1, 1]
            )

            sorted_idx = sortperm(angles)

            polygons[i] = Polygon([
                Point(pmat[idx, 1], pmat[idx, 2])
                for idx in sorted_idx
            ])
        end
    end

    return polygons
end

