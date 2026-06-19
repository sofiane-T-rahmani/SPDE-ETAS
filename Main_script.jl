# ============================================================
# ETAS-SPDE MCMC 
# ============================================================

# -----------------------------
# 1. Packages
# -----------------------------
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Clustering
using Dates: DateTime
using Distributions: MersenneTwister, Exponential, Poisson, Normal
using TriangleMesh
using Random
using Random: AbstractRNG

include("src/catalog.jl")
include("src/branching_process.jl")
include("src/spatialSPDE.jl")
include("src/sampling_utilities.jl")
include("src/SPDE-ETAS_sampler.jl")
include("src/etas.jl")

# ============================================================
# 2. Configuration
# ============================================================

const DATA_FILE = "data/synthetic_data_case_01_patches.txt"
const OUTDIR = "mcmc_results"

const NITER = 1000
const BURNIN = 100

const K0 = 0.03
const α0 = 1.80
const c0 = 0.003
const p0 = 1.10
const q0 = 1.60
const D0 = 0.010
const γ0 = 0.40

const ρ0 = 1.5
const σ0 = 0.5
const μ0 = 0.01

const SPDE_ν = 0.5
const GRID_POINTS = 50

# ============================================================
# 3. Load catalog
# ============================================================

data = readdlm(DATA_FILE)

time = data[:, 1]
mag  = data[:, 2]
lon  = data[:, 3]
lat  = data[:, 4]

Tmax = maximum(time)
pts = hcat(lon, lat)

# ============================================================
# 4. Mesh and SPDE objects
# ============================================================

corners = [
    0.0 0.0
    5.0 0.0
    5.0 5.0
    0.0 5.0
]

domain = [
    (0.0, 0.0),
    (5.0, 0.0),
    (5.0, 5.0),
    (0.0, 5.0),
    (0.0, 0.0)
]

mesh = create_mesh(
    corners;
    info_str = "Triangular mesh of square domain.",
    verbose = false,
    check_triangulation = false,
    voronoi = true,
    delaunay = true,
    output_edges = true,
    output_cell_neighbors = true,
    quality_meshing = true,
    prevent_steiner_points_boundary = false,
    prevent_steiner_points = false,
    set_max_steiner_points = false,
    set_area_max = true,
    set_angle_min = false,
    add_switches = ""
)

C, C_tilde, G = component_matrices(mesh)
C_inv = spdiagm(0 => 1 ./ diag(C_tilde))

Sobs = observation_matrix(mesh, pts')
w = intersected_point_area(mesh, domain)

imat = Diagonal(ones(mesh.n_point))
M = spatialSPDE(2, mesh.n_point, G, C_inv, C)
di = Gridmesh(mesh, w .* Tmax, Sobs, imat)

# ============================================================
# 5. Grid for catalog object
# ============================================================



catalog = Catalog(
    time,
    mag,
    0.0,
    lon,
    lat,
    missing,
    Tmax,
    mesh.point[:, 1],
    mesh.point[:, 2]
)

# ============================================================
# 6. Save mesh points
# ============================================================

mesh_points = hcat(mesh.point[1, :], mesh.point[2, :])
writedlm("mesh_points_case01_patches.txt", mesh_points, '\t')

# ============================================================
# 7. Run MCMC
# ============================================================

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
chain_nbg = etas_spde_mcmc_full(
    catalog,
    M,
    di,
    SPDE_ν,
    Sobs,
    C,
    pts,
    NITER;
    K0 = K0,
    α0 = α0,
    c0 = c0,
    p0 = p0,
    q0 = q0,
    D0 = D0,
    γ0 = γ0,
    ρ0 = ρ0,
    σ0 = σ0,
    μ0 = μ0
)

# ============================================================
# 8. Output directory
# ============================================================

isdir(OUTDIR) || mkdir(OUTDIR)

# ============================================================
# 9. Save parameter chains
# ============================================================

params_matrix = hcat(
    chain_K[BURNIN:end],
    chain_α[BURNIN:end],
    chain_c[BURNIN:end],
    chain_p[BURNIN:end],
    chain_q[BURNIN:end],
    chain_D[BURNIN:end],
    chain_γ[BURNIN:end],
    chain_ρ[BURNIN:end],
    chain_σ[BURNIN:end],
    chain_μspde[BURNIN:end]
)

outfile_params = joinpath(
    OUTDIR,
    "chains_parameters_case01_patches.txt"
)

open(outfile_params, "w") do io
    println(io, "K,α,c,p,q,D,γ,ρ,σ,μspde")

    for i in axes(params_matrix, 1)
        println(io, join(params_matrix[i, :], ","))
    end
end

println("Saved: ", outfile_params)

# ============================================================
# 10. Save raw intensity chains
# ============================================================

outfile_intensity = joinpath(
    OUTDIR,
    "chains_intensity_case01_patches.txt"
)

open(outfile_intensity, "w") do io
    for i in BURNIN:length(chain_intensity)
        println(io, join(chain_intensity[i], ","))
    end
end

println("Saved: ", outfile_intensity)

# ============================================================
# 11. Save intensity quantiles
# ============================================================

intensity_mat = reduce(hcat, chain_intensity[BURNIN:end])'

q025 = mapslices(x -> quantile(x, 0.025), intensity_mat; dims=1)[:]
q250 = mapslices(x -> quantile(x, 0.250), intensity_mat; dims=1)[:]
q500 = mapslices(x -> quantile(x, 0.500), intensity_mat; dims=1)[:]
q750 = mapslices(x -> quantile(x, 0.750), intensity_mat; dims=1)[:]
q975 = mapslices(x -> quantile(x, 0.975), intensity_mat; dims=1)[:]

quantiles_mat = hcat(q975, q750, q500, q250, q025)

outfile_quantiles = joinpath(
    OUTDIR,
    "chains_intensity_quantiles_case01_patches.txt"
)

open(outfile_quantiles, "w") do io
    println(io, "q975,q750,q500,q250,q025")

    for i in axes(quantiles_mat, 1)
        println(io, join(quantiles_mat[i, :], ","))
    end
end

println("Saved: ", outfile_quantiles)

# ============================================================
# 12. Save number of background events
# ============================================================

outfile_nbg = joinpath(
    OUTDIR,
    "chains_nbg_case01_patches.txt"
)

open(outfile_nbg, "w") do io
    println(io, "nbg")

    for value in chain_nbg[BURNIN:end]
        println(io, value)
    end
end

println("Saved: ", outfile_nbg)

println("\nAll results saved in: ", OUTDIR)
