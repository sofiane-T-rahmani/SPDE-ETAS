import Base: length, maximum, minimum,cos
import StatsBase: sample
using Interpolations
using CSV
using Dates: DateTime
#using DeepGaussianSPDEProcesses
using DelimitedFiles
using Distributions: Exponential, MersenneTwister, Normal, Poisson
using JLD2
using LinearAlgebra
using LoopVectorization
using MCMCChains
using ReverseDiff
using Optim
using GeometryBasics
#using Plots
using TriangleMesh
using Random
using SparseArrays
using DelimitedFiles
using DataFrames
import LambertW
using StatsBase
using Clustering
using NearestNeighbors
#using StatsPlots
using Turing
using SliceSampling
using Statistics
using ThreadsX
using Zygote
using AdvancedMH
using LinearAlgebra, SparseArrays, SpecialFunctions
using Zygote
#using CairoMakie
include("src/catalog.jl")
include("src/etas.jl")
include("src/branching_process.jl")
include("src/spatialSPDE.jl")
include("src/sampling_utilities.jl")
include("src/constant_rate_sampler.jl")
#include("src/spatialSPDE.jl")
# include("src/one_layer_sampler.jl")
# include("src/two_layer_sampler.jl")
#Plots.default(show = true)
####################Initial############


const μtrue = 0.2
const Ktrue = 0.03
const αtrue = 1.80
const ptrue = 1.1
const ctrue = 0.003
const Dtrue = 0.010
const qtrue = 1.6
const ɣtrue = 0.4
const tspan = 500.0
const bvalue = 1.0
const N = 129
const X=1000.0
function gamma_moment_tuner(μ, σ)
    # gives parameters for a Gamma with given mean and standard deviation
    a = (μ/σ)^2
    b = μ/σ^2
    θ = 1/b
    return (a, b, θ)
end

function inverse_gamma_tuner(l, u; p=0.01)
    # Tunes an inverse gamma to have p percentile mass below l and above u
    function f(θ)
        a, b = exp.(θ)
        l_cp_res = cdf(InverseGamma(a,b), l) - p
        u_cp_res = 1-cdf(InverseGamma(a,b), u) - p
        l_cp_res^2 + u_cp_res^2
    end
    
    res = optimize(f, zeros(2))
    a, b = exp.(res.minimizer)
    return (a, b)
end
μa, μb, μθ = gamma_moment_tuner(0.01, 0.005)
αa, αb = inverse_gamma_tuner(αtrue/2, 2*αtrue)
ca, cb = inverse_gamma_tuner(ctrue/2, 2*ctrue)
p̃a, p̃b = inverse_gamma_tuner((ptrue-1)/2, 2*(ptrue-1))
qa, qb = inverse_gamma_tuner(qtrue/2, 2*qtrue)
Da, Db = inverse_gamma_tuner(Dtrue/2, 2*Dtrue)
ɣa, ɣb = inverse_gamma_tuner(ɣtrue/2, 2*ɣtrue)
etaspriors = ETASPriors(truncated(Normal(0,0.25),0,0.5), 
                        InverseGamma(αa, αb), 
                        truncated(Normal(0,0.25),0,0.5), 
                        InverseGamma(p̃a, p̃b),
                        truncated(Normal(0,0.25),1.10,2),
                        truncated(Normal(0,0.25),0,0.2),
                        InverseGamma(ɣa, ɣb ))	
 

#############read catalog#################
synth_data = readdlm("data/synthetic_data_case02_three_faults.txt")


############# convert latlonto km ###################
time=synth_data[:,1]
Tmax=maximum(time)
lon=synth_data[:,3]
lat=synth_data[:,4]
pts=hcat(lon,lat)
#mesh = create_mesh(pts)
nodes = collect(kmeans(pts', 150).centers')

corners = [0.0 0.0; 5.0 0.0; 5.0 5.0; 0.0 5.0]
domain = [
    (0.0, 0.0),
    (5.0, 0.0),
    (5.0, 5.0),
    (0.0, 5.0),
    (0.0, 0.0)  # <-- fermer le polygone
]
#mesh = create_mesh([nodes; corners])
#mesh = refine(mesh, divide_cell_into=2, voronoi=true)
mesh = create_mesh(
           corners;
           info_str = "Triangular mesh of convex hull of point cloud.",
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

dual_polys = dual_mesh(mesh)


#mesh= create_mesh(pts)
C, C_tilde, G=component_matrices(mesh)
C_inv = spdiagm(0 => 1 ./ diag(C_tilde))
S=observation_matrix(mesh, pts')
w=intersected_point_area(mesh, domain)
d=2
M=spatialSPDE(d,mesh.n_point,G,C_inv,C)
dual_areas = Float64[]

for poly in dual_polys
    if poly !== nothing
        point = coordinates(poly)
        n = length(point)
        area = 0.0
        for i in 1:n
            x1, y1 = Tuple(point[i])
            x2, y2 = Tuple(point[mod1(i+1, n)])  # CORRECTION ICI
            area += x1*y2 - x2*y1
        end
        area = abs(area)/2
        push!(dual_areas, area)
    else
        push!(dual_areas, 0.0)
    end
end


# rad_earth = 6378.1
#lon = lon.*110.94.*cos.(lat)
#lat= lat.*110.94

#creat a grid#
px=[minimum(lon), maximum(lon)]
py=[minimum(lat), maximum(lat)]
X_borders = hcat(px, py)
S = X_borders[:, 2] - X_borders[:, 1]
grid_points = 50
D=2
X_grid = zeros(Float64, grid_points, D)
for di in 1:D
    X_grid[:, di] = range(0, stop=S[di], length=grid_points)
end
X_mesh = collect(Iterators.product([X_grid[:, i] for i in 1:D]...))
X_mesh = hcat([collect(p) for p in X_mesh]...)'
X_grid = X_mesh .+ X_borders[:, 1]'
S=observation_matrix(mesh, pts')
imat = Diagonal(ones(mesh.n_point))
d=2
M=spatialSPDE(d,mesh.n_point,G,C_inv,C)
di=Gridmesh(mesh,w.* Tmax,S,imat)

# rad_earth = 6378.1
#lon = lon.*110.94.*cos.(lat)
#lat= lat.*110.94

#creat a grid#
px=[minimum(lon), maximum(lon)]
py=[minimum(lat), maximum(lat)]
X_borders = hcat(px, py)
S = X_borders[:, 2] - X_borders[:, 1]
grid_points = 50
D=2
X_grid = zeros(Float64, grid_points, D)
for di in 1:D
    X_grid[:, di] = range(0, stop=S[di], length=grid_points)
end
X_mesh = collect(Iterators.product([X_grid[:, i] for i in 1:D]...))
X_mesh = hcat([collect(p) for p in X_mesh]...)'
X_grid = X_mesh .+ X_borders[:, 1]'
S=observation_matrix(mesh, pts')

############ creat a catalog format suitable for the code ###############
synth_catalog = Catalog(synth_data[:,1],
                           synth_data[:,2],
                           0.0,
                           lon,
                           lat,
                           missing,
                           Tmax,
                           X_grid[:,1],
                           X_grid[:,2])

spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(10, 300), truncated(Normal(0,1),0,Inf))
crpt = ConstantRateParameters(Tmax, μa, μb, etaspriors)
# Assuming mesh.point is 2 x N
x = mesh.point[1, :]
y = mesh.point[2, :]

# Combine into Nx2 matrix
points = hcat(x, y)  # Transpose to get each point as a row


# Save to a text file
writedlm("mesh_points_case02_three_faults.txt", points, '\t')


chain_K, chain_α, chain_c, chain_p, chain_q, chain_D, chain_γ, chain_μ, chain_ρ, chain_σ, chain_μspde, chain_intensity, chain_nbg=etas_spde_mcmc_full(synth_catalog, M, di, 0.5, S,C,pts, 1000; K0=Ktrue, α0=αtrue, c0=ctrue, p0=ptrue, q0=qtrue, D0=Dtrue, γ0=ɣtrue,
                              ρ0=1.5, σ0=0.5, μ0=0.01)

# --- Dossier de sortie
outdir = "mcmc_results"
isdir(outdir) || mkdir(outdir)

# --- Itération de début après burn-in
burnin = 100

# --- Combiner les paramètres dans une seule matrice
n_iter = length(chain_K)
n_save = n_iter - burnin + 1


# ============================
# 1️⃣ Sauvegarde des paramètres
# ============================
params_matrix = hcat(
    chain_K[burnin:end],
    chain_α[burnin:end],
    chain_c[burnin:end],
    chain_p[burnin:end],
    chain_q[burnin:end],
    chain_D[burnin:end],
    chain_γ[burnin:end],
    chain_ρ[burnin:end],
    chain_σ[burnin:end],
    chain_μspde[burnin:end]
)

header_params = "K,α,c,p,q,D,γ,ρ,σ,μspde"
outfile_params = joinpath(outdir, "chains_parameters_case02_three_faults.txt")

open(outfile_params, "w") do io
    write(io, header_params * "\n")
    for i in 1:size(params_matrix, 1)
        println(io, join(params_matrix[i, :], ","))
    end
end
println("✅ Sauvegardé : chains_parameters.txt")

# ============================
# 2️⃣ Sauvegarde intensité brute
# ============================
outfile_intensity = joinpath(outdir, "chains_intensity_case02_three_faults.txt")
open(outfile_intensity, "w") do io
    for i in burnin:length(chain_intensity)
        println(io, join(chain_intensity[i], ","))
    end
end
println("✅ Sauvegardé : chains_intensity.txt")

# ============================
# 3️⃣ Calcul et sauvegarde des quantiles
# ============================
# Convertir la liste d’intensités en matrice
intensity_mat = reduce(hcat, chain_intensity[burnin:end])'  # taille : (n_iter, n_points)

# Calcul des quantiles (colonne par point)
q05  = mapslices(x -> quantile(x, 0.025), intensity_mat; dims=1)[:]
q25  = mapslices(x -> quantile(x, 0.25), intensity_mat; dims=1)[:]
q50  = mapslices(x -> quantile(x, 0.50), intensity_mat; dims=1)[:]
q75  = mapslices(x -> quantile(x, 0.75), intensity_mat; dims=1)[:]
q95  = mapslices(x -> quantile(x, 0.975), intensity_mat; dims=1)[:]
widths = q95 .- q05
println(widths)
quantiles_mat = hcat(q95[:], q75[:], q50[:], q25[:], q05[:])

header_quant = "q95,q75,q50,q25,q05"
outfile_quant = joinpath(outdir, "chains_intensity_quantiles_case02_three_faults.txt")

open(outfile_quant, "w") do io
    write(io, header_quant * "\n")
    for i in 1:size(quantiles_mat, 1)
        println(io, join(quantiles_mat[i, :], '\t'))
    end
end
println("✅ Sauvegardé : chains_intensity_quantiles.txt")

# ============================
# 4️⃣ Sauvegarde du nombre d’événements de fond
# ============================
outfile_nbg = joinpath(outdir, "chains_nbg_case_case02_three_faults.txt")
open(outfile_nbg, "w") do io
    write(io, "nbg\n")
    for val in chain_nbg[burnin:end]
        println(io, val)
    end
end
println("✅ Sauvegardé : chains_nbg_0case02_three_faults.txt")

println("\n✅ Tous les fichiers texte ont été sauvegardés dans le dossier : $outdir") 




