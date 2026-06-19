


struct spatialSPDE{S <: SparseMatrixCSC}
    d::Int # Dimension of SPDE
    N::Int # Number of cells
    G::S 
    C_inv::S
    C::S
end

# Fonction principale pour résoudre le SPDE
function (m::spatialSPDE)(l::Real, σ::Real,ν)
    κ=l
    #κ=sqrt(8)/l
    ν=0.5
    dimension=2.0
    k = Symmetric(κ^2 *m.C + m.G)
    beta=(0.5+dimension/2.0)/2.0
    #Q = markov_precision_matrix(ν, κ, 2, m.C, m.G)
    #L=fractional_precision_matrix(k, m.C, beta, κ^2, 2, σ)
    L=k*m.C_inv*k
    #L=k

    Q= L .* σ^2  
    Qsym=Symmetric(Q)
    return  sparse(Qsym)   # solve SPDE
end

