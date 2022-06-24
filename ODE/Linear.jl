using JLD2
using Statistics
using LinearAlgebra
using PyPlot

# https://link.springer.com/content/pdf/10.1007/s00332-015-9258-5.pdf

function Linear_Map(x::Array{FT,1}, A::Array{FT,2}, N_t::Int) where {FT<:AbstractFloat, IT<:Int}
    N_x = length(x0)
    xs = zeros(N_x, N_t+1)
    xs[:,1] = x0
    
    for i=1:N_t
        xs[:,i+1] = A * xs[:,i]
    end
    
    return xs
end







