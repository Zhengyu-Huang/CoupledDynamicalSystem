using JLD2
using Statistics
using LinearAlgebra
using PyPlot

# https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

function Lotka_Volterra(x::Array{FT,1}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    α, β, δ, γ = θ
    return [α*x[1] - β*x[1]*x[2] ; δ*x[1]*x[2] - γ*x[2]] 
end

# integrate ODE with 4th order Runge-Kutta method
function compute_ODE_RK4(f::Function, x0::Array{FT,1}, θ::Array{FT,1}, Δt::FT, N_t::IT) where {FT<:AbstractFloat, IT<:Int}
    N_x = length(x0)
    xs = zeros(N_x, N_t+1)
    xs[:,1] = x0
    
    for i=1:N_t
        k1 = Δt*f(xs[:,i], θ)
        k2 = Δt*f(xs[:,i] + k1/2, θ)
        k3 = Δt*f(xs[:,i] + k2/2, θ)
        k4 = Δt*f(xs[:,i] + k3, θ)
        xs[:,i+1] = xs[:,i] + k1/6 + k2/3 + k3/3 + k4/6
    end
    
    return xs
end




f = Lotka_Volterra
x0 = [2.0; 1.0]
θ = [2.0/3.0  ; 4.0/3.0 ; 1.0  ; 1.0]
Δt, N_t = 0.1, 10
xs = compute_ODE_RK4(f, x0, θ, Δt, N_t)

ts = Δt*Array(0:N_t)

plot(ts, xs[1, :], label="Prey")
plot(ts, xs[2, :], label="Predator")
legend()
using NPZ
npzwrite("LV.npy", xs)

include("../Algorithms/DMD.jl")
# dmd = DMD()
# fit!(dmd, xs)
# @info dmd._λ, dmd._basis, dmd._w, dmd._Ã


# dmd = DMD(;n_rank=0, exact=true, total=false)
# fit!(dmd, xs)
# @info dmd._λ, dmd._basis, dmd._w, dmd._Ã

kernel_fun = PolyKernel(2, 1.0)
dmd = KDMD(kernel_fun; n_rank=6, exact=true, total=true)
fit!(dmd, xs)


@show   sum(dmd._modes)
@show   sum(dmd._λ)
@show   sum(dmd._PhiX)
@show   sum(dmd._Ã)
@show   sum(dmd._Ĝ)
@show   sum(dmd._Â)

@show   dmd._λ
@show   dmd._modes

