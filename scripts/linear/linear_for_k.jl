# SETTING
# K = d, d = 10, 1000
# A_{i,j} standard normal
# sample size n = 2d

include("./../../src/online_picard.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
using StatsBase
norm2(x) = dot(x,x)


struct GaussianTarget <: Target 
    μ::Vector{Float64}
    Γ::Matrix{Float64}
end

function potential(T::GaussianTarget, x::Vector{Float64})
   dot((x - T.μ), T.Γ*(x - T.μ))/2
end



function runall_k(kk)
    gain = zeros(length(kk), 5)
    d = 500
    Random.seed!(0)
    γ0 = 1.0
    Random.seed!(0)
    n = d*5
    A = randn(n,d)./sqrt(d)
    @show n, d
    σ2 = 1.0
    xtrue = randn(d)
    y=zeros(Float64,n)
    for j in 1:n
        y[j]=randn()*sqrt(σ2)+ dot(A[j,1:end], xtrue)
    end
    Λ_post = (A'A + I(d)*γ0)
    x̂ =  inv(A'A)*A'*y
    μ_post = inv(Λ_post)*((A'A)*x̂)
    Γ_post = Λ_post/σ2
    T = GaussianTarget(μ_post, Γ_post)
    x0 = μ_post
    h = 1.0/sqrt(d)
    Niter = 10_000
    for i in eachindex(kk)
        K = kk[i]
        @show K
        println("Exact Online Picard-RWM")
        res0, W = online_picard_rwm(T, x0, h, K, Niter, 0.0);
        gain[i, 1] = mean(res0.G)
        @show res0.acc
        err1 = 0.05
        println("Online Picard-RWM err = $(err1)")
        res1, W = online_picard(T, x0, K, W, err1)
        gain[i, 2] = mean(res1.G)

        err2 = 0.1
        println("Online Picard-RWM err = $(err2)")
        res2, W = online_picard(T, x0, K, W, err2)
        gain[i, 3] = mean(res2.G)

        err3 = 0.2
        println("Online Picard-RWM err = $(err3)")
        res3, W = online_picard(T, x0, K, W, err3)
        gain[i, 4] = mean(res3.G)

        println("Exact Online Picard-ORWM")
        res4, W = online_picard_orwm(T, x0, h, K, Niter);
        gain[i, 5] = mean(res4.G)

        @show res0.acc, res1.acc, res2.acc, res3.acc
        @show res4.acc
    end 
    gain 
end



kk = [2, 5, 7, 10, 25, 50, 70, 100, 150, 200, 500, 1_000, 1_500]
g = runall_k(kk)
using CSV, Tables
CSV.write("lr_k.csv",  Tables.table(g), writeheader=false)



error("")
dd = kk
p1 = plot(dd, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , 
xlabel = L"K", ylabel = L"\mathbb{E}[G]", label = L"\bar Y",legend=:topleft)
plot!(dd, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(dd, g[:,5], marker = :circ, label = L"\bar Y_{o}")
vline!([sqrt(500)], label = L"\sqrt{d}", linewidth=2, ls=:dash)
vline!([500], label = L"\sqrt{d}", linewidth=2, ls=:dash)
plot!(dd, dd*0.66, label = L"2K/3", linewidth=2, ls=:dash)
