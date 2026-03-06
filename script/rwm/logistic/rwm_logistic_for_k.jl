# SETTING
# K = d, d = 10, 1000
# A_{i,j} standard normal
# sample size n = 2d

include("./../../../src/online_picard.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
using StatsBase
norm2(x) = dot(x,x)
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)


struct LogisticTarget <: Target 
    A::Matrix{Float64}
    y::Vector{Float64}
    γ0::Float64
end

function potential(T::LogisticTarget, x::Vector{Float64})
    T.γ0*dot(x,x)/2 - sum(T.y .* lsigmoid.(T.A*x) + (1 .- T.y) .* lsigmoid.(-T.A*x))
end

struct GaussianTarget <: Target 
    μ::Vector{Float64}
    Γ::Matrix{Float64}
end

function potential(T::GaussianTarget, x::Vector{Float64})
   dot((x - T.μ), T.Γ*(x - T.μ))/2
end


function runall_k(kk)
    gain = zeros(length(kk), 4)
    d = 200
    Random.seed!(0)
    n = d*10

    A = randn(n,d)./sqrt(d)
    @show size(A)
    γ0 = 1.0 #prior precision
    # Data from the model
    xtrue = randn(d)

    y = rand(n) .< sigmoid.(A*xtrue)*1.0
    T = LogisticTarget(A, y, γ0)
    x0 = xtrue
    h = 1.4/sqrt(d)
    Niter = 50*d
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

        @show res0.acc, res1.acc, res2.acc, res3.acc
    end 
    gain 
end



kk = [5, 10, 15, 25, 50, 100, 200, 400, 600]
g = runall_k(kk)
using CSV, Tables
CSV.write("rwm_log_r_k.csv",  Tables.table(g), writeheader=false)

