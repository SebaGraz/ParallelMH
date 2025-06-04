include("./../../src/online_picard.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
using StatsBase
norm2(x) = dot(x,x)


struct PoissTarget <: Target 
    A::Matrix{Float64}
    y::Vector{Float64}
    γ0::Float64
end

function potential(T::PoissTarget, x::Vector{Float64})
    res = 0.0
    for i in eachindex(T.y)
        res -= T.y[i]*dot(x, T.A[i, :]) - exp(dot(x, T.A[i, :]))
    end
    res += T.γ0*dot(x,x)/2
    return res
end


function runall_d(dd)
    gain = zeros(length(dd), 5)
    for (i, d) in enumerate(dd)
        Random.seed!(0)
        n = d*10
        A = randn(n,d)./sqrt(d)
        @show size(A)
        γ0 = 1.0 #prior precision
        # Data from the model
        xtrue = randn(d)
        y = [rand(Poisson(exp(dot(xtrue,A[i,:])))) for i in 1:n]
        T = PoissTarget(A, y, γ0)
        x0 = xtrue
        Niter =  10^4
        K = d
        h = 0.6/sqrt(d)


        # no error online picard
        println("Exact Online Picard-RWM")
        res0, W = online_picard_rwm(T, x0, h, K, Niter, 0.0);
        gain[i, 1] = mean(res0.G)

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




dd = [30, 50, 100, 200, 300, 400, 500]
g = runall_d(dd)
using CSV, Tables
CSV.write("poisson_r_d.csv",  Tables.table(g), writeheader=false)
error("")
