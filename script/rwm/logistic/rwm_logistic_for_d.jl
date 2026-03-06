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
        y = rand(n) .< sigmoid.(A*xtrue)*1.0
        T = LogisticTarget(A, y, γ0)
        x0 = xtrue
        Niter =  10^4
        K = d
        K2 = ceil(Int, sqrt(d))
        h = 1.4/sqrt(d)


        # no error online picard
        println("Exact Online Picard-RWM K = $(K2)")
        res0, W = online_picard_rwm(T, x0, h, K2, Niter, 0.0);
        gain[i, 1] = mean(res0.G)

        println("Online Picard-RWM  K = $(K)")
        res01, W = online_picard(T, x0, K, W, 0.0)
        gain[i, 2] = mean(res01.G)
        err1 = 0.05

        println("Online Picard-RWM err = $(err1)")
        res1, W = online_picard(T, x0, K, W, err1)
        gain[i, 3] = mean(res1.G)

        err2 = 0.1
        println("Online Picard-RWM err = $(err2)")
        res2, W = online_picard(T, x0, K, W, err2)
        gain[i, 4] = mean(res2.G)

        err3 = 0.2
        println("Online Picard-RWM err = $(err3)")
        res3, W = online_picard(T, x0, K, W, err3)
        gain[i, 5] = mean(res3.G)

      

        @show res0.acc, res1.acc, res2.acc, res3.acc
    end
    gain 
end 




dd = [30, 50, 100, 200, 300, 400, 500]
g = runall_d(dd)
using CSV, Tables
CSV.write("rwm_log_r_d.csv",  Tables.table(g), writeheader=false)
error("")

g
using Plots, LaTeXStrings
p1 = plot(dd, g[:, 1], xaxis = :log, yaxis = :log, marker = :circ, xlabel = L"d", label = "OP, "*L"K = \sqr{d}")
plot!(dd, g[:, 2], marker = :circ, label =  "OP, "*L"K = d")
plot!(dd, g[:, 3], marker = :circ, label = "AOP, "*L"{r = 5\%},"*L"\,K = d")
plot!(dd, g[:, 4], marker = :circ, label = "AOP, "*L"{r = 10\%},"*L"\,K = d")
plot!(dd, g[:, 5], marker = :circ, label = "AOP, "*L"{r = 20\%},"*L"\,K = d")

plot!(dd, sqrt.(dd), label = L"\sqrt{d}", ls=:dash)
plot!(dd, dd*0.15, label = L"d", ls=:dashdot)
ylabel!(L"\mathbb{E}[G]")
title!("Logistic regression - Online-Picard RWM")
savefig("logistic_regression_d.pdf")


