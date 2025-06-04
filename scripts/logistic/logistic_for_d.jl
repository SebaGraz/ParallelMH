include("./../../src/online_picard.jl")
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
        h = 1.4/sqrt(d)


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




dd = [30, 50, 100, 200, 300, 400, 500, 1000]
g = runall_d(dd)
using CSV, Tables
CSV.write("log_r_d.csv",  Tables.table(g), writeheader=false)
error("")
g

using Plots, LaTeXStrings
p1 = plot(dd, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , xlabel = L"d", ylabel = L"\mathbb{E}[G]", label = L"\bar Y")
plot!(dd, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(dd, g[:,5], marker = :circ, label = L"\bar Y_{o}")
plot!(dd, sqrt.(dd), label = L"\sqrt{d}", ls=:dash)
plot!(dd, dd*0.15, label = L"d", ls=:dashdot)



using Plots, LaTeXStrings
p1 = plot(dd, res0, marker = :circ, xlabel = "d", label = L"\bar Y")
plot!(dd, res1, marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, res2, marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, res3, marker = :circ, label = L"\bar Y_{20\%}")
# plot!(dd, res4, marker = :circ, label = L"\bar Y_{(o)}")
ylabel!(L"\mathbb{E}[G]")
title!("Linear regression - Online-Picard")
plot!(dd, sqrt.(dd), label = L"\sqrt{d}", ls=:dash)
savefig("linear_regression_1.pdf")


p2 = plot(dd, sdv[5,:], marker = :circ, xlabel = L"d", label = L"\bar Y")
plot!(dd, sdv[1,:], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, sdv[2,:], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, sdv[3,:], marker = :circ, label = L"\bar Y_{20\%}")
# plot!(dd, res4, marker = :circ, label = L"\bar Y_{(o)}")
ylabel!(L"\mathcal{E}")
title!("Linear regression - Online-Picard")
savefig("linear_regression_2.pdf")

p2 = plot(dd, mm[5,:], marker = :circ, xlabel = L"d", label = L"\bar Y")
plot!(dd, mm[1,:], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, mm[2,:], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, mm[3,:], marker = :circ, label = L"\bar Y_{20\%}")
# plot!(dd, res4, marker = :circ, label = L"\bar Y_{(o)}")
ylabel!(L"\mathcal{E}")
title!("Linear regression - Online-Picard")
savefig("linear_regression_2.pdf")
