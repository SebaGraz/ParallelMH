include("./../../src/online_picard.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
using StatsBase
norm2(x) = dot(x,x)



struct PREMTarget <: Target
    n::Int64 
    σ2η::Float64
    σ2μ::Float64
    Data::Vector{Vector{Float64}}
end



function potential(T::PREMTarget, x::Vector{Float64})
    μ = x[1]
    η = x[2:end]
    res = 0.0
    for (i, yi) in enumerate(T.Data)
        res += T.n*(-η[i]*mean(yi) +  exp(η[i]))
    end
    res += norm2(η .- μ)/(2*T.σ2η) + norm2(μ)/(2*T.σ2μ)
    return res
end


function simulate_data(I, n, μ)
    Data = Vector{Vector{Float64}}()
    η = randn(I) .+ μ
    for i in eachindex(η)
        push!( Data, rand(Poisson(exp(η[i])), n))
    end
    return Data, η
end




function runall_d(II)
    gain = zeros(length(II), 5)
    for (i, I) in enumerate(II)
        Random.seed!(0)
        n = 5
        μ = 5.0
        data, η = simulate_data(I, n, μ)
        σ2η =  1.0
        σ2μ = 10.0^2
        T = PREMTarget(n, σ2η, σ2μ, data)

        
        x0 = [μ, η...]
        d = length(x0)
        Niter =  10^5
        K = d
        h = 0.1/sqrt(d)


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
        ho = 2*h
        res4, W = online_picard_orwm(T, x0, ho, K, Niter);
        gain[i, 5] = mean(res4.G)

        @show res0.acc, res1.acc, res2.acc, res3.acc
        @show res4.acc
    end
    gain 
end 




II = [5 ,10, 30, 50, 100, 200]
g = runall_d(II)
using CSV, Tables
CSV.write("prem_r_d.csv",  Tables.table(g), writeheader=false)
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
