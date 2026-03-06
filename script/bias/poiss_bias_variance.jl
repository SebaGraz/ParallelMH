# SETTING
# K = d, d = 10, 1000
# A_{i,j} standard normal
# sample size n = 2d

include("./../../src/rwm.jl")
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

function per1(mhat,m, s) 
    norm((mhat - m)./s)/sqrt(length(s))
end

function per2(shat, s) 
    norm((shat - s)./s)/sqrt(length(s))
end

function runall_bias_variance(errors)
    mm = zeros(length(errors) + 1, 2)
    ss = zeros(length(errors) + 1, 2)
    Random.seed!(0)
    # d = 200
    d = 100
    n = d*10
    A = randn(n,d)./sqrt(d)
    @show size(A)
    γ0 = 1.0 #prior precision
    # Data from the model
    xtrue = randn(d)
    y = [rand(Poisson(exp(dot(xtrue,A[i,:])))) for i in 1:n]
    T = PoissTarget(A, y, γ0)
    x0 = xtrue
    h = 0.6/sqrt(d)

    # LONG RUN FOR ESITMATION MEAN AND VARIANCES
    println("LONG RUN")
    Nlong = 10^6
    M1, M2, ACC = sequential_random_walk_mh(T, xtrue, h, Nlong ÷ 5, Nlong)
    @show ACC
    x0 = M1
  
    # Niter = 200*d
    Niter = 1000*d
    K = d
    μ_post = M1
    σ_post = sqrt.(M2 - (M1).^2)

    println("Exact Online Picard-RWM")
    res0, W = online_picard_rwm(T, x0, h, K, Niter, 0.0,  save = true);
    mm[1, 1] = per1(mean(res0.XX, dims = 2)[:], μ_post, σ_post)
    ss[1, 1] = per2(std(res0.XX, dims = 2)[:], σ_post)
    for (i,err) in enumerate(errors)
        println("Online Picard-RWM err = $(err)")
        res1, W = online_picard(T, x0, K, W, err,  save = true)
        mm[i+1, 1] = per1(mean(res1.XX, dims = 2)[:], μ_post, σ_post)
        ss[i+1, 1] = per2(std(res1.XX, dims = 2)[:], σ_post)
        @show res1.acc
    end
    
    println("Exact Online Picard-MwG")
    res0, Wo = online_picard_orwm(T, x0, h, K, Niter,  save = true);
    mm[1, 2] = per1(mean(res0.XX, dims = 2)[:], μ_post, σ_post)
    ss[1, 2] = per2(std(res0.XX, dims = 2)[:], σ_post)
    @show res0.acc
    for (i,err) in enumerate(errors)
        println("Online Picard-MwG err = $(err)")
        res1, Wo = online_picard(T, x0, K, Wo, err,  save = true)
        mm[i+1, 2] = per1(mean(res1.XX, dims = 2)[:], μ_post, σ_post)
        ss[i+1, 2] = per2(std(res1.XX, dims = 2)[:], σ_post)
        @show res1.acc
    end
    mm, ss
end


mm, ss = runall_bias_variance(0.025:0.025:0.2)
using CSV, Tables
CSV.write("poisson_r_mm.csv",  Tables.table(mm), writeheader=false)
CSV.write("poisson_r_ss.csv",  Tables.table(ss), writeheader=false)
