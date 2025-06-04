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

function per1(mhat,m, s) 
    norm((mhat - m)./s)/sqrt(length(s))
end

function per2(shat, s) 
    norm((shat - s)./s)/sqrt(length(s))
end

# performance(shat, s) = norm((shat - s))

function runall_bias_variance(errors)
    mm = zeros(length(errors) + 1, 2)
    ss = zeros(length(errors) + 1, 2)
    
    Random.seed!(0)
    d = 200
    γ0 = 1.0
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
    ho = 2*h
    Niter = 200*d
    K = d
    σ_post = sqrt.(diag(inv(Γ_post)))


    println("Exact Online Picard-RWM")
    res0, W = online_picard_rwm(T, x0, h, K, Niter, 0.0);
    mm[1, 1] = per1(mean(res0.XX, dims = 2)[:], μ_post, σ_post)
    ss[1, 1] = per2(std(res0.XX, dims = 2)[:], σ_post)
    @show res0.acc
    for (i,err) in enumerate(errors)
        println("Online Picard-RWM err = $(err)")
        res1, W = online_picard(T, x0, K, W, err)
        mm[i+1, 1] = per1(mean(res1.XX, dims = 2)[:], μ_post, σ_post)
        ss[i+1, 1] = per2(std(res1.XX, dims = 2)[:], σ_post)
        @show res1.acc
    end

    println("Exact Online Picard-MwG")
    res0, Wo = online_picard_orwm(T, x0, h, K, Niter);
    mm[1, 2] = per1(mean(res0.XX, dims = 2)[:], μ_post, σ_post)
    ss[1, 2] = per2(std(res0.XX, dims = 2)[:], σ_post)
    @show res0.acc
    for (i,err) in enumerate(errors)
        println("Online Picard-MwG err = $(err)")
        res1, Wo = online_picard(T, x0, K, Wo, err)
        mm[i+1, 2] = per1(mean(res1.XX, dims = 2)[:], μ_post, σ_post)
        ss[i+1, 2] = per2(std(res1.XX, dims = 2)[:], σ_post)
        @show res1.acc
    end
    mm, ss
end

errors = (0.025:0.025:0.2)
length(errors)
mm, ss = runall_bias_variance(errors)
using CSV, Tables
CSV.write("lr_mm.csv",  Tables.table(mm), writeheader=false)
CSV.write("lr_ss.csv",  Tables.table(ss), writeheader=false)
