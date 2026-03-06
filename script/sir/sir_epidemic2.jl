include("./../../src/rwm.jl")
include("sir_data_generation.jl")
include("./../../src/online_picard_hmc.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
using DelimitedFiles
using StatsBase
norm2(x) = dot(x,x)

struct SIRTarget <: Target 
        λβ::Float64
        νβ::Float64
        λγ::Float64
        νγ::Float64
        xᵒ::Vector{Float64}
        N::Int64
    end


    


function potential(T::SIRTarget, x::Vector{Float64})
    d = length(x)
    # CHEK DOMAIN
    for i in eachindex(x)
        if x[i] > T.xᵒ[i]
            return Inf
        end
    end
    # COMPUTE F1
    F1 = 0.0
    istar = findmin(x)[2]
    for i in 1:d
        fi = 0
        if i != istar
            for j in 1:d
                if (j != i) && x[j] < x[i] < T.xᵒ[j]
                    fi += 1
                end
            end
            if fi == 0
                return Inf
            end
            F1 += log(fi)
        end
    end
    # COMPUTE F2
    F2 = 0.0
    for i in 1:d
        F2 += (T.N - d)*(T.xᵒ[i] - x[i])
        for j in 1:i
            if i != j
                F2 += min(T.xᵒ[i], x[j]) + min(T.xᵒ[j], x[i]) - 2*min(x[j], x[i]) 
            end
        end
    end
    F3 = 0.0
    for j in 1:d
        F3 += T.xᵒ[j] - x[j]
    end
    -F1 + (d + T.νβ - 1)*log(T.λβ + F2) + (d + T.νγ)*log(T.λγ + F3)
end

function check_likelihood(x, xᵒ)
    d = length(x)
    F1 = 0.0

    istar = findmin(x)[2]
    for i in 1:d
        f1 = 0
        if i != istar
            for j in 1:d
                if (j != i) && x[j] < x[i] < xᵒ[j]
                    f1 += 1
                end
            end
            if f1 < 1
                error("")
            end
            F1 += log(f1)
        end
    end
    println("ok")
end

function run_all()
    Random.seed!(1)
    N = 400
    β = 0.001
    γ = 0.15
    println("Reproduction number R = $(N*β/γ)")
    xtrue, xᵒ= generate_data(N, β, γ);
    

    d = length(xtrue)
    λβ = 0.001
    νβ = 1.0
    λγ = 0.001
    νγ = 1.0


    xtrue .> xᵒ
    T = SIRTarget(λβ, νβ, λγ, νγ, xᵒ, N)
    x0 = [ xᵒ[i] + log(rand())/0.05 for i in 1:d]
    check_likelihood(x0, xᵒ)


    function per1(mhat,m, s) 
        norm((mhat - m)./s)/sqrt(length(s))
    end

    function per2(shat, s) 
        norm((shat - s)./s)/sqrt(length(s))
    end
    @show length(xtrue)
    # LONG-RUN 
    println("LONG RUN")
    Nlong = 5*10^7
    h = 0.07
    M1, M2, ACC = sequential_random_walk_mh(T, xtrue, h, Nlong ÷ 5, Nlong)
    @show ACC
    μ_post = M1
    σ_post = sqrt.(M2 - (M1).^2)

    GC.gc() 
    # hrwm = 0.1
    hrwm = 0.07
    Niter = 2*10^6
    K = floor(Int, sqrt(d))
    println("RUNNING ONLINE PICARD RWM")
    res, W = online_picard_rwm(T, x0, hrwm, K, Niter, 0.0; save = true);

    @show res.acc
    println("G hat RWM = $(mean(res.G))")
    println("Mu = $(per1(mean(res.XX[:,Niter  ÷ 2:end], dims = 2)[:], μ_post, σ_post))")
    println("E  = $(per2(std(res.XX[:,Niter  ÷ 2:end], dims = 2)[:], σ_post))")
    writedlm("SIR4.csv", res.XX[:,Niter  ÷ 2:100:end], ',')
    res = nothing
    W = nothing
    GC.gc() 
    hmwg = 0.2
    println("RUNNING ONLINE PICARD MwG")
    res1, W = online_picard_orwm(T, x0, hmwg, K, Niter; save = true)
    @show res1.acc
    println("G hat MwG = $(mean(res1.G))")
    println("Mu = $(per1(mean(res1.XX[:,Niter  ÷ 2:end], dims = 2)[:], μ_post, σ_post))")
    println("E  = $(per2(std(res1.XX[:,Niter  ÷ 2:end], dims = 2)[:], σ_post))")
    writedlm("SIR5.csv", res1.XX[:,Niter  ÷ 2:100:end], ',')
    res = nothing
    W = nothing
    GC.gc() 
    println("RUNNING HMCD")
    L = 3
    N0 = Niter ÷ (L*d) + 2
    hh =  hmwg*sqrt(d)*abs.(randn(N0))
    vv = rand(Laplace(), d, N0)
    P, O, su, ar = online_picard_hmc(T, x0, vv, hh, K, L) 
    @show ar
    M = reduce(hcat, O.xx)
    @show size(M,2)
    @show Niter
    println("G hat HMCD = $(su)")
    println("Mu = $(per1(mean(M[:,Niter  ÷ 2:end], dims = 2)[:], μ_post, σ_post))")
    println("E  = $(per2(std(M[:,Niter  ÷ 2:end], dims = 2)[:], σ_post))")
    writedlm("SIR6.csv", M[:,(Niter  ÷ 2):100:end], ',')
    res = nothing
    W = nothing
    GC.gc() 
end

run_all()