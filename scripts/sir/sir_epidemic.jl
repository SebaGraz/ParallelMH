include("./../../src/rwm.jl")
include("sir_data_generation.jl")
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using Test
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





Random.seed!(3)
N = 200
β = 0.001
γ = 0.15
println("Reproduction number R = $(N*β/γ)")
xtrue, xᵒ= generate_data(N, β, γ);
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
Nlong = 10^7
h = 0.09
M1, M2, ACC = sequential_random_walk_mh(T, xtrue, h, Nlong ÷ 5, Nlong)
@show ACC
μ_post = M1
σ_post = sqrt.(M2 - (M1).^2)



# Niter = 1_000_000
Niter = 10^6
K = floor(Int, sqrt(d))
println("RUNNING ONLINE PICARD RWM")
# h = 0.2
res, W = online_picard_rwm(T, x0, h, K, Niter, 0.0);
@show res.acc
println("G hat RWM = $(mean(res.G))")
println("Mu = $(per1(mean(res.XX[:,Niter  ÷ 2:end], dims = 2)[:], μ_post, σ_post))")
println("E  = $(per2(std(res.XX[:,Niter  ÷ 2:end], dims = 2)[:], σ_post))")
println("RUNNING ONLINE PICARD MwG")
res1, W = online_picard_orwm(T, x0, h, K, Niter)
@show res1.acc
println("G hat MwG = $(mean(res1.G))")
println("Mu = $(per1(mean(res1.XX[:,Niter  ÷ 2:end], dims = 2)[:], μ_post, σ_post))")
println("E  = $(per2(std(res1.XX[:,Niter  ÷ 2:end], dims = 2)[:], σ_post))")


error("")
res.acc
mean(res.G)

function functional_trace(XX, xᵒ, N, xtrue)
    d, n = size(XX)
    res = zeros(2,n)
    for i in 1:n
        x = XX[:, i]
         # COMPUTE F2
        F2 = 0.0
        for j in 1:d
            F2 += (N - d)*(xᵒ[j] - x[j])
            for k in 1:d
                F2 += min(xᵒ[j], x[k]) -  min(x[j], x[k]) 
            end
        end
        F3 = 0.0
        for j in 1:d
            F3 += xᵒ[j] - x[j]
        end
        res[1,i] = F2
        res[2,i] = F3
    end
    A = 0.0
    for j in 1:d
        A += (N - d)*(xᵒ[j] - xtrue[j])
        for k in 1:d
            A += min(xᵒ[j], xtrue[k]) -  min(xtrue[j], xtrue[k]) 
        end
    end
    B = 0.0
    for j in 1:d
       B += xᵒ[j] - xtrue[j]
    end
    res, A, B
end

function sample_β_and_γ(AB, νβ, λβ, νγ, λγ, d)
    n = size(AB, 2)
    bg = similar(AB)
    for i in 1:n
        # update Beta
        bg[1, i] = rand(Gamma(d + νβ - 1, 1/(AB[1,i] + λβ)))
        # update Gamma
        bg[2,i] = rand(Gamma(d + νγ, 1/(AB[2,i] + λγ)))
    end
    return bg
end


AB, Atrue, Btrue = functional_trace(res.XX, xᵒ, N, xtrue)
ABs = AB[:,1:100:end]
bg = sample_β_and_γ(ABs, νβ, λβ, νγ, λγ, length(x0))

using Plots, Colors, LaTeXStrings
using Plots.PlotMeasures
p1 = plot(ABs[1,:], ABs[2,:], colorbar = false,
    color = cgrad(:haline, rev=true), line_z = (1:size(ABs,2)),
    xaxis = L"A(x^\star)", yaxis =L"B(x^\star)", label = "",
    left_margin=30px)

scatter!([Atrue], [Btrue], color = :red, label = L"(A(x),B(x))")


xx = 0.0:0.0001:0.003
yy2 = N*xx
yy1 = zeros(length(xx))
p2 = plot(xx, yy2, fillrange = yy1, fillalpha = 0.15, c = 1, label = L"R>1", legend = :topleft)
plot!(bg[1,:],bg[2,:],
    color = cgrad(:haline, rev=true), line_z = (1:size(ABs,2)),
    xaxis = L"\beta", yaxis =L"\gamma", label = ""
    )

scatter!([β], [γ], color = :red, label = L"(\beta^\star,\gamma^\star)")
p = plot(p1, p2,  plot_title = "Online-Picard RWM - SIR model ", size = (900,500))
savefig("sir.png")