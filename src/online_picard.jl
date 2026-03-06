using LinearAlgebra, Distributions

abstract type Target end

struct Innovation
    ZZ::Matrix{Float64}
    UU::Vector{Float64}
end

struct Out
    XX::Union{Nothing, Matrix{Float64}}
    G::Vector{Int64}
    acc::Vector{Float64}
end

struct PicardState 
    X::Matrix{Float64} # K +1
    B::Vector{Bool} # K
    Bc::Vector{Bool} # K 
    Z::Matrix{Float64} # K
    U::Vector{Float64} # K
end

function PicardState(K::Int64, x0::Vector{Float64})
    d = length(x0)
    X = repeat(x0, 1, K+1)
    B = zeros(Bool, K)
    Bc = zeros(Bool, K)
    Z = zeros(d, K)
    U = zeros(K)
    return PicardState(X, B, Bc, Z, U) 
end

# PicardState(10, randn(19))

# done sequentially
function solve_path!(P::PicardState, K)
    for i in 1:K
        P.X[:, i+1] =  P.X[:, i] + P.B[i]*P.Z[:, i]
    end
    return P
end

function B(V::Target, x::Vector{Float64}, z::Vector{Float64}, u::Float64)
    y = x + z
    if log(u) < potential(V, x) - potential(V, y) 
        return 1.0
    else
        return 0.0
    end
end


# done in parallel
function refresh!(P::PicardState, V::Target, K::Int64)
    for i in 1:K
        P.B[i] = B(V,  P.X[:, i], P.Z[:, i], P.U[i])
    end
    return P
end

function gain(P::PicardState, K::Int64, err::Float64)
    c = 0
    for i in 1:K
        if P.B[i] == P.Bc[i]
            c += 1
        elseif c/i < (1.0 - err)
            break
        end
    end
    return c
end


function move_forward!(P::PicardState, c::Int64, K::Int64)
    d = size(P.Z, 1)
    P.X[:, 1] = P.X[:, c+1]
    if c < K  # SHIFT 
        for i in 1: K - c  
            P.X[:, i + 1] = P.X[:, c + i + 1]
            P.B[i] = P.B[c + i]
        end
    end # DRAW
    if c >= 1
        for i in K - c + 1: K 
            P.X[:, i+1] = P.X[:, K - c + 1]
            P.B[i] = 0
        end
    end
    return P
end



function inner_online_picard!(V::Target, O::Out, P::PicardState, W::Innovation, (Lo, Up), err::Float64)
    N = size(W.ZZ,2) + 1
    P.Bc .= P.B
    K = Up - Lo + 1 
    P.Z[:, 1:K] = W.ZZ[:, Lo:Up]
    P.U[1:K] = W.UU[Lo:Up]
    # done in parallel
    P = refresh!(P, V, K)
    c = gain(P, K, err)
    push!(O.G, c)
    Ln = Lo + c
    # done sequentially
    P = solve_path!(P, K)
    if O.XX != nothing
        O.XX[:,Lo+1:Ln] = P.X[:, 2:c+1]
    end
    O.acc[1] += sum(P.B[1:c])/(N-1)
    P = move_forward!(P, c, K)
    return  O, P, Ln
end





function online_picard(V::Target, x0::Vector{Float64}, K::Int64, W::Innovation, err::Float64; save = false)
    d, N1 = size(W.ZZ)
    N = N1 + 1
    if d != length(x0)
        error("PROBLEM")
    end
    if save
        O = Out(zeros(d, N), Vector{Int64}(), [0.0])
        O.XX[:,1] = x0
    else
        O = Out(nothing,  Vector{Int64}(), [0.0])
    end
    P = PicardState(K, x0)
    Lo = 1
    perc = 10
    while true 
        Up = min(Lo + K - 1, N - 1)
        O, P, Lo = inner_online_picard!(V, O, P, W, (Lo, Up), err)
        if Lo == N 
            break
        end
        if Lo/N*100 > perc
            println("...$(perc)%...")
            perc += 10
        end
    end
    return O, W
end

function online_picard_rwm(V, x0::Vector{Float64}, h::Float64, K::Int64, N::Int64, err::Float64; save = false)
    d = length(x0)
    Z = randn(d, N-1)*h
    # @show var(Z, dims = 2)
    # error("")
    U = rand(N-1)
    W = Innovation(Z, U)
    online_picard(V, x0, K, W, err, save = save)
end

online_picard_rwm(V, x0, h, K, N) = online_picard_rwm(V, x0, h, K, N, 0.0)

function online_picard_orwm(V, x0, h, K, N, err::Float64; save = false)
    d = length(x0)
    Zi = randn(N-1)*sqrt(d)*h
    Z = zeros(d, N-1)
    for i in 1:N-1
        i2 = (i-1) % d + 1
        Z[i2,i] = Zi[i]
    end
    U = rand(N-1)
    W = Innovation(Z, U)
    online_picard(V, x0, K, W, err, save = save)
end


online_picard_orwm(V, x0, h, K, N; save = false) = online_picard_orwm(V, x0, h, K, N, 0.0, save = save)  





################# TESTING

# struct MyGaussTarget <: Target
#     Γ::Matrix{Float64}
#     μ::Vector{Float64}
# end

# function potential(T::MyGaussTarget, x::Vector{Float64})
#     return dot(x - T.μ, T.Γ*(x - T.μ))/2
#     # return dot(x,x)/2
# end 


# d = 10
# x = randn(d)
# μ = zeros(d) .+ 3
# T = MyGaussTarget(I(d)*2.0, μ)
# # T = MyGaussTarget()
# err = 0.0
# h = 0.5
# res, W = online_picard_rwm(T, x, h, 30, 100_000, err);
# res.acc
# XX = res.XX
# ZZ = W.ZZ

# mean(XX, dims = 2)
# var(XX, dims = 2)
# # sum((T.μ[1] - 1.96)/sqrt(T.Γ[1]) .< XX[1,:] .< (T.μ[1] + 1.96)/sqrt(T.Γ[1]))/size(XX, 2)
# sum(-1.96 .< XX[1,:] .< 1.96)/size(XX, 2)
# var(ZZ, dims = 2)
# UU = W.UU
# using Plots
# plot(XX[1,:])

# seqX = similar(XX)
# seqX[:, 1] = XX[:, 1]
# acc = 0
# for i in 1:size(ZZ,2)
#     Bi = B(T, seqX[:,i], ZZ[:,i], UU[i])
#     acc += Bi
#     seqX[:, i+1] = seqX[:, i] + Bi*ZZ[:,i]
# end
# plot!(seqX[1, :])



# x = randn(10)
# res, W = online_picard_orwm(T, x, 0.1, 30, 100);
# size(res.XX)
# res.G
