using LinearAlgebra, Random
using Optim 
norm2(x) = dot(x,x)
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)

abstract type Target end

struct Innovation
    ZZ::Matrix{Float64}
    UU::Vector{Float64}
end

struct Out
    XX::Matrix{Float64}
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
    ar = 0.0
    for i in 1:K
        if P.B[i] == P.Bc[i]
            c += 1
            ar += P.B[i]
        elseif c/i < (1.0 - err)
            break
        end
    end
    return c, ar/c
end


function inner_online_picard_tails!(V::Target, P::PicardState, W::Innovation, (Lo, Up), err::Float64)
    K = Up - Lo + 1 
    P.Z[:, 1:K] = W.ZZ[:, Lo:Up]
    P.U[1:K] = W.UU[Lo:Up]
    P = refresh!(P, V, K)
    P = solve_path!(P, K)
    P.Bc .= P.B
    P = refresh!(P, V, K)
    c, ar = gain(P, K, err)
    return c, ar
end

function online_picard_tails(V::Target, x0::Vector{Float64}, K::Int64, W::Innovation, err::Float64)
    d, N1 = size(W.ZZ)
    N = N1 + 1
    if d != length(x0)
        error("PROBLEM")
    end
    P = PicardState(K, x0)
    Lo = 1
    Up = Lo + K - 1
    c, ar = inner_online_picard_tails!(V, P, W, (Lo, Up), err)
    return c, ar
end


function online_picard_rwm_tails(V, x0::Vector{Float64}, h::Float64, h2, K::Int64, err::Float64)
    d = length(x0)
    Z = randn(d, K)*h
    U = rand(K)
    W = Innovation(Z, U)
    g, ar = online_picard_tails(V, x0, K, W, err)
    Zi = randn(K)*h2*sqrt(d)
    Z2 = zeros(d, K)
    for i in 1:K
        Z2[i,i] = Zi[i]
    end
    W2 = Innovation(Z2, U)
    g2, ar2 = online_picard_tails(V, x0, K, W2, err)
    g, ar, g2, ar2 
end


# function online_picard_orwm(V, x0, h, K, N, err::Float64)
#     d = length(x0)
#     Zi = randn(N-1)*sqrt(d)*h
#     Z = zeros(d, N-1)
#     for i in 1:N-1
#         i2 = (i-1) % d + 1
#         Z[i2,i] = Zi[i]
#     end
#     U = rand(N-1)
#     W = Innovation(Z, U)
#     online_picard(V, x0, K, W, err)
# end
# nline_picard_orwm(V, x0, h, K, N) = online_picard_orwm(V, x0, h, K, N, 0.0)  










struct LogisticTarget <: Target 
    A::Matrix{Float64}
    y::Vector{Float64}
    γ0::Float64
end

function potential(T::LogisticTarget, x::Vector{Float64})
    T.γ0*dot(x,x)/2 - sum(T.y .* lsigmoid.(T.A*x) + (1 .- T.y) .* lsigmoid.(-T.A*x))
end



function runall()
    Random.seed!(0)
    h = 0.5
    d = K = 200
    γ0 = 1.0
    n = d*10
    A = randn(n,d)./sqrt(d)
    @show size(A)
    xtrue = randn(d)
    y = rand(n) .< sigmoid.(A*xtrue)*1.0
    T = LogisticTarget(A, y, γ0)
    res = optimize(x -> potential(T, x), xtrue, LBFGS())
    h = 0.5/sqrt(d)
    h2 = h*2.0
    x0 = randn(d)
    x0 ./= norm(x0)
    tot = 10
    scales = collect(range(0.0, 2000.0, length=50))
    G = zeros(Float64, length(scales)) 
    AR = zeros(Float64, length(scales))
    G2 = zeros(Float64, length(scales)) 
    AR2 = zeros(Float64, length(scales))
    for _ in 1:tot
        Z = randn(d, K)*h
        U = rand(K)
        W = Innovation(Z, U)
        Zi = randn(K)*h2*sqrt(d)
        Z2 = zeros(d, K)
        for i in 1:K
            Z2[i,i] = Zi[i]
        end
        W2 = Innovation(Z2, U)
        for (i, scale) in enumerate(scales) 
            xstart = res.minimizer + x0*scale
            g, ar = online_picard_tails(T, xstart, K, W, 0.0)
            g2, ar2 = online_picard_tails(T, xstart, K, W2, 0.0)
            G[i] += g/tot
            AR[i] += ar/tot
            G2[i] += g2/tot
            AR2[i] += ar2/tot
        end
    end
    println("Acceptance rates: $(AR) and $(AR2)")
    G, AR, G2, AR2, scales
end
g, ar, g2, ar2, scales = runall()


using Plots, LaTeXStrings
plot(scales, g, ylabel = L"L^{(1)}", marker = :circ, label = "RWM", xlabel = L"\|x_0 - x^\star \|", title = "Convergence in the tails", ylims = (0,210))
plot!(scales, g2, marker = :circ, label = "MwG")
hline!([200], line = (1, :dash), color = "red", label = L"K")
savefig("tail_convergence.pdf")