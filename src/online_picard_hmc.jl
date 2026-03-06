### ADD RANDOM STEPSIZE
using Distributions, Random, LinearAlgebra
f(x) = x
f2(x) = x^2

struct PicardStateHMC 
    X::Matrix{Float64} # K +1
    V::Matrix{Float64} # K +1
    fx::Vector{Float64} # K 
    fxc::Vector{Float64} # K
    fv::Vector{Float64} # K 
    fvc::Vector{Float64} # K
    idx::Vector{Int64} # K
end

struct OutputHMC
    xx::Vector{Vector{Float64}} # samples
end

# Convert (j,l,k) -> idx
function triple_to_index(j, l, k, L, d)
    return k + d * ((l - 1) + L * (j - 1))
end

# Convert idx -> (j,l,k)
function index_to_triple(idx, L, d)
    idx0 = idx - 1
    k = (idx0 % d) + 1
    l = ((idx0 ÷ d) % L) + 1
    j = (idx0 ÷ (d * L)) + 1
    return j, l, k
end


function online_picard_hmc(T, x0, vv, hh, K, L::Int64;
            verbose = true,
            ) 
    d = length(x0)
    N0 = size(vv, 2) - 1
    N = L*d*N0
    perc = 10
    acc = 0
    # @show N
    x = copy(x0)
    v0 = vv[:,1]
    v = copy(v0)
    P = PicardStateHMC(repeat(x, 1, K+1),
                 repeat(v, 1, K+1), 
                 zeros(Float64,  K), 
                 zeros(Float64,  K), 
                 zeros(Float64,  K), 
                 zeros(Float64,  K),
                 Int.(1:K))
    O = OutputHMC([x0,])
    count = 0   
    Lo = 1   
    while true
        count += 1
        # println("Picard iteration $count")
        P.fxc .= P.fx
        P.fvc .= P.fv
       
        # update fx and fv: this can be parallelized
        P = update_picard_incrementsHMC!(P, T, hh, L, vv)
        
        # update X and V
        P = solve_pathHMC!(P, L, d, vv)

        # gain
        c, acc = gainHMC!(P, acc)

        #save 
        for i in 1:c
            push!(O.xx, P.X[:, i+1])
        end

        # Move forward
        P.idx .= P.idx .+ c
        P.X[:, 1] = P.X[:, c+1]
        P.V[:, 1] = P.V[:, c+1]
        if c < K 
            for i in 1:K-c
                P.X[:,i+1] = P.X[:,i+c+1]
                P.V[:,i+1] = P.V[:,i+c+1]
                P.fx[i] = P.fx[i+c]
                P.fv[i] = P.fv[i+c]
            end
        end
        if c >= 1
            for i in K-c+1:K
                j, l, k = index_to_triple(P.idx[i], L, d)
                P.X[:, i+1] = P.X[:, i] 
                P.V[:, i+1] = P.V[:, i] 
                P.fx[i] = 0.0
                P.fv[i] = 0.0
            end
        end
        Lo += c
        if verbose && Lo/N*100 > perc
            println("...$(perc)%...")
            perc += 10
        end
        # @show Lo
        if Lo >= N
            # @show Lo
            break
        end

    end
    return P, O, Lo/count, acc/(Lo-1)
end

function update_picard_incrementsHMC!(P, T, hh, L, VV)
    K = length(P.fx)
    d = size(P.X, 1)
    for i0 in 1:K
        j, l, i = index_to_triple(P.idx[i0], L, d)
        h = hh[j]
        if l == 1 && i == 1
            w = copy(VV[:, j])
        else
            w = copy(P.V[:, i0])
        end
        y = copy(P.X[:, i0])
        δxi = h*sign(w[i])
        y[i] += δxi
        ΔU = potential(T, y) - potential(T, P.X[:, i0])
        if abs(w[i]) > ΔU
            # accept
            P.fx[i0] = δxi
            P.fv[i0] = -sign(w[i])*ΔU
        else
            # reject 
            P.fx[i0] = 0.0
            P.fv[i0] = -2*w[i]
        end 
    end
    return P
end

function solve_pathHMC!(P, L, d, VV)
    K = length(P.fx)
    for i0 in 1:K
        j, l, i = index_to_triple(P.idx[i0], L, d)
        if l == 1 && i == 1
            # @show j
            P.V[:, i0] = VV[:,j]
        end
        P.X[:, i0+1] = P.X[:, i0] 
        P.V[:, i0+1] = P.V[:, i0] 
        P.X[i, i0+1] += P.fx[i0]
        P.V[i, i0+1] += P.fv[i0]
    end
    return P
end

function gainHMC!(P, acc)
    c = 0
    K = length(P.fx)
    d = size(P.X, 1)
    for i0 in 1:K
        if P.fx[i0] == P.fxc[i0] && P.fv[i0] == P.fvc[i0]
            if norm(P.fx[i0])/d > eps()
               acc += 1 
            end
            c += 1
        end
    end
    return c, acc
end


function sequential_dhmc(T, y0,  VV, hh, burnin, L)
    N0 = size(VV, 2)
    d = length(y0)
    x = copy(y0)
    ar = 0
    m = zeros(d)
    m2 = zeros(d)
    perc = 10
    N = N0*L*d
    v = zeros(d)
    resx = zeros(d, N + 1)
    resv = zeros(d, N + 1)
    resx[:,1] = copy(x)
    resv[:,1] = copy(VV[:,1])
    for j in 1:N # initilising trajectory (can be done in parallel)
        x, v, ar = update!(T, VV, x, v, j, hh, ar, L, d)
        resx[:,j+1] = copy(x)
        resv[:,j+1] = copy(v)
        if j > burnin
            jn = j + 1 - burnin
            m = (jn - 1)/jn * m + f.(x)/jn
            m2 = (jn - 1)/jn * m2 + f2.(x)/jn
        end
        if j/N*100 > perc
            println("...$(perc)%...")
            perc += 10
        end
   
    end
    return resx, resv, m, m2, ar/(N-1)
end

function  updateHMC!(T, VV, x, v, idx, hh, ar, L, d)
    j, l, i = index_to_triple(idx, L, d)
    h = hh[j]
    if l == 1 && i == 1
        v .= VV[:, j] 
    end
    y = copy(x)
    y[i] += h*sign(v[i]) 
    ΔU = potential(T, y) - potential(T, x)
    if abs(v[i]) > ΔU
        ar += 1
        x .= y
        v[i] -= sign(v[i])*ΔU
    else
        v[i] = -v[i]
    end
    return x,v,ar
end


# abstract type Target end

# struct MyGaussTarget <: Target
#     Γ::Matrix{Float64}
#     μ::Vector{Float64}
# end

# function potential(T::MyGaussTarget, x::Vector{Float64})
#     return dot(x - T.μ, T.Γ*(x - T.μ))/2
#     # return dot(x,x)/2
# end 

# function runall()
#     d = 10
#     μ = zeros(d) .- 0.0
#     σ2 = 1.0
#     T = MyGaussTarget(I(d)/σ2, μ)
#     Random.seed!(1234)
#     N0 = 100_000
#     x0 = randn(d)
#     vv = rand(Laplace(), d, N0)
#     hh = 0.25 .+ rand(N0)*0.5
#     L = 5
#     K = 20
#     P, O, su, ar = Picard(T, x0, vv, hh, K, L) 
#     x0, vv, P, O, hh, L, N0, d, T, su, ar
# end
# x0, vv, P0, O, h0, L0, N0, d, T, su, ar = runall()
# error("")

# using Plots
# plot(getindex.(O.xx,1))
# plot(getindex.(O.vv,1))
# resx, resv, m1, m2, acc = sequential_dhmc(T, x0, vv, h0, 1, L0)
# μ = zeros(d) .- 0.0
# σ2 = 1.0
# m1 - μ
# m2 - m1.^2
# @show sqrt(sum(abs2, (m1 - μ)/sqrt(σ2)))
# @show sqrt(sum(abs2, (sqrt.(m2 - m1.^2) .- sqrt(σ2))./sqrt(σ2)))

# using Plots
# l = length(O.xx)
# plot(getindex.(O.xx,1)[6000:10000])
# plot(resx[1,:][6000:10000])



