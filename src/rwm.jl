include("./online_picard.jl")

f(x) = x
f2(x) = x^2

function sequential_random_walk_mh(T, y0, h, burnin, N)
    d = length(y0)
    x = copy(y0)
    Ux = potential(T, x)
    ar = 0
    m = zeros(d)
    m2 = zeros(d)
    perc = 10
    for j in 2:N # initilising trajectory (can be done in parallel)
        y = x + h*randn(d)
        Uy = potential(T, y) 
        if log(rand()) < Ux - Uy
            Ux = Uy
            x = y
            ar += 1
        end
        if j > burnin
            jn = j - burnin
            m = (jn - 1)/jn * m + f.(x)/jn
            m2 = (jn - 1)/jn * m2 + f2.(x)/jn
        end
        if j/N*100 > perc
            println("...$(perc)%...")
            perc += 10
        end
    end
    return m, m2, ar/(N-1)
end


struct MyGaussTarget <: Target
    Γ::Matrix{Float64}
    μ::Vector{Float64}
end

function potential(T::MyGaussTarget, x::Vector{Float64})
    return dot(x - T.μ, T.Γ*(x - T.μ))/2
    # return dot(x,x)/2
end 

# d = 10
# x = randn(d)
# μ = zeros(d) .+ 3
# T = MyGaussTarget(I(d)*2.0, μ)
# h = 1.0
# m1,m2,ac = sequential_random_walk_mh(T, x, h, 100, 10000)
# ac
# m1 - μ
# m2 - m1.^2