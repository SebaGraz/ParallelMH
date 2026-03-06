# See https://link.springer.com/article/10.1007/s11222-005-4074-7
using Random


function simulate_epidemic(β, γ, N; verbose = true)
    println("population size = $(N )")
    I = zeros(N)
    I[1] = 1
    println("infected initial population = $(sum(I))")
    d = length(I)
    R = zeros(d)
    t = 0.0
    Ξ = Vector{Tuple{Float64, Int64, Int64}}()
    for j in eachindex(I)[I .== 1]
        push!(Ξ, (t, j, 1))
    end
    while true 
        first_t = Inf
        type = 0
        j = 0
        for i in 1:d
            if I[i] == 1 
                if R[i] == 1
                    continue
                else # I -> R
                    t0 = -log(rand())/γ
                    if t0 < first_t
                        j = i
                        type = 2
                        first_t = t0
                    end
                end
            else # S -> I
                βi = (sum(I) - sum(R))*β
                 t0 = -log(rand())/βi
                if t0 < first_t
                    j = i
                    type = 1
                    first_t = t0
                end
            end        
        end
        t += first_t
        if I[j] == 0
            I[j] = 1
        elseif R[j] == 0
            R[j] = 1
        else
            error("")
        end
        push!(Ξ, (t, j, type))
        if (sum(I) - sum(R)) == 0
            println("last removed individual: $(t)")
            println("total number of infected $(sum(I))")
            break
        end
    end
    return Ξ
end

function generate_data(N, β, γ)
    res = simulate_epidemic(β, γ, N)
    jj = getindex.(Set(getindex.(res, 2)),1)

    d = length(jj)
    x = zeros(d)
    xᵒ = zeros(d)
    for i in 1:d
        x[i] = getindex.(res, 1)[getindex.(res,2) .== jj[i]][1]
        xᵒ[i] = getindex.(res, 1)[getindex.(res,2) .== jj[i]][2]   
    end
    x, xᵒ
end

