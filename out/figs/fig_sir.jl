using DelimitedFiles
filename = "/Users/zanella/Documents/bocconi/PinT/ParallelMH/out/sir/"
sir = Array{Float64}(readdlm(filename*"SIR1.csv", ','))

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


AB, Atrue, Btrue = functional_trace(sir, xᵒ, N, xtrue)
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