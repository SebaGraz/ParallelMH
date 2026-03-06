using Plots, LaTeXStrings
pythonplot()

function obs_speedup(Ghat, c, e)
    return Ghat/(1 + e/c)
end

function heatmap_obs_speedup(title, Ghat; cs= 10 .^range(-5, 1, length=200), es=10 .^range(-5, 1, length=100), cmap=:viridis, savepath=nothing)
    M = Array{Float64}(undef, length(cs), length(es))
    for (i, c) in enumerate(cs)
        for (j, e) in enumerate(es)
            M[i, j] = obs_speedup(Ghat, c, e)
        end
    end
    plt = heatmap(title = title, collect(es), collect(cs), M; xaxis = :log, yaxis = :log, xlabel=L"\epsilon", ylabel="c", colorbar_title = L"G_{obs}", c=cmap)
    if savepath !== nothing
        savefig(plt, savepath)
    end
    return plt, M, cs, es
end


### RWM 
Ghat1 = 29.75
cs = 10 .^range(-7, 1, length=400)
es = 10 .^range(-7, 1, length=400)
plt, M, cs, es = heatmap_obs_speedup("RWM effective speed-up", Ghat1; cs=cs, es=es, savepath="observed_speedup_heatmap.png")
plt
T = sqrt(500)
cs_line_2 = ((Ghat1 / T - 1)^(-1)) .* es
plot!(es[1:end], cs_line_2[1:end]; label= L"\hat G_{obs} = \sqrt{d}", linewidth=2, color=:black,  linestyle=:dash)
plot!(es, es, label = L"c = \epsilon",  linewidth=2, color=:red,  linestyle=:dash)
es_line = collect(es)
savefig(plt, "rwm_tradeoff.png")


Ghat1 = 128.0
cs = 10 .^range(-7, 1, length=400)
es = 10 .^range(-7, 1, length=400)
plt, M, cs, es = heatmap_obs_speedup("MwG effective speed-up",Ghat1; cs=cs, es=es, savepath="observed_speedup_heatmap.png")
plt
T = sqrt(500)
cs_line_2 = ((Ghat1 / T - 1)^(-1)) .* es
plot!(es[40:end], cs_line_2[40:end]; label= L"\hat G_{obs} = \sqrt{d}", linewidth=2, color=:black,  linestyle=:dash)
plot!(es, es, label = L"c = \epsilon",  linewidth=2, color=:red,  linestyle=:dash)
savefig(plt, "mwg_tradeoff.png")


println("Saved heatmap to observed_speedup_heatmap.png")


X = collect(10 .^(range(1,5)))
Y = collect(10 .^(range(1,5)))
Z = randn(5,5)
plt = heatmap(X, Y, Z, xaxis = :log, yaxis = :log)
plot!(X)
plot!(X, Y, line= :blue, linewidth = 2,  xaxis = :log, yaxis = :log)