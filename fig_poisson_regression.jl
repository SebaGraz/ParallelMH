using Plots, LaTeXStrings, CSV, DelimitedFiles
filename1 = "./out_csv/poisson/poisson_r_d.csv"
g = readdlm(filename1, ',', Float64)

dd = [30, 50, 100, 200, 300, 400, 500]
p1 = plot(dd, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , 
    xlabel = L"d", ylabel = L"\hat{G}", label = L"\bar Y")
plot!(dd, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(dd, g[:,5], marker = :circ, label = L"\bar Y_{MwG}")
plot!(dd, sqrt.(dd), label = L"\sqrt{d}", linewidth=2, ls=:dash)
plot!(dd, dd*0.25, label = L"d/4", linewidth=2, ls=:dash)


filename2 = "./out_csv/poisson/poisson_r_k.csv"
g = readdlm(filename2, ',', Float64)
kk =  [5, 10, 15, 25, 50, 100, 200, 400, 600]
p2 = plot(kk, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , 
xlabel = L"K", ylabel = L"\hat{G}", label = L"\bar Y",legend=:topleft)
plot!(kk, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(kk, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(kk, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(kk, g[:,5], marker = :circ, label = L"\bar Y_{MwG}")
vline!([sqrt(200)], label = L"\sqrt{d}", linewidth=2, ls=:dash)
vline!([200], label = L"d", linewidth=2, ls=:dash)
plot!(kk, kk*0.66, label = L"2K/3", linewidth=2, ls=:dash)


using Plots.PlotMeasures
filename3 = "./out_csv/poisson/poisson_r_mm.csv"
filename4 = "./out_csv/poisson/poisson_r_ss.csv"
mm = readdlm(filename3, ',', Float64)
ss = readdlm(filename4, ',', Float64)
err = 0.0:0.05:0.2
p3 = plot(err, mm[:, 1],  ylims = (0.0, 0.2), marker = :circ, xlabel = L"r", 
            label = L"\mathcal{M}_r"*" RWM",
            legend=:bottomright)
plot!(err, mm[:, 2], ls=:dash, marker = :circ, label = L"\mathcal{M}_r"*" MwG")

plot!(err, ss[:, 1],  marker = :circ, label = L"\mathcal{E}_r"*" RWM")
plot!(err, ss[:, 2],  marker = :circ, ls=:dash,  label = L"\mathcal{E}_r"*" MwG")
pf = plot(p1,p2,p3, layout = (1,3), 
    size = (1000,500), 
    plot_title = "Poisson Regression",
    left_margin=15px, bottom_margin=10px)

savefig("./poisson_regression.pdf")