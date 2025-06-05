using Plots, LaTeXStrings, CSV, DelimitedFiles
filename1 = "./out_csv/linear_regression/lr_d.csv"
g = readdlm(filename1, ',', Float64)

dd = [100, 200, 300, 500, 1000]
p1 = plot(dd, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , xlabel = L"d", 
    ylabel = L"\hat{G}", label = L"\bar Y")
plot!(dd, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(dd, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(dd, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(dd, g[:,5], marker = :circ, label = L"\bar Y_{o}")
plot!(dd, sqrt.(dd), label = L"\sqrt{d}", linewidth=2, ls=:dash)
plot!(dd, dd*0.33, label = L"d/3", linewidth=2, ls=:dash)


filename2 = "./out_csv/linear_regression/lr_k.csv"
g = readdlm(filename2, ',', Float64)
kk = [2, 5, 7, 10, 25, 50, 70, 100, 150, 200, 500, 1_000, 1_500]
p2 = plot(kk, g[:,1], marker = :circ, xaxis= :log, yaxis = :log , 
xlabel = L"K", ylabel = L"\hat{G}", label = L"\bar Y",legend=:topleft)
plot!(kk, g[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(kk, g[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(kk, g[:,4], marker = :circ, label = L"\bar Y_{20\%}")
plot!(kk, g[:,5], marker = :circ, label = L"\bar Y_{o}")
vline!([sqrt(500)], label = L"\sqrt{d}", linewidth=2, ls=:dash)
vline!([500], label = L"d", linewidth=2, ls=:dash)
plot!(kk, kk*0.66, label = L"2K/3", linewidth=2, ls=:dash)


using Plots.PlotMeasures
filename3 = "./out_csv/linear_regression/lr_mm.csv"
filename4 = "./out_csv/linear_regression/lr_ss.csv"
mm = readdlm(filename3, ',', Float64)
ss = readdlm(filename4, ',', Float64)
err = 0.0:0.025:0.2
p3 = plot(err, mm[:, 1], ylims = (0.0, 0.2),  marker = :circ, xlabel = L"ϵ", label = L"\mathcal{M}_\varepsilon"*" RWM",legend=:bottomright)
plot!(err, mm[:, 2], ls=:dash, marker = :circ, label = L"\mathcal{M}_\varepsilon"*" MwG")

plot!(err, ss[:, 1],  marker = :circ, label = L"\mathcal{E}_\varepsilon"*" RWM")
plot!(err, ss[:, 2],  marker = :circ, ls=:dash,  label = L"\mathcal{E}_\varepsilon"*" MwG")
pf = plot(p1,p2,p3, layout = (1,3), 
    size = (1000,500), 
    plot_title = "Linear Regression",
    left_margin=15px, bottom_margin=10px)

savefig("./linear_regression.pdf")