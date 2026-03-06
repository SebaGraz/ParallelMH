using Plots, LaTeXStrings, Measures
using CSV, DataFrames, Tables
srcdir = "/Users/zanella/Documents/bocconi/PinT/ParallelMH/out/logistic/"
df = DataFrame(CSV.File(srcdir*"mwg_log_r_d.csv"; header=false))
g1 = Matrix(df)


dd = [30, 50, 100, 200, 300, 400, 500]
kk = [5, 10, 15, 25, 50, 100, 200, 400, 600]

p1 = plot(dd, g1[:, 1], xaxis = :log, yaxis = :log, marker = :circ, xlabel = L"d", label = "OP, "*L"K = \sqr{d}")
plot!(dd, g1[:, 2], marker = :circ, label =  "OP, "*L"K = d")
plot!(dd, g1[:, 3], marker = :circ, label = "AOP, "*L"{r = 5\%},"*L"\,K = d")
plot!(dd, g1[:, 4], marker = :circ, label = "AOP, "*L"{r = 10\%},"*L"\,K = d")
plot!(dd, g1[:, 5], marker = :circ, label = "AOP, "*L"{r = 20\%},"*L"\,K = d")

# plot!(dd, res4, marker = :circ, label = L"\bar Y_{(o)}")
ylabel!(L"\hat G")
# title!("Linear regression - Online-Picard RWM")
plot!(dd, sqrt.(dd), label = L"\sqrt{d}", ls=:dash)
plot!(dd, dd./3, label = L"d/3", ls=:dash)


df = DataFrame(CSV.File(srcdir*"rwm_log_r_k.csv"; header=false))
g2 = Matrix(df)
p2 = plot(kk, g2[:,1], marker = :circ, xaxis= :log, yaxis = :log , 
xlabel = L"K", ylabel = L"\hat G", label = L"\bar Y",legend=:topleft)
plot!(kk, g2[:,2], marker = :circ, label = L"\bar Y_{5\%}")
plot!(kk, g2[:,3], marker = :circ, label = L"\bar Y_{10\%}")
plot!(kk, g2[:,4], marker = :circ, label = L"\bar Y_{20\%}")
vline!([sqrt(200)], label = L"\sqrt{d}", linewidth=2, ls=:dash)
vline!([200], label = L"d", linewidth=2, ls=:dash)
p = plot(p1,p2, plot_title = "Logistic regression - Online-Picard MwG", size=(800,400), bottom_margin = 3mm, left_margin = 3mm)
savefig("log_mwg.pdf")