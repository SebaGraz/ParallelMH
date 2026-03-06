using Plots.PlotMeasures, LaTeXStrings, Plots
using CSV, DataFrames, Tables

srcdir = "/Users/zanella/Documents/bocconi/PinT/ParallelMH/out/bias/"
df1 = DataFrame(CSV.File(srcdir*"lin_r_mm.csv"; header=false))
mm = Matrix(df1)
df2 = DataFrame(CSV.File(srcdir*"lin_r_ss.csv"; header=false))
ss = Matrix(df2)

err = 0.0:0.025:0.2
p1 = plot(err, mm[:, 1], ylims = (0.0, 0.2),  marker = :circ, xlabel = L"r", label = L"\mathcal{M}_r"*" RWM",legend=:topleft)
plot!(err, mm[:, 2], ls=:dash, marker = :circ, label = L"\mathcal{M}_r"*" MwG")

plot!(err, ss[:, 1],  marker = :circ, label = L"\mathcal{E}_r"*" RWM")
plot!(err, ss[:, 2],  marker = :circ, ls=:dash,  label = L"\mathcal{E}_r"*" MwG")
title!("Linear regression")



df3 = DataFrame(CSV.File(srcdir*"log_r_mm.csv"; header=false))
mm = Matrix(df3)
df4 = DataFrame(CSV.File(srcdir*"log_r_ss.csv"; header=false))
ss = Matrix(df4)

p2 = plot(err, mm[:, 1], ylims = (0.0, 0.2),  marker = :circ, xlabel = L"r", label = L"\mathcal{M}_r"*" RWM",legend=:topleft)
plot!(err, mm[:, 2], ls=:dash, marker = :circ, label = L"\mathcal{M}_r"*" MwG")

plot!(err, ss[:, 1],  marker = :circ, label = L"\mathcal{E}_r"*" RWM")
plot!(err, ss[:, 2],  marker = :circ, ls=:dash,  label = L"\mathcal{E}_r"*" MwG")
title!("Logistic regression")


df5 = DataFrame(CSV.File(srcdir*"poisson_r_mm.csv"; header=false))
mm = Matrix(df5)
df6 = DataFrame(CSV.File(srcdir*"poisson_r_ss.csv"; header=false))
ss = Matrix(df6)


p3 = plot(err, mm[:, 1], ylims = (0.0, 0.2),  marker = :circ, xlabel = L"r", label = L"\mathcal{M}_r"*" RWM",legend=:topleft)
plot!(err, mm[:, 2], ls=:dash, marker = :circ, label = L"\mathcal{M}_r"*" MwG")

plot!(err, ss[:, 1],  marker = :circ, label = L"\mathcal{E}_r"*" RWM")
plot!(err, ss[:, 2],  marker = :circ, ls=:dash,  label = L"\mathcal{E}_r"*" MwG")
title!("Poisson regression")
p = plot(p1,p2,p3, plot_title = "Bias Approximate Online Picard", layout = (1,3), size = (800, 400))
savefig("./bias.pdf")