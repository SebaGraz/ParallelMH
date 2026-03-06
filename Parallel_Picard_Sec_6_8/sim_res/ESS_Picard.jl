using MAT
vars = matread("./sim_res/res.mat")

# Accedi a una variabile specifica (es. una matrice 'A')
X = vars["XX_out"]
X = exp.(X)
using RCall, DelimitedFiles, Statistics

# Ensure the coda package is installed in R
R"""
if (!requireNamespace("coda", quietly = TRUE)) {
    install.packages("coda", repos = "https://cloud.r-project.org")
}
library(coda)
"""

# Pass samples to R and compute ESS using coda::effectiveSize
@rput X  # move Julia array to R

println("computing ESS")
R"""
ess_value <- effectiveSize(X)
"""
@rget ess_value
println("Precision medicine mean ESS RWM = ", mean(ess_value))
