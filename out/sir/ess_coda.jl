using RCall, DelimitedFiles, Statistics

# Ensure the coda package is installed in R
R"""
if (!requireNamespace("coda", quietly = TRUE)) {
    install.packages("coda", repos = "https://cloud.r-project.org")
}
library(coda)
"""
dir_file = "/Users/zanella/Documents/bocconi/PinT/ParallelMH/out/sir"
# SIR1 = readdlm(dir_file*"/SIR1.csv", ',')'
# SIR2 = readdlm(dir_file*"/SIR2.csv", ',')'
# SIR3 = readdlm(dir_file*"/SIR3.csv", ',')'
SIR4 = readdlm(dir_file*"/SIR4.csv", ',')'
SIR5 = readdlm(dir_file*"/SIR5.csv", ',')'
SIR6 = readdlm(dir_file*"/SIR6.csv", ',')'
# Pass samples to R and compute ESS using coda::effectiveSize
# @rput SIR1  # move Julia array to R
# @rput SIR2  # move Julia array to R
# @rput SIR3  # move Julia array to R
@rput SIR4  # move Julia array to R
@rput SIR5  # move Julia array to R
@rput SIR6  # move Julia array to R
println("computing ESS")
R"""
# ess_value1 <- effectiveSize(SIR1)
# ess_value2 <- effectiveSize(SIR2)
# ess_value3 <- effectiveSize(SIR3)
ess_value4 <- effectiveSize(SIR4)
ess_value5 <- effectiveSize(SIR5)
ess_value6 <- effectiveSize(SIR6)

"""
# @rget ess_value1
# @rget ess_value2
# @rget ess_value3  # bring result back to Julia
@rget ess_value4
@rget ess_value5
@rget ess_value6
# println("Small SIR Estimated ESS RWM = ", mean(ess_value1))
# println("Small SIR ESS MwG = ", mean(ess_value2))
# println("Small SIR ESS DHMC = ", mean(ess_value3))
println("Large SIR Estimated ESS RWM = ", mean(ess_value4))
println("Large SIR ESS MwG = ", mean(ess_value5))
println("Large SIR ESS DHMC = ", mean(ess_value6))

# println("Small SIR Estimated min ESS RWM = ", minimum(ess_value1))
# println("Small SIR ESS min MwG = ", minimum(ess_value2))
# println("Small SIR ESS min DHMC = ", minimum(ess_value3))
println("Large SIR Estimated ESS RWM = ", minimum(ess_value4))
println("Large SIR ESS MwG = ", minimum(ess_value5))
println("Large SIR ESS DHMC = ", minimum(ess_value6))
