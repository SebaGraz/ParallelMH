using Random, LinearAlgebra
using RandomMatrices, Distributions
using Plots.PlotMeasures

Random.seed!(0)
function B(U, x, z, u, args...)
    y = x + z
    if log(u) < U(x, args...) - U(y, args...) 
        return 1.0
    else
        return 0.0
    end
end

function visual_moving_window(U, y0, h, N::Int64, args...)
    d = length(y0)
    u = rand(N-1)
    z = randn(d, N-1)*h
    return visual_moving_window_pit_random_walk_mh(U, y0, N::Int64, u, z, args...)
end



function visual_moving_window_pit_random_walk_mh(U, y0, N::Int64, u, z, args...)
    pp = Vector{Int64}()
    d = length(y0)
    BB = zeros(Bool, N -1)
    BBold = zeros(Bool, N -1)
    Y = zeros(d, N)
    Y[:,1] .= y0
    p = 1
    for j in 2:N # initilising trajectory (can be done in parallel)
        Y[:, j] = Y[:, j-1]  # + z[:, j-1] 
    end
    OUT = [copy(Y),]
    for i in p:N # this must be done sequentially
        BBold .= BB
        first = true
        count = 0
        for k in p:N-1  # this step can be done in parallel
            BB[k] = B(U, Y[:, k], z[:, k], u[k], args...)
            if first == true && BB[k] == BBold[k]
                count += 1
            else 
                first = false
            end
        end
        p = p + count
        push!(pp, copy(p))
        for j in 2:N # can be definetly optimised
            Y[:, j] = Y[:, j - 1] + BB[j-1]*z[:, j-1]
        end
        push!(OUT, copy(Y))
        if p == N 
            return OUT, pp, sum(BB)/(N-1)
        end
    end
    return OUT, pp, sum(BB)/(N-1)
end

   








using Random, LinearAlgebra, Distributions
function runall()
    Random.seed!(1)
    d = 100
    h = 2.0/sqrt(d)
    N = 1000
    y0 = randn(d) .- 10
    U(x) = dot(x,x)/2
    YY, pp, ar = visual_moving_window(U, y0, h, N)  
    @show ar
    YY, pp
end


YY, pp = runall()
error("")
using Plots, LaTeXStrings
# p1
p1 = plot(YY[end][1,:],YY[end][2,:],  link=:all,  ylims = (-10,5), yaxis = L"x_2",
     alpha = 0.2, color = :black, lw=2, label = "Fixed Point",legend=false)
p1 = plot!(YY[2][1,:],YY[2][2,:], colorbar = false,
    color = cgrad(:autumn1, rev=true), 
    line_z = (1:length(YY[end][2,:])), 
    alpha = 0.7, lw=2, 
    label = "",
    title = L"X^{(1)}")
plot!(YY[2][1,1:pp[2]],YY[2][2,1:pp[2]], label = L"X^{(1)}_{0:G_1}", lw=2, color = :black, ls=:dash,  alpha = 0.7)



ii = 3
p2 = plot(YY[end][1,:],YY[end][2,:], ylims = (-10,5), alpha = 0.2, color = :black,
         lw=2, label = "Fixed Point",legend=false)
plot!(YY[ii][1,:],YY[ii][2,:], colorbar = false,
        color = cgrad(:autumn1, rev=true), 
        line_z = (1:length(YY[end][2,:])), 
        alpha = 0.7, lw=2, 
        label = "",
        title = "\$X^{($(ii-1))}\$")
plot!(YY[ii][1,1:pp[ii]],YY[ii][2,1:pp[ii]], label = "\$X^{($(ii-1))}_{0:G_{$(ii-1)}}\$", ls=:dash, lw=2, color = :black, alpha = 0.7)


ii = 11
p3 = plot(YY[end][1,:],YY[end][2,:], ylims = (-10,5), xaxis = L"x_1",
     yaxis = L"x_2", alpha = 0.2, color = :black, lw=2, label = "Fixed Point",legend=false)
plot!(YY[ii][1,:],YY[ii][2,:], colorbar = false,
        color = cgrad(:autumn1, rev=true), 
        line_z = (1:length(YY[end][2,:])), 
        alpha = 0.7, lw=2, 
        label = "",
        title = "\$X^{($(ii-1))}\$")
plot!(YY[ii][1,1:pp[ii]],YY[ii][2,1:pp[ii]], label = "\$X^{($(ii-1))}_{0:G_{$(ii-1)}}\$", ls=:dash,lw=2, color = :black, alpha = 0.7)



ii = 14
p4 = plot(YY[end][1,:],YY[end][2,:], ylims = (-10,5), xaxis = L"x_1", 
    alpha = 0.2, color = :black, lw=2, label = "Fixed Point",legend=false)
plot!(YY[ii][1,:],YY[ii][2,:], colorbar = false,
        color = cgrad(:autumn1, rev=true), 
        line_z = (1:length(YY[end][2,:])), 
        alpha = 0.7, lw=2, 
        label = "",
        title = "\$X^{($(ii-1))}\$")
plot!(YY[ii][1,1:pp[ii]],YY[ii][2,1:pp[ii]], label = "\$X^{($(ii-1))}_{0:G_{$(ii-1)}}\$", ls=:dash, lw=2, color = :black, alpha = 0.7)
l = @layout[grid(2,2) a{0.05w}] 


p = plot(p1,p2,p3, p4, 
    heatmap((0:0.01:1).*ones(101,1), size = (900,500), color = cgrad(:autumn1, rev=true), legend=:none, xticks=:none, yticks=(1:10:101, string.(0:100:1000))), 
    layout=l,  
    plot_title = "Picard recursion",
    left_margin=15px)

savefig("./ilustration_picard.pdf")

error("")
