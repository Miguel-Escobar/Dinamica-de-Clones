using Optim
using Plots
using LaTeXStrings
include("simulate_func.jl")
include("analysis_func.jl")

"""
Returns the values at sizes N of the log of the complementary cumulative distribution function of the system size distribution at time t.
This should be improved! As to not depend on parameters birth_rate, death_rate, and n₀.
"""
function logmodel(Ndata, params; t=13*24)
    δ = params[1]
    n_crit = params[2]
    birth_rate = 1/82
    death_rate = 1/(82*4)
    n₀ = 1
    times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, n_crit, δ, t, 10_000)
    n, ccdf = ccdfunc(t, times, populations)
    n = n[ccdf .> 0]
    ccdf = ccdf[ccdf .> 0]
    indexes = [findfirst(x -> x >= ncells, n) for ncells in Ndata]
    valid_indexes = indexes[indexes .!= nothing]
    
    if length(valid_indexes) > 0
        return [log.(ccdf[valid_indexes]); [-Inf for i in indexes if i ∉ valid_indexes]]
    else
        return [-Inf for i in indexes]
    end
    
end

"""
Define the optimization model to feed into Optim.jl. Params is the variable to minimize. Data is the
data to fit to, as an array of the form [Ndata, logdata], where N is the array of sizes and logdata is the
array of log of the complementary cumulative distribution function of the system size distribution at time t.
"""
function optmodel(params, data)
    Ndata = data[1]
    logdata = data[2]
    squares = (logmodel(Ndata, params; t=13*24) - logdata).^2
    returnable = sum(squares)
    return returnable
end

"""
Performs fitting of the model to data, hopefully. Currently barely working I guess :)
"""
function fit_model(Ndata, logdata, guess)
    sol = Optim.optimize(params -> optmodel(params, [Ndata, logdata]), [0., 10.], [3., 50.], guess, SAMIN(), Optim.Options(iterations=1000))
    return sol
end


function main()

    birth_rate = 1/82 # En 1/Horas
    death_rate = birth_rate/4 
    n₀ = 1 
    critical_size = 30.0
    δ = 2.0
    simulation_time = 13*24 # En horas (13 días)
    n_simulations = 10_000
    bin_width = 1

    # times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, critical_size, δ, simulation_time, n_simulations)
    # sample_times = LinRange(0, simulation_time, 100)
    # averages = simulation_averages(sample_times, times, populations)

    # Probamos el fiteo:

    timecode, ccdfdatas = read_ccdf_in_data("Data/20220222_idx.xlsm")

    ccdfdata = ccdfdatas[7]
    firstzero = findfirst(x -> x .≤ 0, ccdfdata)
    Nmin = 1
    Nmax = min(600, firstzero-1)
    logccdfdata = log.(ccdfdata[Nmin:Nmax])
    p = plot(Nmin:Nmax, logccdfdata, xlabel=L"N", ylabel=L"\log(1-P(N))", title="Data")
    
    ccdfdata = ccdfdatas[8]
    firstzero = findfirst(x -> x .≤ 0, ccdfdata)
    if firstzero === nothing
        firstzero = length(ccdfdata)
    end

    Nmin = 1
    Nmax = min(500, firstzero-1)
    logccdfdata = log.(ccdfdata[Nmin:Nmax])
    plot!(p, Nmin:Nmax, logccdfdata)
    display(p)
    # Ndata = Nmin:Nmax
    # guess = [0.2, 30]
    # fit = fit_model(Ndata, logccdfdata, guess)
    # p = plot(Ndata, [logccdfdata, logmodel(Ndata, fit.minimizer)], label=["Data" "Fit"], xlabel=L"N", ylabel=L"\log(1-P(N))", title="Fit of the model to data", legend=:topleft)
    # display(p)
    # return fit,p


end

