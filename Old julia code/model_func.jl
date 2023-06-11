using Optim
using Plots
using LaTeXStrings
include("simulate_func.jl")
include("analysis_func.jl")

"""
Returns the values at sizes N of the log of the complementary cumulative distribution function of the system size distribution at time t.
"""
function logmodel(Ndata, params; t=13*24, n₀=1, birth_rate=1/82, death_rate=1/(82*4), n_simulation_steps=10_000)
    δ = params[1]
    n_crit = params[2]
    times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, n_crit, δ, t, n_simulation_steps)
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
Generalizes logmodel so that it can accept data at different times, and returns the model at those times. This is sort of messy code I think, but I'll leave it for now.
"""
function generalized_logmodel(timesandNdatas, params)
    times = timesandNdatas[1]
    Ndatas = timesandNdatas[2]
    model_at_times = [logmodel(Ndata, params; t=t) for (t, Ndata) in zip(times, Ndatas)]
    return model_at_times
end



"""
Define the optimization model to feed into Optim.jl, now taking into account different times of measurement. Params 
is the variable to minimize. data_at_times is an array which encodes the time of the measurement and the sizes measured at that time in it's first coordinate,
and the log ccdf of the sizes measured, at the time of measurement. 
"""
function optmodel(params, data_at_times)
    timesandNdatas = data_at_times[1]
    logdata = data_at_times[2]  # Should be an array of the same dimensions as generalized_logmodel(timeandNdata, params). Keep this in mind!
    squares = [element.^2 for element in (generalized_logmodel(timesandNdatas, params) - logdata)]
    returnable = sum([sum(sq) for sq in squares])
    println(returnable)
    return returnable
end



"""
Performs fitting of the model to data, hopefully.
"""
function fit_model(timesandNdata, logdata, guess; iter=10_000)
    sol = Optim.optimize(params -> optmodel(params, [timesandNdata, logdata]), [0., 10.], [3., 50.], guess, SAMIN(), Optim.Options(iterations=iter))
    return sol
end




function main()
    timecode, dist_per_timecode = read_ccdf_in_data("Data/20220222_idx.xlsm")
    times = [0, 3, 6, 13]
    evendata = [dist_per_timecode[i] for i in 2 .* (1:4)]
    odddata = [dist_per_timecode[i-1] for i in 2 .* (1:4)]
    maxN = min(length(evendata[1]), 200) # Tunear para ver q onda
    minN = 10

    # Messy, pero me construye logdata:

    restrictedevendata = [ed[minN:maxN] for ed in evendata]

    firstzero_indices = [findfirst(x -> x == 0, ed) for ed in restrictedevendata]
    firstzero_indices[firstzero_indices .== nothing] .= maxN-minN

    logevendata = [log.(ed[1:firstzero-1]) for (ed, firstzero) in zip(restrictedevendata, firstzero_indices) if firstzero > 1]

    # Now to construct timesandNdata:

    # timesandevenNdata = [times, [minN:firstzero-1 for firstzero in firstzero_indices if firstzero != nothing]] 

    # logevendata
    #fit = fit_model(timesandevenNdata, logevendata, guess; iter=100)
    






end

