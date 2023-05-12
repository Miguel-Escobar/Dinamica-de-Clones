using Optim
using Plots
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
    sol = Optim.optimize(params -> optmodel(params, [Ndata, logdata]), [0., 10.], [3., 50.], guess, SAMIN(), Optim.Options(iterations=10_000))
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

    Ndata = 10:500
    ydata = logmodel(Ndata, [δ, critical_size]; t = 13*24)
    guess = [0.5, 11]
    fit = fit_model(Ndata, ydata, guess)

    # fit = curve_fit(logmodel, N, ydata, [δ / 2, critical_size/2], lower=[0.001, 0.001], upper=[10.0, 50.0])
    
    # println(coef(fit), [δ, critical_size])
    # println(standard_errors(fit))

    test = plot(Ndata, ydata, label="Data")
    plot!(test, Ndata, logmodel(Ndata, fit.minimizer), label="Fit")
    display(test)
    println(fit)

    # plot1 = canvas()
    # plot!(plot1, x -> n₀*exp((birth_rate-death_rate)*x), 0, simulation_time, label=L"$n_0e^{(r-m)x}$")
    # plot!(plot1, sample_times, averages, label="Simulation Average")

    # N, distribution = system_size_distribution(simulation_time-1, bin_width, times, populations)
    # ccdf = 1 .- cumsum(distribution)

    # plot2 = plot(N[ccdf .> 0],
    #             ccdf[ccdf.> 0],
    #             #xlim=(0, 500),
    #             #ylim=(1e-2, 1),
    #             xlabel="Population Size [Cell number]",
    #             ylabel="CCDF",
    #             yscale=:log10,
    #             st=:steppost,
    #             label=nothing,
    #         )
    # display(plot1)
    # savefig(plot1, "Comparison.png")
    # savefig(plot2, "Distribution.png")

    # time_array = LinRange(0, simulation_time, 180)
    # animate_ccdf(time_array, times, populations)


end

