using Random
using Plots
using ProgressBars
using LaTeXStrings
using LsqFit
using Optimization
using Optim


"""
Performs a stochastic simulation of a birth-death process with birth rate that changes at a critical size
using the Gillespie algorithm.

# Arguments:
- n₀ = initial population size.
- birth_rate.
- death_rate.
- critical_size: the population size at which the birth rate changes.
- δ: the change in birth rate given as (1+δ)*birth_rate.
- simulation_time

# Returns:
- t: an array of times at which an event took place.
- population: an array of population sizes at times t.
"""
function modified_birth_death(n₀, birth_rate, death_rate, critical_size, δ, simulation_time; max_steps=100_000_000)

    times = Float64[0.]
    population = Int[n₀]
    nstep = 0 

    while nstep < max_steps && times[end] < simulation_time
        if population[end] < critical_size
            birth_propensity = population[end]*birth_rate
            death_propensity = population[end]*death_rate   
        else
            birth_propensity = population[end]*(birth_rate*(1 + δ))
            death_propensity = population[end]*death_rate
        end

        α = birth_propensity + death_propensity

        r₁ = 1 - rand()
        r₂ = 1 - rand()

        τ = (1/α)*log(1/r₁)

        if r₂*α < birth_propensity
            push!(population, population[end] + 1)
        elseif population[end] == 0
            push!(population, population[end])
        else
            push!(population, population[end] - 1)
        end
        
        push!(times, times[end] + τ)
        nstep += 1
    end

    if nstep == max_steps
        print("Warning: max number of steps reached. Simulation did not reach final time.")
    end

    return times, population
end

"""
Performs n_simulations of a birth-death process with birth rate that changes at a critical size.

# Arguments
- n₀ = initial population size
- birth_rate
- death_rate
- critical_size: the population size at which the birth rate changes.
- δ: the change in birth rate given as (1+δ)*birth_rate.
- simulation_time
- n_simulations

# Returns
- times: an array of arrays of times at which an event took place.
- populations: an array of arrays of population sizes at times t.
"""
function modified_birth_death_processes(n₀, birth_rate, death_rate, critical_size, δ, simulation_time, n_simulations)
    times = Vector{Float64}[]
    populations = Vector{Int}[]
    for i in 1:n_simulations
        t, p = modified_birth_death(n₀, birth_rate, death_rate, critical_size, δ, simulation_time)
        push!(times, t)
        push!(populations, p)
    end
    print("Simulations completed.")
    return times, populations
end


"""
Calculates the average population size at each time in sample_times.

# Arguments

- sample_times: an array of times at which to calculate the average population size.
- times: an array of arrays of times at which an event took place.
- populations: an array of arrays of population sizes at times t.

# Returns

- averages: an array of average population sizes at times in sample_times.
"""
function simulation_averages(sample_times, times, populations)
    averages = zeros(length(sample_times))
    for (i, time) in enumerate(sample_times)
        for j in 1:length(populations) 
            averages[i] += element_at_time(time, times[j], populations[j])
        end
        averages[i] /= length(populations)
    end
    return averages
end


"""
Helper function, returns the population size at time t, even when t is not the time at which an event took place.
Time is an array of times at which an event took place, population is an array of population sizes at times in time.
"""
function element_at_time(t, time, population)
    index = findfirst(x -> x > t, time)
    if index !== nothing
        return population[max(index-1, 1)]
    else
        return population[end]
    end
end


"""
Returns the system size distribution at time t. Note that this isn't precisely a histogram.
This should be improved!
"""
function system_size_distribution(t, bin_width::Int,times, populations)
    n = length(populations)
    max_size = maximum(x->maximum(x), populations) + 1
    distribution = zeros(max_size)
    for i in 1:n
        distribution[element_at_time(t, times[i], populations[i])+1] += 1
    end
    nbins = floor(Int, max_size/bin_width)
    binned_distribution = zeros(nbins)
    for i in 1:(nbins-1)
        binned_distribution[i] = sum(distribution[i*bin_width:(i+1)*bin_width])
    end
    return [(k-1)*bin_width for k in 1:nbins], binned_distribution./sum(binned_distribution)
end

"""
Returns the complementary cumulative distribution function of the system size distribution at time t.
"""
function ccdfunc(t, times, populations)
    n, dist = system_size_distribution(t, 1, times, populations)
    return n, 1 .- cumsum(dist)
end


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
    # if returnable == Inf
    #     returnable = 1e10
    # end  # Recomendación de Github Copilot que ojalá no necesite.
    return returnable
end

"""
Performs fitting of the model to data, hopefully. Currently barely working I guess :)
"""
function fit_model(Ndata, logdata, guess)
    # problem = OptimizationProblem(optmodel, guess, [Ndata, logdata])
    # sol = solve(problem,)
    # return sol.minimizer

    # sol = optimize(params -> optmodel(params, [Ndata, logdata]), guess)

    #optimize(params -> optmodel(params, [Ndata, logdata]), guess, SimulatedAnnealing())

    sol = Optim.optimize(params -> optmodel(params, [Ndata, logdata]), [0., 10.], [4., 50.], guess, SAMIN(), Optim.Options(iterations=500))
    return sol
end






"""
Just creates a canvas to plot nicely (enough).
"""
function canvas()
    return plot(xlabel="Time [Hrs]", ylabel="Size [Cell Count]", yscale=:log10, legend=:bottomright, dpi=600)
end


"""
Let's try to make an animation of the system size distribution.
"""
function animate_ccdf(time_array, times, populations)
    bin_width = 1
    animation = @animate for t in time_array
        n, dist = system_size_distribution(t, bin_width, times, populations)
        ccdf = 1 .- cumsum(dist)
        plot(n[ccdf.>0], ccdf[ccdf.>0],
            label="t = $t",
            xlabel="Size [Cell Count]",
            ylabel="CCDF",
            yscale=:log10,
            legend=:bottomright,
            xlim=(0, 5000),
            ylim=(1e-4, 1),
            dpi=300)
    end
    gif(animation)
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
    println(fit, [δ, critical_size])

    # fit = curve_fit(logmodel, N, ydata, [δ / 2, critical_size/2], lower=[0.001, 0.001], upper=[10.0, 50.0])
    
    # println(coef(fit), [δ, critical_size])
    # println(standard_errors(fit))

    test = plot(Ndata, ydata, label="Data")
    plot!(test, Ndata, logmodel(Ndata, fit.minimizer), label="Fit")
    display(test)

    return fit

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

