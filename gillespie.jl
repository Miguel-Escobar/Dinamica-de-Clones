using Random
using Plots
using ProgressBars
"""
Performs a stochastic simulation of a birth-death process using the Gillespie algorithm.

# Args:
- n₀ = initial population size
- birth_rate
- death_rate
- simulation_time

# Returns:
- t: an array of times at which an event took place.
- population: an array of population sizes at times t.
"""
function birth_death(n₀, birth_rate, death_rate, simulation_time; max_steps=1_000_000)
    times = Float64[0.]
    population = Int[n₀]
    nstep = 0 
    while nstep < max_steps && times[end] < simulation_time
        birth_propensity = population[end]*birth_rate
        death_propensity = population[end]*death_rate   
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
        println("Warning: max number of steps reached. Simulation did not reach final time.")
    end

    return times, population

end


"""
Performs n_simulations of a birth-death process using the Gillespie algorithm.

# Arguments
- n₀ = initial population size
- birth_rate
- death_rate
- simulation_time
- n_simulations

# Returns
- times: an array of arrays of times at which an event took place.
- populations: an array of arrays of population sizes at times t.
"""
function birth_death_processes(n₀, birth_rate, death_rate, simulation_time, n_simulations)
    times = Vector{Float64}[]
    populations = Vector{Int}[]
    for i in ProgressBar(1:n_simulations)
        t, p = birth_death(n₀, birth_rate, death_rate, simulation_time)
        push!(times, t)
        push!(populations, p)
    end
    println(typeof(times), typeof(populations))
    return times, populations
end

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
function modified_birth_death(n₀, birth_rate, death_rate, critical_size, δ, simulation_time; max_steps=10_000_000)

    times = Float64[0.]
    population = Int[n₀]
    nstep = 0 

    while nstep < max_steps && times[end] < simulation_time
        if population[end] < critical_size
            birth_propensity = population[end]*birth_rate
            death_propensity = population[end]*death_rate   
        else
            birth_propensity = population[end]*(birth_rate + δ)
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
        println("Warning: max number of steps reached. Simulation did not reach final time.")
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
    for i in ProgressBar(1:n_simulations)
        t, p = modified_birth_death(n₀, birth_rate, death_rate, critical_size, δ, simulation_time)
        push!(times, t)
        push!(populations, p)
    end
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
Just creates a canvas to plot nicely (enough).
"""
function canvas()
    return plot(xlabel="Time", ylabel="Population Size", yscale=:log10, legend=:bottomright)
end


function main()

    birth_rate = 1/82 # En 1/Horas
    death_rate = birth_rate/4 
    n₀ = 1 
    critical_size = 30
    δ = .6
    simulation_time = 6*24 # En horas (13 días)
    n_simulations = 10_000
    bin_width = 1
    

    # times, populations = birth_death_processes(n₀, birth_rate, death_rate, simulation_time, n_simulations)
    times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, critical_size, δ, simulation_time, n_simulations)
    sample_times = LinRange(0, simulation_time, 100)
    averages = simulation_averages(sample_times, times, populations)

    plot1 = canvas()
    plot!(plot1, x -> n₀*exp((birth_rate-death_rate)*x), 0, simulation_time, label="n₀*exp((birth_rate-death_rate)*x))")
    plot!(plot1, sample_times, averages, label="Simulation Averages")

    N, distribution = system_size_distribution(simulation_time-1, bin_width, times, populations)
    ccdf = 1 .- cumsum(distribution)
    plot2 = plot(N[ccdf .> 0],
                ccdf[ccdf.> 0],
                xlim=(0, 1e3),
                xlabel="Population Size [Cell number]",
                ylabel="Fraction of cells",
                legend=:topright,
                yscale=:log10,
                st=:steppost
            )
    savefig(plot1, "Comparison.png")
    savefig(plot2, "Distribution.png")
end
