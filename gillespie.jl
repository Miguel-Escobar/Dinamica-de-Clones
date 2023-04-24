using Random
using Plots
using ProgressBars
using LaTeXStrings

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
This just takes in a clone size and time and returns the value of the ccdf at time t and clone size ncells.
NT is a vector of the form [ncells, t].
"""
function model(NT, params)
    δ = params[0] 
    birth_rate = params[1] 
    death_rate = params[2]  
    critical_size = params[3]
    n₀ = params[4]  # Maybe I'll just set it to one.

    ncells = NT[1]
    t = NT[2]
    times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, critical_size, δ, NT, 1000)
    ccdf = 1 .- cumsum(system_size_distribution(t, 1, times, populations)[2])
    
    return ccdf[ncells + 1]
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
            dpi=600)
    end
    gif(animation)
end




function main()

    birth_rate = 1/82 # En 1/Horas
    death_rate = birth_rate/4 
    n₀ = 1 
    critical_size = 30
    δ = 2
    simulation_time = 13*24 # En horas (13 días)
    n_simulations = 10_000
    bin_width = 1

    times, populations = modified_birth_death_processes(n₀, birth_rate, death_rate, critical_size, δ, simulation_time, n_simulations)
    sample_times = LinRange(0, simulation_time, 100)
    averages = simulation_averages(sample_times, times, populations)

    
    plot1 = canvas()
    plot!(plot1, x -> n₀*exp((birth_rate-death_rate)*x), 0, simulation_time, label=L"$n_0e^{(r-m)x}$")
    plot!(plot1, sample_times, averages, label="Simulation Average")

    N, distribution = system_size_distribution(simulation_time-1, bin_width, times, populations)
    ccdf = 1 .- cumsum(distribution)

    plot2 = plot(N[ccdf .> 0],
                ccdf[ccdf.> 0],
                #xlim=(0, 500),
                #ylim=(1e-2, 1),
                xlabel="Population Size [Cell number]",
                ylabel="CCDF",
                yscale=:log10,
                st=:steppost,
                label=nothing,
            )
    display(plot1)
    savefig(plot1, "Comparison.png")
    #savefig(plot2, "Distribution.png")

    #time_array = LinRange(0, simulation_time, 180)
    #animate_ccdf(time_array, times, populations)
end
