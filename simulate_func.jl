using Random
using ProgressBars


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

    # We want each thread to fill it's own list, to avoid data racing:
    temp_times = [Vector{Float64}[] for i in 1:Threads.nthreads()]  
    temp_populations = [Vector{Float64}[] for i in 1:Threads.nthreads()]

    Threads.@threads for i in ProgressBar(1:n_simulations)
        id = Threads.threadid()
        t, p = modified_birth_death(n₀, birth_rate, death_rate, critical_size, δ, simulation_time)
        push!(temp_times[id], t)
        push!(temp_populations[id], p)
    end

    # Now we need to concatenate the lists:
    for i in 1:Threads.nthreads()
        append!(times, temp_times[i])
        append!(populations, temp_populations[i])
    end

    return times, populations
end
