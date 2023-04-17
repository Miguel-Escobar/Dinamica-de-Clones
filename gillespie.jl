using Random
using Plots
"""
Performs a stochastic simulation of a birth-death process using the Gillespie algorithm.

Args:
- n₀ = initial population size
- birth_rate
- death_rate
- simulation_time

Returns:
- t: an array of times at which an event took place.
- population: an array of population sizes at times t.
"""
function birth_death(n₀, birth_rate, death_rate, simulation_time, max_steps=10_000)
    times = [0.]
    population = [n₀]
    nstep = 0 
    while nstep < max_steps && times[end] < simulation_time
        birth_propensity = population[end]*birth_rate
        death_propensity = population[end]*death_rate   
        α = birth_propensity + death_propensity

        r₁ = rand() # Tiene probabilidad de 1e-308 de colapsar el código.
        r₂ = rand()

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

    return times, population

end


function plot_simulation(times, population)
    plot(times, population, xlabel="Time", ylabel="Population Size", label="Population Size", legend=:topleft)
end