using XLSX
using StatsBase
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
        for j in 1:length(populations) # Deprecated. Should be replaced.
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
This should be improved! It's easy to multi-thread I think.
"""
function system_size_distribution(t, bin_width::Int,times, populations)
    n = length(populations)
    max_size = maximum(x->maximum(x), populations) + 1
    tempdistribution = [zeros(max_size) for thread in 1:Threads.nthreads()] # This is to avoid data-racing
    Threads.@threads for i in 1:n
        threadnumber = Threads.threadid()
        tempdistribution[threadnumber][element_at_time(t, times[i], populations[i])+1] += 1
    end
    
    distribution = zeros(max_size)
    for threadnumber in 1:Threads.nthreads()
        distribution = distribution .+ tempdistribution[threadnumber]
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


function read_excel_data(datalocation)
    data = XLSX.readxlsx(datalocation)["raw"]
    timecodes = convert(Vector{Int64}, data["L"][2:end])
    clusterdata = convert(Vector{Int64},data["N"][2:end]) # Cambiar a N
    clusters_at_different_timecodes = [clusterdata[timecodes .== t] for t in unique(timecodes)]
    clusters_at_different_timecodes = [counts(clusters) for clusters in clusters_at_different_timecodes]
    max_size = maximum(maximum.(clusters_at_different_timecodes))

    timecodes = unique(timecodes)
    distributions = []
    for cluster_sizes in clusters_at_different_timecodes
        temp_distribution = zeros(max_size)
        for clustersize in cluster_sizes
            if clustersize>0
                temp_distribution[clustersize] += 1
            end
        end
        push!(distributions, temp_distribution./sum(temp_distribution))
    end
    # println(length(clusters_at_different_timecodes[8]))
    return timecodes, distributions
end


function read_ccdf_in_data(datalocation)
    timecodes, distribution = read_excel_data(datalocation)
    return timecodes, [1 .- cumsum(dist) for dist in distribution]
end
