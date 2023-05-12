using Plots
using LaTeXStrings


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

