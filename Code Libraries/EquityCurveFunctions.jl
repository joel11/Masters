#module EquityCurveFunctions

workspace()

using DatabaseOps
using DataFrames
using Plots
plotlyjs()
#Apply trading strategy to prices to get price based return
#These are price returns at each time step: Use to construct equity curves

#config_ids = [12,13,14]



function PlotPrices(config_ids, file_name)

    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = "select * from prediction_results where configuration_id in($configs)"

    results = RunQuery(query)
    groups = by(results, [:stock, :configuration_id], df -> [df])

    comparisons = DataFrame()
    for g_index in 1:size(groups, 1)
        row = Array(groups[g_index, :])
        sn = string(get(row[1]))
        data = row[3]
        comparisons[parse(string(sn, "_actual"))] = exp.(cumsum(Array(data[:,:actual])))
        comparisons[parse(string(sn, "_predicted_", get(row[2])))] = exp.(cumsum(Array(data[:,:predicted])))
    end

    #Plot to compare input & output
    colnames = string.(names(comparisons))
    price_plot = plot(comparisons[:,parse(colnames[1])],  labels = colnames[1])
    for i in 2:size(comparisons)[2]
        plot!(price_plot, comparisons[:,parse(colnames[i])],  labels = colnames[i])
    end

    savefig(price_plot, string("/users/joeldacosta/desktop/", file_name, ".html"))

end


PlotPrices([12,13], "12_13")

#end
