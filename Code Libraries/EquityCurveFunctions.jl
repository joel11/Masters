module EquityCurveFunctions

using DatabaseOps
using DataFrames
using FinancialFunctions
using DataGenerator
using DataProcessor
using Plots
export ResultPlots

function DeltaOnePlot(config_ids)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = string("select * from dataset_config where configuration_id in ($configs)")
    dc = RunQuery(query)

    vv = get(dc[1, :variation_values])
    variation_values = map(x -> (parse(Float64, replace(split(x, ',')[1], "(", "")), parse(Float64, replace(split(x, ',')[2], ")", ""))),  split(replace(vv, " ", ""), "),("))


    dataset = GenerateDataset(get(dc[1, :data_seed]), get(dc[1, :steps]), variation_values)
    ogdsplit = parse(Float64, split(get(dc[1, :process_splits]), ',')[(end-1)])
    data_splits = SplitData(dataset,  [ogdsplit, 1.0])
    processed_data = map(x -> ProcessData(x, [1], [1]), data_splits[1:2])

    allsteps = DataFrame()
    ogdsteps = DataFrame()

    for i in names(processed_data[1][1])
        allsteps[(i)] = cumsum(vcat(processed_data[1][1][:,i], processed_data[2][1][:,i]))
        ogdsteps[(i)] = cumsum(processed_data[2][1][:,i])
    end

    colnames = string.(names(allsteps))
    allplot = plot(allsteps[:,parse(colnames[1])],  labels = colnames[1], title = "All delta-1 price changes")
    ogdplot = plot(ogdsteps[:,parse(colnames[1])],  labels = colnames[1], title = "OGD delta-1 price changes")
    for i in 2:size(colnames)[1]
        plot!(allplot, allsteps[:,parse(colnames[i])],  labels = colnames[i])
        plot!(ogdplot, ogdsteps[:,parse(colnames[i])],  labels = colnames[i])
    end

    return (allplot, ogdplot)
end

function PredictedVsActualPlot(config_ids)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = "select * from prediction_results where configuration_id in($configs)"

    results = RunQuery(query)
    groups = by(results, [:stock, :configuration_id], df -> [df])

    comparisons = DataFrame()
    noncum_comparisons = DataFrame()
    for g_index in 1:size(groups, 1)
        row = Array(groups[g_index, :])
        sn = string(get(row[1]))
        data = row[3]
        comparisons[parse(string(sn, "_actual"))] = exp.(cumsum(Array(data[:,:actual])))
        comparisons[parse(string(sn, "_predicted_", get(row[2])))] = exp.(cumsum(Array(data[:,:predicted])))

        noncum_comparisons[parse(string(sn, "_actual"))] = exp.(Array(data[:,:actual]))
        noncum_comparisons[parse(string(sn, "_predicted_", get(row[2])))] = exp.(Array(data[:,:predicted]))
    end

    #Plot to compare input & output
    colnames = string.(names(comparisons))
    price_plot = plot(comparisons[:,parse(colnames[1])],  labels = colnames[1], title = "Predicted vs Actual Timestep Values")
    for i in 2:size(comparisons)[2]
        plot!(price_plot, comparisons[:,parse(colnames[i])],  labels = colnames[i])
    end

    return (price_plot)
end

function PlotEquity(config_ids)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = "select * from prediction_results where configuration_id in($configs)"

    results = RunQuery(query)
    groups = by(results, [:stock, :configuration_id], df -> [df])

    comparisons = DataFrame()
    for g_index in 1:size(groups, 1)
        row = Array(groups[g_index, :])
        sn = string(get(row[1]))
        data = row[3]
        comparisons[parse(string(sn, "_actual"))] = (Array(data[:,:actual]))
        comparisons[parse(string(sn, "_predicted_", get(row[2])))] = (Array(data[:,:predicted]))
    end


    actuals = comparisons[:, filter(x -> (endswith(string(x), "actual")), names(comparisons))]
    ra = cumsum(CalculateReturns(actuals, actuals))
    eq = mapreduce(x -> ra[:,x], +, 1:size(ra,2))
    equity_plot = plot(eq,  labels = "actual", title="Trading Profits")

    for c in config_ids
        println(c)
        predicted = comparisons[:, filter(x -> (endswith(string(x), string(c))), names(comparisons))]
        returns = cumsum(CalculateReturns(actuals, predicted))
        equity_curve = mapreduce(x -> returns[:,x], +, 1:size(returns,2))
        plot!(equity_plot, equity_curve,  labels = string("config_", c))
    end

    return equity_plot
end

function ResultPlots(config_ids, file_name)
    plotlyjs()

    delta_plots = DeltaOnePlot(config_ids)
    prediction_plot = PredictedVsActualPlot(config_ids)
    equity_plot = PlotEquity(config_ids)

    all_plots = [delta_plots[1] delta_plots[2] prediction_plot equity_plot]
    savefig(plot(all_plots..., layout = 4, size=(1400, 700)), string("/users/joeldacosta/desktop/", file_name, ".html"))
end

end
