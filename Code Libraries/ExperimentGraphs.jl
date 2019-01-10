module ExperimentGraphs

using DatabaseOps
using DataFrames
using FinancialFunctions
using DataGenerator
using DataProcessor
using Plots
export PlotResults, PlotEpochs, PlotSAERecontructions

plotlyjs()

function DeltaOnePlot(config_ids)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = string("select * from dataset_config where configuration_id in ($configs)")
    dc = RunQuery(query)

    vv = get(dc[1, :variation_values])
    variation_values = map(x -> (parse(Float64, replace(split(x, ',')[1], "(", "")), parse(Float64, replace(split(x, ',')[2], ")", ""))),  split(replace(vv, " ", ""), "),("))

    dataset = GenerateDataset(get(dc[1, :data_seed]), get(dc[1, :steps]), variation_values)
    ogdsplit = parse(Float64, split(get(dc[1, :process_splits]), ',')[max(1, end-1)])
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
    query = string("select pr.*, cr.experiment_set_name from prediction_results pr inner join configuration_run cr on cr.configuration_id = pr.configuration_id where cr.configuration_id in ($configs)")
    results = RunQuery(query)
    groups = by(results, [:stock, :experiment_set_name], df -> [df])

    comparisons = DataFrame()
    noncum_comparisons = DataFrame()

    for g_index in 1:size(groups, 1)
        row = Array(groups[g_index, :])
        sn = string(get(row[1]))
        data = row[3]
        comparisons[parse(string(sn, "_actual"))] = (cumsum(Array(data[:,:actual])))
        comparisons[parse(string(sn, "_predicted_", replace(string(get(row[2])), ".", "_")))] = (cumsum(Array(data[:,:predicted])))

        #noncum_comparisons[parse(string(sn, "_actual"))] = exp.(Array(data[:,:actual]))
        #noncum_comparisons[parse(string(sn, "_predicted_", replace(string(get(row[2])), ".", "_")))] = exp.(Array(data[:,:predicted]))
    end

    #Plot to compare input & output
    colnames = string.(names(comparisons))
    price_plot = plot(comparisons[:,parse(colnames[1])],  labels = colnames[1], title = "Predicted vs Actual Timestep Values")
    for i in 2:size(comparisons)[2]
        style = contains(colnames[i], "actual") ? :solid : :dash
        plot!(price_plot, linestyle = style, comparisons[:,parse(colnames[i])],  labels = colnames[i])
    end

    return (price_plot)
end

function PlotEquity(config_ids)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = string("select pr.*, cr.experiment_set_name from prediction_results pr inner join configuration_run cr on cr.configuration_id = pr.configuration_id where cr.configuration_id in ($configs)")

    results = RunQuery(query)
    groups = by(results, [:stock, :experiment_set_name, :configuration_id], df -> [df])

    comparisons = DataFrame()
    for g_index in 1:size(groups, 1)
        row = Array(groups[g_index, :])
        sn = string(get(row[1]))
        configset = replace(string(get(row[2])), ".", "_")
        c_id = get(row[3])
        data = row[4]

        comparisons[parse(string(sn, "_actual"))] = (Array(data[:,:actual]))
        comparisons[parse(string(sn, "_predicted_", configset, "_", c_id))] = (Array(data[:,:predicted]))
    end


    actuals = comparisons[:, filter(x -> (endswith(string(x), "actual")), names(comparisons))]
    ra = cumsum(CalculateReturns(actuals, actuals))
    eq = mapreduce(x -> ra[:,x], +, 1:size(ra,2))
    equity_plot = plot(eq, labels = "actual", title="Trading Profits")

    for c in config_ids
        set_name = filter(x -> (endswith(string(x), string(c))), names(comparisons))[1]
        predicted = comparisons[:, filter(x -> (endswith(string(x), string(c))), names(comparisons))]
        returns = cumsum(CalculateReturns(actuals, predicted))
        equity_curve = mapreduce(x -> returns[:,x], +, 1:size(returns,2))
        plot!(equity_plot, linestyle=:dash, equity_curve,  labels = string("config_", set_name))
    end

    return equity_plot
end

function PlotEpochs(config_ids, file_name)
    configs = mapreduce(x->string(x, ","), string, config_ids)[1:(end-1)]
    query = string("select * from epoch_records where configuration_id in ($configs)")
    results = RunQuery(query)

    function ProcessValueArray(array_vals)
        vals = deepcopy(array_vals)
        for i in 1:length(vals)
            if isnull(vals[i])
                vals[i] = 1
            end
        end
        return vals
    end

    function PlotErrorsAndTimes(cat)
        epoch_records = results[Array(results[:,:category]) .== cat, :]
        config_groups = by(epoch_records, [:configuration_id], df -> [df])

        costsplot = plot(log.(Array(ProcessValueArray(config_groups[1, 2][:, :training_cost]))), xlabel = "Epoch", ylabel = "Log Cost",  labels = string(get(config_groups[1, 1]), "_", cat, "_training"), title = string(cat, " Costs"))
        plot!(costsplot, log.(Array(ProcessValueArray(config_groups[1, 2][:, :testing_cost]))), xlabel = "Epoch", ylabel = "Log Cost",linestyle = :dash, labels = string(get(config_groups[1, 1]), "_", cat, "_testing"))

        #costsplot = plot((Array(ProcessValueArray(config_groups[1, 2][:, :training_cost]))), xlabel = "Epoch", ylabel = " Cost",  labels = string(get(config_groups[1, 1]), "_", cat, "_training"), title = string(cat, " Costs"))
        #plot!(costsplot, (Array(ProcessValueArray(config_groups[1, 2][:, :testing_cost]))), xlabel = "Epoch", ylabel = " Cost",linestyle = :dash, labels = string(get(config_groups[1, 1]), "_", cat, "_testing"))

        for i in 2:size(config_groups, 1)
            println(i)
            plot!(log.(Array(ProcessValueArray(config_groups[i, 2][:, :training_cost]))), xlabel = "Epoch", ylabel = "Log Cost", labels = string(get(config_groups[i, 1]), "_", cat, "_training"))
            plot!(costsplot, log.(Array(ProcessValueArray(config_groups[i, 2][:, :testing_cost]))), xlabel = "Epoch", ylabel = " Cost", linestyle = :dash, labels = string(get(config_groups[i, 1]), "_", cat, "_testing"))
            #plot!((Array(ProcessValueArray(config_groups[i, 2][:, :training_cost]))), xlabel = "Epoch", ylabel = " Cost", labels = string(get(config_groups[i, 1]), "_", cat, "_training"))
            #plot!(costsplot, (Array(ProcessValueArray(config_groups[i, 2][:, :testing_cost]))), xlabel = "Epoch", ylabel = " Cost", linestyle = :dash, labels = string(get(config_groups[i, 1]), "_", cat, "_testing"))
        end

        timesplot = plot(cumsum(Array(config_groups[1, 2][:, :run_time])), labels = string(get(config_groups[1, 1]), "_", cat, "_training"), title = string(cat, " Runtimes"))

        for i in 2:size(config_groups, 1)
            plot!(timesplot, cumsum(Array(config_groups[i, 2][:, :run_time])), labels = string(get(config_groups[i, 1]), "_", cat, "_training"))
        end

        return [costsplot, timesplot]
    end

    categories = unique(Array(results[:, :category]))
    plots = mapreduce(PlotErrorsAndTimes, vcat, filter(x -> x != "OGD", categories))

    savefig(plot(plots..., layout = length(plots), size=(1400, 700)), string("/users/joeldacosta/desktop/", file_name, ".html"))
end

function PlotResults(config_ids, file_name)
    delta_plots = DeltaOnePlot(config_ids)
    prediction_plot = PredictedVsActualPlot(config_ids)
    equity_plot = PlotEquity(config_ids)

    all_plots = [delta_plots[1] delta_plots[2] prediction_plot equity_plot]
    savefig(plot(all_plots..., layout = 4, size=(1400, 700)), string("/users/joeldacosta/desktop/", file_name, ".html"))
end

function PlotSAERecontructions(training_pairs, file_name)
    function ReconPlot(pair)
        training_inputplot = plot(cumsum(pair[3][1]), linestyle = :solid, labels = "actual", title=string("Testing Data Reconstructions ", pair[1]))
        plot!(training_inputplot, cumsum(pair[3][2][end]), labels = pair[2], linestyle = :dash)
        return training_inputplot
    end

    reconplots = map(ReconPlot, training_pairs)
    savefig(plot(reconplots..., layout = length(reconplots), size=(1400, 700)), string("/users/joeldacosta/desktop/", file_name, ".html"))
end

end
