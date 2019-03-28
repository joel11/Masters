#module ExperimentGraphs2

workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess
using DataJSETop40
using BSON

using PlotlyJS

################################################################################
##Profit Records

function UpdateTotalProfits(config_ids)

    #Original Setup
    #TotalProfits = DataFrame()
    #TotalProfits[:configuration_id] = config_ids
    #TotalProfits[:profit] = NaN

    TotalProfits = BSON.load("ProfitVals.bson")[:profits]

    current_configs = TotalProfits[:,1]
    needed_configs = collect(setdiff(Set(config_ids), Set(current_configs)))

    if (length(needed_configs) > 0)

        configs = mapreduce(x->string(x, ","), string, needed_configs)[1:(end-1)]
        query = string("select * from prediction_results where configuration_id in ($configs)")
        results = RunQuery(query)

        for i in 1:(length(needed_configs))

            c = needed_configs[i]
            println(c)
            query = string("select * from prediction_results where configuration_id in ($c)")
            results = RunQuery(query)
            profits = sum(CalculateReturnsOneD(Array(results[:,:actual]), Array(results[:,:predicted])))
            #profits = sum(CalculateReturnsOneD(Array(results[:,:actual]), Array(results[:,:actual])))

            TotalProfits = cat(1, TotalProfits, [c profits])
        end

        profit_array = Array(TotalProfits)
        file_name = string("ProfitVals.bson")
        values = Dict(:profits => profit_array)
        bson(file_name, values)
    end
end

function ReadProfits()
    pa = BSON.load("ProfitVals.bson")

    TotalProfits = DataFrame()
    TotalProfits[:configuration_id] = pa[:profits][:,1]
    TotalProfits[:profit] = pa[:profits][:,2]
    return TotalProfits
end

################################################################################
##Boxplots

function getlayerstruc(set_name)
    return ascii(split(split(set_name, "_")[1])[end])
end

function general_boxplot(layer_groups, prefix, filename, variable_name)

    y_vals = layer_groups[1,2][:,variable_name]
    trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
    data = [trace]

    for i in 2:size(layer_groups,1)
        y_vals = layer_groups[i,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
        push!(data, trace)
    end
    plot(data)
    savefig(plot(data), string("/users/joeldacosta/desktop/", filename, ".html"))
end

function sae_boxplot(sae_groups, filename, variable_name)

    encodings = Dict()
    for id in sae_groups[1]
        encodings[id] = OutputSize(GetAutoencoder(ReadSAE(id)[1]))
    end

    y_vals = sae_groups[1,2][:,variable_name]
    trace = box(;y=y_vals, name = string("encoding ", encodings[sae_groups[1,1]]))
    data = [trace]

    for i in 2:size(sae_groups,1)
        y_vals = sae_groups[i,2][:,variable_name]
        trace = box(;y=y_vals, name = string("encoding ", encodings[sae_groups[i,1]]))
        push!(data, trace)
    end

    d = Dict()
    for t in data
        d[parse(Int, SubString(string(t[:name]),10))] = t
    end

    data = map(i -> d[i], sort(map(i -> Int(i), keys(d))))

    savefig(plot(data), string("/users/joeldacosta/desktop/", filename, ".html"))
end

function ProfitBoxplot(query, group_column, prefix, filename, secondary_type, transform_function)
    results = RunQuery(query)

    results = transform_function(results)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{secondary_type,1}(results[:,2])
    layer_returns = join(TotalProfits, results, on = :configuration_id)
    groups = by(layer_returns, [group_column], df -> [df])

    general_boxplot(groups, prefix, filename, :profit)
end

function MSEBoxplot(query, group_column, prefix, filename, secondary_type, transform_function)
    results = RunQuery(query)

    results = transform_function(results)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{secondary_type,1}(results[:,3])
    #layer_returns = join(TotalProfits, results, on = :configuration_id)
    groups = by(results, [group_column], df -> [df])

    general_boxplot(groups, prefix, filename, :cost)
end

function NullTransform(dataset)
    return dataset
end

function LayerTransform(dataset)
    dataset[:layers] = map(getlayerstruc, Array(dataset[:experiment_set_name]))
    return dataset
end

##Profit BoxPlots

function Layer_BxProfit(min_config)
    layers_query = string("select configuration_id, experiment_set_name from configuration_run where configuration_id >= $min_config")
    ProfitBoxplot(layers_query, :layers, "layers", "Layers Profits", String, LayerTransform)
end

function OGD_LR_BxProfit(min_config)
    lr_query = "select configuration_id, learning_rate from training_parameters where configuration_id >= $min_config and category = \"FFN-OGD\""
    ProfitBoxplot(lr_query, :learning_rate, "OGD Learning Rate", "OGD LR Profits", Float64, NullTransform)
end

function FFN_LR_BxProfit(min_config)
    lr_query = "select configuration_id, learning_rate from training_parameters where configuration_id >= $min_config and category = \"FFN\""
    ProfitBoxplot(lr_query, :learning_rate, "FFN Learning Rate", "FFN LR Profits", Float64, NullTransform)
end

function FFN_LR_Sched_BxProfit(min_config)
    min_config = 2064
    lr_query = "select tp.configuration_id, (cast(learning_rate as text) || '-' ||  cast(min_learning_rate as text)) learning_rates
                from training_parameters tp
                inner join configuration_run cr on cr.configuration_id = tp.configuration_id
                where tp.configuration_id >= $min_config and category = 'FFN'
                order by tp.configuration_id desc"

    ProfitBoxplot(lr_query, :learning_rates, "FFN Learning Rates Schedules", "FFN LR-Schedule Profits", String, NullTransform)
end

function SAEProfitBoxPlot(min_config)

    sae_query = string("select configuration_id, sae_config_id from configuration_run where configuration_id >= $min_config")
    sae_results = RunQuery(sae_query)

    sae_results[:,1] = Array{Int64,1}(sae_results[:,1])
    sae_results[:,2] = Array{Int64,1}(sae_results[:,2])
    sae_returns = join(TotalProfits, sae_results, on = :configuration_id)
    groups = by(sae_returns, [:sae_config_id], df -> [df])

    sae_boxplot(groups, "SAE Profit Boxplots", :profit)
end

##MSE BoxPlots

function FFN_LR_MinTest_BxMSE(min_config)

    lr_msequery = "select tp.configuration_id, min(testing_cost) cost, learning_rate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id >= $min_config
                    and tp.category = \"FFN\"
                    and er.category = \"FFN-SGD\"
                group by tp.configuration_id, learning_rate"

    MSEBoxplot(lr_msequery, :learning_rate, "FFN LR", "FFN Learning Rate Min Test MSE", Float64, NullTransform)
end

function OGD_LR_AvgTrain_BxMSE(min_config)

    ogd_mse_query = "select tp.configuration_id, avg(training_cost) cost, learning_rate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id >= $min_config
                    and tp.category = \"FFN-OGD\"
                    and er.category = \"OGD\"
                group by tp.configuration_id, learning_rate
                having training_cost not null"

    MSEBoxplot(ogd_mse_query, :learning_rate, "OGD LR", "OGD Learning Rate Avg Train MSE", Float64, NullTransform)
end

function Layer_MinTest_MxMSE(min_config)
    layer_msequery = string("select er.configuration_id, min(testing_cost) cost, experiment_set_name
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category = \"FFN-SGD\"
                            group by er.configuration_id, sae_config_id")

    MSEBoxplot(layer_msequery, :layers, "Layers", "Layers Min Test MSE", String, LayerTransform)
end

function LastLayer_MinTest_MxMSE(min_config)
    layer_last_query = string("select er.configuration_id, avg(testing_cost) cost, experiment_set_name
                                from epoch_records er
                                inner join configuration_run cr on cr.configuration_id = er.configuration_id
                                where er.configuration_id >= $min_config
                                and er.category = \"FFN-SGD\"
                                and epoch_number > 1900
                                group by er.configuration_id, experiment_set_name
                                order by cost")

    MSEBoxplot(layer_last_query, :layers, "Layers", "Last Layer Min Test MSE", String, LayerTransform)
end

function Init_MinTest_MxMSE(min_config)
    init_query = string("select er.configuration_id, min(testing_cost) cost, initialization init
                        from epoch_records er
                        inner join network_parameters np on np.configuration_id = er.configuration_id
                        where np.category = \"SAE\"
                        group by er.configuration_id, init
                        having cost not null")

    MSEBoxplot(init_query, :init, "Init", "Inits Min Test MSE", String, NullTransform)
end

function SAEMSEBoxPlot(min_config) #Best Epoch - Min Test
    sae_msequery = string("select er.configuration_id, min(testing_cost) min_test_cost, sae_config_id
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category = \"FFN-SGD\"
                            group by er.configuration_id, sae_config_id")

    sae_mseresults = RunQuery(sae_msequery)
    sae_mseresults[:,1] = Array{Int64,1}(sae_mseresults[:,1])
    sae_mseresults[:,2] = Array{Float64,1}(sae_mseresults[:,2])
    sae_mseresults[:,3] = Array{Int64,1}(sae_mseresults[:,3])
    mse_groups = by(sae_mseresults, [:sae_config_id], df -> [df])
    sae_boxplot(mse_groups, "SAE Best Epoch - Min MSE Boxplot", :min_test_cost)
end

function SAELastEpochsMSEBoxPlot(min_config) #Average Test over last 100

    sae_last_query = string("select er.configuration_id, epoch_number, avg(testing_cost) min_test_cost, sae_config_id
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category = \"FFN-SGD\"
                            and epoch_number > 1900
                            group by er.configuration_id
                            order by min_test_cost")
    sae_lastresults = RunQuery(sae_last_query)
    sae_lastresults[:,1] = Array{Int64,1}(sae_lastresults[:,1])
    sae_lastresults[:,2] = Array{Int64,1}(sae_lastresults[:,2])
    sae_lastresults[:,3] = Array{Float64,1}(sae_lastresults[:,3])
    sae_lastresults[:,4] = Array{Int64,1}(sae_lastresults[:,4])
    last_groups = by(sae_lastresults, [:sae_config_id], df -> [df])
    sae_boxplot(last_groups, "SAE Last 100 Average MSE Boxplot", :min_test_cost)
end

################################################################################
##Heatmaps

function FFN_LR_x_Layers_ProfitHeatmap(min_config)

    layer_msequery = string("select er.configuration_id, learning_rate, min(testing_cost) min_test_cost, experiment_set_name
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            inner join training_parameters tp on tp.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category = \"FFN-SGD\"
                            and tp. category = \"FFN\"
                            group by er.configuration_id, sae_config_id, learning_rate")
    layer_mseresults = RunQuery(layer_msequery)

    layer_mseresults[:,1] = Array{Int64,1}(layer_mseresults[:,1])
    layer_mseresults[:,2] = Array{Float64,1}(layer_mseresults[:,2])
    layer_mseresults[:,3] = Array{Float64,1}(layer_mseresults[:,3])
    layer_mseresults[:,4] = Array{String,1}(layer_mseresults[:,4])
    layer_mseresults = LayerTransform(layer_mseresults)
    layer_mseresults = join(TotalProfits, layer_mseresults, on = :configuration_id)

    comb_mse = by(layer_mseresults, [:layers, :learning_rate], df -> maximum(df[:profit]))
    #comb_mse = by(layer_mseresults, [:layers, :learning_rate], df -> mean(df[:min_test_cost]))

    layers_order = Array(unique(comb_mse[:layers]))

    layers_order = Array(sort(map(l -> string(split(l, "x")[2], "x", split(l, "x")[1]), layers_order)))
    rates_order = Array(unique(comb_mse[:learning_rate]))

    vals = Array{Float64,2}(fill(NaN, length(layers_order), length(rates_order)))

    for r in 1:size(comb_mse, 1)
        lval = string(split(comb_mse[r,1], "x")[2], "x", split(comb_mse[r,1], "x")[1])
        l_index = findfirst(layers_order, lval)
        r_index = findfirst(rates_order, comb_mse[r,2])
        vals[l_index, r_index] = comb_mse[r,3]
    end

    yvals = map(i -> string("lr ", i), rates_order)
    trace = heatmap(z = vals, y = yvals, x = layers_order)
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/layers_lr_heatmap_max_profit.html"))
end

################################################################################
##Price Line Plots

function StockPricePlot()
    var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
    ds = GenerateDataset(167, 5000, var_pairs)

    price_plot = plot(Array(ds), name = ["stock 1" "stock 2" "stock 3" "stock 4" "stock 5" "stock 6"])
    savefig(price_plot, "/users/joeldacosta/desktop/PriceGraphs.html")
end

function RecreateStockPrices(config_names)
    configs = mapreduce(x->string(x, ","), string, collect(keys(config_names)))[1:(end-1)]
    best_query = string("select * from prediction_results where configuration_id in ($configs)")
    best_results = RunQuery(best_query)
    best_groups = by(best_results, [:stock], df -> [df])

    for i in 1:size(best_groups,1)
        timesteps = best_groups[i,2][:time_step]
        config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

        actual = cumsum(config_groups[1,2][:actual])
        predicted_one = cumsum(config_groups[1,2][:predicted])
        predicted_two = cumsum(config_groups[2,2][:predicted])

        stock_name = get(best_groups[i,1])

        t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
        t1 = scatter(;y=predicted_one, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][1])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))
        t2 = scatter(;y=predicted_two, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][2])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))

        recreation_plots = [t0, t1, t2]
        filename = string("recreation_", stock_name)
        savefig(plot(recreation_plots), string("/users/joeldacosta/desktop/", filename, ".html"))

    end
end

#names = Dict(902 => "Profit", 510 => "MSE")
#RecreateStockPrices(names)

################################################################################
##Profit PDF

function AllProfitsPDF(min_config)
    indexes = Array{Bool}(TotalProfits[:configuration_id] .> min_config)

    x0 = Array(TotalProfits[indexes,:profit])
    trace = histogram(;x=x0, name="Configuration Profits")
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/Profits PDF.html"))
end

###############################################################################
##General Plots

#config_ids = 504:999
config_ids = 2064:2183
UpdateTotalProfits(config_ids)
TotalProfits = ReadProfits()
#min_config = 504
#sae_choices = (253, 265, 256, 260, 264)
#min_config = minimum(TotalProfits[:,1])

#end
