module ExperimentGraphs

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using DataJSETop40
using BSON
using MLBase

using ColorBrewer
using PlotlyJS
using Combinatorics

export ClusterDistributionPercCluster, PrintClusterAnalysis, OGD_MSE_vs_PL, ClusterOGDMSEPlot, ClusterDistributionPercAll, PlotPBOBySplits, PlotCombinationSizes, PlotConfusion, SharpeRatiosPDF, MSE_Reg_EncodingLines, PL_OOS_OGDLR_Delta_Encoding, PL_IS_Encoding, PL_OOS_OGDLR_Deltas, PL_Scaling, PL_Heatmap_NetworkSize_DataAggregation, PL_SAE_Encoding_SizeLines, PL_Activations, PL_ScalingOutputActivation, MSE_OutputActivation_Scaling_Filters, MSE_Min_EncodingSize_Activation, MSE_Encoding_Activation, MSE_Output_Activation, MSE_Hidden_Activation, MSE_Scaling_Filters, PL_Heatmap_LearningRate_MaxEpochs, MSE_EpochCycle, MSE_LearningRate_MaxMin, MSE_LayerSizesLines, PL_NetworkSizeLines, MSE_LayerSizes3, PL_LearningRates_MaxMin, GenerateDeltaDistribution, BestStrategyVsBenchmark, AllProfitsPDF, ReadProfits, OGD_Heatmap_LayerSizes_Epochs, PL_EpochCycle, AllProfitsPDF, TransformConfigIDs, ConfigStrategyOutput, PL_MaxEpochs, PL_Denoising, ReadProfits, UpdateTotalProfits, MSE_Pretraining, OGD_ScalingOutputActivation_Profits_Bx, MSE_Activation_Scaling_Filters, FFN_LR_Sched_BxProfit, PL_SAE_Encoding_Size, PL_L1Reg, PL_NetworkSize, PL_Init, MSE_LayerSizes, SAE_EncodingSizes_MinMSE, MSE_Deltas, MSE_Init, SAE_LREpochs_MinTest_BxMSE, RecreateStockPricesSingle, BestStrategyGraphs, PL_DataDeltas, SAEProfitBoxPlot, OGD_DataVariances_Profits_Bx, OGD_NetworkVariances_Profits_Bx,MSE_Lambda1, MSE_Denoising, PL_ValidationSplit, MSE_MaxLearningRate_Activation, FFN_LR_BxProfit, OGD_LR_AvgTrain_BxMSE, PL_OGD_LearningRate, PL_LayerActivation_OutputActivation, OGD_SAE_Selection_Profits_bx, PL_Activations_NetworkSize, SAE_ActivationsNetworkSizes_MinMSE, MSE_ActivationsEncodingSizes

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end

##Generic Plot Functions##############################################################################

const default_fontsize = 18

const colourSets = Dict("SyntheticMSE" => "Pastel2",
                  "SyntheticPL" => "Pastel1",
                  "ActualMSE" => "Set2",
                  "ActualPL" => "Set1")

function OrderDataFrame(groups, ordering)

    new_frame = deepcopy(groups)

    found = 1

    for r in 1:size(ordering,1)
        key = ordering[r]
        positions = findin(Array(groups[:,1]), [key])

        if size(positions,1) > 0
            index = positions[1]
            for c in 1:size(groups, 2)
                new_frame[found,c] = groups[index, c]
            end
            found = found + 1
        end
    end

    return new_frame
end

function general_boxplot(layer_groups, xaxis_label, filename, variable_name, yaxis_label, xlabels_show, colourSetChoice = "", font_size = 18)
    #Boxpoints = false, "all", "outliers", "suspectedoutliers"

    colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    group_sizes = Array{Int,1}()

    y_vals = layer_groups[1,2][:,variable_name]
    push!(group_sizes, size(y_vals,1))
    trace = box(;y=y_vals
                ,name = string(layer_groups[1,1])
                ,boxpoints="outliers"
                ,marker_color = colors[1]
                ,boxmean = "mean"
                #,jitter=0.25
                )
    data = [trace]

    for i in 2:size(layer_groups,1)
        colourIndex = i #>= 6 ? i + 1 : i
        colourIndex = i > 8 ? i % 8 + 1: i
        println(i)
        println(colourIndex)


        y_vals = layer_groups[i,2][:,variable_name]
        push!(group_sizes, size(y_vals,1))
        trace = box(;y=y_vals,
                    name = string(layer_groups[i,1])
                    ,boxpoints="outliers"
                    ,marker_color = colors[colourIndex]
                    ,boxmean = "mean"
                    #,jitter=0.25
                    )
        push!(data, trace)
    end

    layout_x_axis_label = (xaxis_label == nothing || xaxis_label == "") ? "" : string("<b>", xaxis_label, "</b><br><br><br>")

    l = Layout(width = 900, height = 600, margin = Dict(:b => 140, :l => 100)
        , yaxis = Dict(:title => string("<b>", yaxis_label, "<br> </b>"))
        , xaxis = Dict(:title => string(layout_x_axis_label, "<i> Sample Sizes: ", string(group_sizes), "</i>")
                     , :showticklabels => xlabels_show)
        , font = Dict(:size => 18)
         )


    fig = Plot(data, l)
    savefig(plot(fig), string("./GraphOutputDir/", filename, ".html"))

    return fig
end

function sae_boxplot(sae_groups, filename, variable_name)

    encodings = Dict()
    for id in sae_groups[1]
        encodings[id] = OutputSize(GetAutoencoder(ReadSAE(id)[1]))
        if id == 1
            encodings[id] = 0
        end
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

    savefig(plot(data), string("./GraphOutputDir/", filename, export_ext))
end

function ProfitBoxplot(query, group_column, xaxis_label, filename, secondary_type, testDatabase = false, colourSetChoice = "", ordering = nothing, xlabels_show = true, font_size = 18, minimum_val = nothing, in_sample = false)

    results = RunQuery(query, testDatabase)

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{secondary_type,1}(results[:,2])
    layer_returns = join(GetProfits(testDatabase, in_sample), results, on = :configuration_id)

    println(minimum_val)
    if minimum_val != nothing
        #layer_returns = layer_returns[layer_returns[:profit] .> minimum_val, :]
        layer_returns = layer_returns[Array{Bool,1}(layer_returns[:profit] .> minimum_val), :]
    end

    groups = by(layer_returns, [group_column], df -> [df])

    groups = ordering == nothing ? groups : OrderDataFrame(groups, ordering)

    set = in_sample ? "IS" : "OOS"

    general_boxplot(groups, xaxis_label, filename, :profit, string(set, " P&L"), xlabels_show, colourSetChoice, font_size)
end

function MSEBoxplot(query, group_column, xaxis_label, filename, secondary_type, colourSetChoice = "", testDatabase = false, xlabels_show = true, font_size = 18, ordering = nothing)

    results = RunQuery(query, testDatabase)

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{secondary_type,1}(results[:,3])
    groups = by(results, [group_column], df -> [df])

    groups = ordering == nothing ? groups : OrderDataFrame(groups, ordering)

    general_boxplot(groups, xaxis_label, filename, :cost, "MSE", xlabels_show, colourSetChoice, font_size)
end

##P&L Plot Functions##################################################################################

function PL_ScalingOutputActivation(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                    replace((output_activation || '-' || scaling_function), 'Activation', '') scaling_methodology
            from configuration_run cr
            inner join dataset_config dc on cr.configuration_id = dc.configuration_id
            inner join network_parameters np on np.configuration_id = cr.configuration_id
            where cr.configuration_id in ($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :scaling_methodology, " ", string(file_prefix, "Scaling Methodolgy Profits"), String, testDatabase, colourSetChoice, nothing, false)
end

function PL_Scaling(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                    scaling_function scaling_methodology
            from configuration_run cr
            inner join dataset_config dc on cr.configuration_id = dc.configuration_id
            inner join network_parameters np on np.configuration_id = cr.configuration_id
            where cr.configuration_id in ($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :scaling_methodology, " ", string(file_prefix, "Scaling Methodolgy Profits"), String, testDatabase, colourSetChoice, nothing, true)
end

function PL_ScalingHiddenOutputActivation(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) || '-' || output_activation || '-' || scaling_function) scaling_methodology
            from configuration_run cr
            inner join dataset_config dc on cr.configuration_id = dc.configuration_id
            inner join network_parameters np on np.configuration_id = cr.configuration_id
            where cr.configuration_id in ($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :scaling_methodology, " ", string(file_prefix, "Scaling Methodolgy Profits"), String, testDatabase, colourSetChoice, false)
end

function PL_LearningRates_MaxMin(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    ids = TransformConfigIDs(config_ids)

    lr_query = "select tp.configuration_id,
                replace(('' || cast(tp.learning_rate as text) || '-' ||  cast(min_learning_rate as text)), '1.0e-05', '0.00001') learning_rates
                from training_parameters tp
                inner join configuration_run cr on cr.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids) and category = 'FFN'
                    and learning_rate != min_learning_rate
                order by learning_rates"

    ProfitBoxplot(lr_query, :learning_rates, "SGD Learning Rates", string(file_prefix, "PL SGD LearningRates MaxMin"), String, testDatabase, colourSetChoice, nothing, true, 18 , nothing, in_sample)
end

function PL_SAE_Size(config_ids)

    ids = TransformConfigIDs(config_ids)

    sae_query = string("select configuration_id, sae_config_id from configuration_run
                            where configuration_id in ($ids)")
    sae_results = RunQuery(sae_query)

    sae_results[:,1] = Array{Int64,1}(sae_results[:,1])
    sae_results[:,2] = Array{Int64,1}(sae_results[:,2])
    sae_returns = join(GetProfits(), sae_results, on = :configuration_id)
    groups = by(sae_returns, [:sae_config_id], df -> [df])

    sae_boxplot(groups, "SAE Profit Boxplots", :profit)
end

function PL_L1Reg(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    (cast(l1_lambda as string) || ' lambda') lambda
                from training_parameters tp
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""

    ProfitBoxplot(lr_query, :lambda, "L1 Lambda ", string(file_prefix, "L1 Reg Effects on Profits"), String, testDatabase, colourSetChoice, nothing, true, default_fontsize, nothing, in_sample)
end

function PL_L1Reg_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    aggregation = median

    ids = TransformConfigIDs(config_ids)
    query = "select tp.configuration_id,
                    l1_lambda,
                    deltas
                from training_parameters tp
                inner join dataset_config dc on dc.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""


    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{String,1}(results[:,3])
    results = join(GetProfits(testDatabase, in_sample), results, on = :configuration_id)
    #results[:encoding] = map(i -> parse(Int64, ascii(i[end])), split.(results[:layer_sizes], ','))

    groups = by(results, [:l1_lambda, :deltas], df -> aggregation(df[:profit]))


    encoding_groups = by(groups, [:deltas], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in 1:size(encoding_groups, 1)
        trace = scatter(;x=encoding_groups[i,2][:l1_lambda],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " "),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
        , font = Dict(:size => default_fontsize)
         )

    savefig(plot(data, l), string("./GraphOutputDir/FFN PL by Lamda and Deltas ", string(aggregation), ".html"))
end

function PL_ValidationSplit(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                case when experiment_set_name like '%CVSplit_1.0%' then '0%'
                     when experiment_set_name like '%CVSplit_0.9%' then '10%'
                     when experiment_set_name like '%CVSplit_0.8%' then '20%'
                     when experiment_set_name like '%CVSplit_0.7%' then '30%'
                     when experiment_set_name like '%CVSplit_0.6%' then '40%'
                     when experiment_set_name like '%CVSplit_0.5%' then '50%'
                     when experiment_set_name like '%CVSplit_0.4%' then '60%'
                     when experiment_set_name like '%CVSplit_0.3%' then '70%'
                     when experiment_set_name like '%CVSplit_0.2%' then '80%'
                     when experiment_set_name like '%CVSplit_0.1%' then '90%'
                end Validation_Percentage_Size
            from configuration_run cr
            where configuration_id in ($ids)
            order by cr.configuration_id desc"


    r = RunQuery(query, testDatabase)

    ProfitBoxplot(query, :Validation_Percentage_Size, "% Training Data Excluded ", string(file_prefix, "Validation Set Effects on Profits"), String, testDatabase, colourSetChoice,  [], false, 18, 5)
end

function PL_OGD_LearningRate(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    'Learning Rate:' || cast(learning_rate as string) learning_rate
                from training_parameters tp
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN-OGD\"
                order by learning_rate
                "

     ProfitBoxplot(lr_query, :learning_rate, nothing, string(file_prefix, "OGD LR Profits"), String, testDatabase, colourSetChoice, nothing, false, 18)
end

function PL_Denoising(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    tp.denoising_variance
                from training_parameters tp
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""

    ProfitBoxplot(lr_query, :denoising_variance, "SGD Denoising Variance", string(file_prefix, "OGD Denoising Variance Profits"), Float64, testDatabase, colourSetChoice, nothing, true, default_fontsize, nothing, in_sample)
end

function PL_MaxEpochs(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, ordering = ["10 Epochs", "100 Epochs", "500 Epochs", "1000 Epochs"])

    ids = TransformConfigIDs(config_ids)
    query = "select tp.configuration_id,
                    (cast(tp.max_epochs as string) || ' Epochs') max_epochs
                from training_parameters tp
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""

    ProfitBoxplot(query, :max_epochs, "IS Training Epochs", string(file_prefix, "PL Max Epochs"), String, testDatabase, colourSetChoice, ordering, true)
end

function PL_EpochCycle(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    (case when epoch_cycle_max = -1 then 'Constant Learning Rate' else cast(tp.epoch_cycle_max as string) || ' Epochs' end) max_epochs
                from training_parameters tp
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""

    ProfitBoxplot(lr_query, :max_epochs, "SGD Learning Rate Epoch Cycle", string(file_prefix, "PL SGD Epoch Cycle"), String, testDatabase, colourSetChoice, nothing, true, default_fontsize, nothing, in_sample)
end

function PL_LayerActivation_OutputActivation(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
            (
            substr(layer_activations, 1,  instr(layer_activations, ',')-11)
            || '-' || output_activation
            ) networkconfig
            from configuration_run cr
            inner join  network_parameters np on cr.configuration_id = np.configuration_id
            where cr.configuration_id in ($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :networkconfig, "Hidden-Output", string(file_prefix, "Network Activations Profits"), String, testDatabase, colourSetChoice)
end

function PL_Activations(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
            (
            substr(layer_activations, 1,  instr(layer_activations, ',')-11) || ''
            ) networkconfig
            from configuration_run cr
            inner join  network_parameters np on cr.configuration_id = np.configuration_id
            where cr.configuration_id in($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :networkconfig, "", string(file_prefix, "Network Size Profits"), String, testDatabase, colourSetChoice)
end

function PL_Activations_NetworkSize(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
            (
            substr(layer_activations, 1,  instr(layer_activations, ',')-11)
            --|| '-' || output_activation
            || '-' ||
            substr(layer_sizes, instr(layer_sizes, ',') + 1, length(layer_sizes))
            ) networkconfig
            from configuration_run cr
            inner join  network_parameters np on cr.configuration_id = np.configuration_id
            where cr.configuration_id in($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :networkconfig, "", string(file_prefix, "Network Size Profits"), String, testDatabase, colourSetChoice)
end

function PL_NetworkSize(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, ordering = nothing)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
            ('Layers: ' || substr(layer_sizes, instr(layer_sizes, ',') + 1, length(layer_sizes)-3-instr(layer_sizes, ','))) layer_sizes
            --substr(np.layer_sizes, 4) layer_sizes
        from network_parameters np
        where np.configuration_id in ($ids)"

        d = RunQuery(query, testDatabase)

        if ordering != nothing
            ordering = map(i -> string("Layers: ", i), ordering)
        end

    ProfitBoxplot(query, :layer_sizes, "Network Layers", string(file_prefix, "Network Size Profits"), String, testDatabase, colourSetChoice, ordering, false, 16, 5)
end

function PL_NetworkSizeLines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
            layer_sizes
        from network_parameters np
        inner join epoch_records er on er.configuration_id = np.configuration_id and er.category = 'OGD'
        where np.configuration_id in ($ids)
        and training_cost < 0.5"

    results = RunQuery(query, testDatabase)

    results[:LayerSize] = map(i -> parse(Int32, split(get(results[i, :layer_sizes]), ",")[2]), 1:size(results,1))
    results[:NumberLayers] = map(i -> size(split(get(results[i, :layer_sizes]), ","),1) - 2, 1:size(results,1))

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{String,1}(results[:,2])
    layer_returns = join(GetProfits(testDatabase), results, on = :configuration_id)

    groups = by(layer_returns, [:LayerSize, :NumberLayers], df -> median(df[:profit]))

    node_groups = by(groups, [:LayerSize], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8) #"ActualPL"], 8)

    for i in 1:size(node_groups,1)
        trace = scatter(;x=node_groups[i,2][:NumberLayers],y=node_groups[i,2][:x1], name=string(node_groups[i,1], " Hidden Layer Nodes"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end


    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> P&L (median) </br> </b>"))
        , xaxis = Dict(:dtick => 1.0, :title => string("<b> Number of Layers </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, "Network Size Profits - Lines.html"))
end

function PL_LearningRateReg_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
            tp.configuration_id,
            tp.min_learning_rate learning_rate,
            tp.l1_lambda lambda
        from training_parameters tp
        where configuration_id in ($ids)
            and category = 'FFN'
        order by learning_rate"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])

    results = join(GetProfits(testDatabase), results, on = :configuration_id)

    groups = by(results, [:lambda, :learning_rate], df -> median(df[:profit]))

    node_groups = by(groups, [:lambda], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in 1:size(node_groups,1)
        trace = scatter(;x=node_groups[i,2][:learning_rate],
                        y=node_groups[i,2][:x1],
                        name=string(node_groups[i,1], " Lambda"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> P&L </br> </b>"))
        , xaxis = Dict(:dtick => 1.0, :title => string("<b> Learning Rate </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " Learning Rate Lambda PL.html"))
end

function PL_SAE_Encoding_Size(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, learning_rate = nothing)

    ids = TransformConfigIDs(config_ids)

    learning_rate_clause = learning_rate == nothing ? "" : " and tp.learning_rate = $learning_rate"

    query = "select np.configuration_id,
                case when substr(layer_sizes, 0, instr(layer_sizes, ',')) == '$noneSize' then 0
                else cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT) end encoding
            from network_parameters np
                inner join training_parameters tp on tp.configuration_id = np.configuration_id and tp.category = 'FFN-OGD'
            where np.configuration_id in ($ids)
                $learning_rate_clause
            order by cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT)"

    ProfitBoxplot(query, :encoding, "Encoding Size", string(file_prefix, "Encoding Size Profits"), Int64, testDatabase, colourSetChoice)
end

function PL_DataDeltas(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, in_sample = false)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
             ('[' || deltas || '] Horizons') deltas
            from dataset_config np
            where np.configuration_id in($ids)"

    results = RunQuery(query, testDatabase)

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{String,1}(results[:,2])
    groups = by(results, [:deltas], df -> [df])

    ordering = groups[[1,3,2],1]


    ProfitBoxplot(query, :deltas, "", string(file_prefix, "Data Deltas Profits"), String, testDatabase, colourSetChoice, ordering, false, 18, nothing, in_sample)
end

function PL_Init(config_ids, file_prefix = "", colourSetChoice = "", testDatase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
                case when initialization like '%Xavier%' then 'Xavier'
                when initialization like '%He%' then 'He'
                when initialization like '%DC%' then 'He-Adj'
                else null end initialization
            from network_parameters
            where configuration_id in($ids)"

    ProfitBoxplot(query, :initialization, "Initialization", string(file_prefix, "Init Profits"), String, testDatase, colourSetChoice, nothing, true, default_fontsize)
end

function PL_OGDLearningDeltas_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
                dc.configuration_id,
                dc.deltas,
                tp.learning_rate
        from dataset_config dc
            inner join training_parameters tp on tp.configuration_id = dc.configuration_id and tp.category = 'FFN-OGD'
        where dc.configuration_id in ($ids)
        order by learning_rate"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{String,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])

    results = join(GetProfits(testDatabase), results, on = :configuration_id)

    groups = by(results, [:deltas, :learning_rate], df -> median(df[:profit]))

    node_groups = by(groups, [:deltas], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in [1,3,2]
        trace = scatter(;x=node_groups[i,2][:learning_rate],
                        y=node_groups[i,2][:x1],
                        name=string(node_groups[i,1], " Deltas"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> P&L </br> </b>"))
        , xaxis = Dict(:dtick => 0.01, :title => string("<b> OGD Learning Rate </b>"), :showticklabels => true )
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " OGD Learning Rate Deltas PL.html"))
end

function PL_OOS_OGDLR_Deltas(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, aggregation = median, in_sample = false, delta_order = [1, 3, 2])

    ids = TransformConfigIDs(config_ids)

    query = "select tp.configuration_id,
                ('[' || deltas || ']') deltas,
                tp.learning_rate
            from training_parameters tp
            inner join dataset_config dc on dc.configuration_id = tp.configuration_id
            where tp.configuration_id in ($ids)
                and tp.category = 'FFN-OGD'
            "

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{String,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    pl_returns = join(GetProfits(false, false), results, on = :configuration_id)

    groups = by(pl_returns, [:learning_rate, :deltas], df -> aggregation(df[:profit]))
    encoding_groups = by(groups, [:deltas], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in delta_order
        trace = scatter(;x=encoding_groups[i,2][:learning_rate],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " Horizons"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> OOS P&L (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> OGD Learning Rate</b>"))
        , font = Dict(:size => 18)
         )

    savefig(plot(data, l), string("./GraphOutputDir/OOS OGDLR-Delta P&L ", string(aggregation), ".html"))
end

function PL_OOS_OGDLR_Delta_Encoding(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, aggregation = median, encoding = nothing, delta_order = [1,3,2])

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
                case when substr(layer_sizes, 0, instr(layer_sizes, ',')) == '$noneSize' then 0
                else cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT) end encoding,
                tp.learning_rate,
                ('[' || deltas || ']') deltas
            from network_parameters np
            inner join training_parameters tp on tp.configuration_id = np.configuration_id and tp.category = 'FFN-OGD'
            inner join dataset_config dc on dc.configuration_id = np.configuration_id
            where np.configuration_id in ($ids)
                and layer_sizes like '$encoding,%'
            "

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Int64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    results[:,4] = Array{String,1}(results[:,4])
    pl_returns = join(GetProfits(false, false), results, on = :configuration_id)

    aggregation = median
    groups = by(pl_returns, [:deltas, :learning_rate], df -> aggregation(df[:profit]))
    encoding_groups = by(groups, [:deltas], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in delta_order
        trace = scatter(;x=encoding_groups[i,2][:learning_rate],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " Horizons"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> OOS P&L (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> OGD Learning Rate </b>"))
        , font = Dict(:size => 18)
         )

    savefig(plot(data, l), string("./GraphOutputDir/OOS OGDLR-Delta-Encoding ", string(encoding), "_", string(aggregation), ".html"))
end

function PL_IS_Encoding(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, aggregation = median)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
                case when substr(layer_sizes, 0, instr(layer_sizes, ',')) == '$noneSize' then 0
                else cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT) end encoding
            from network_parameters np
            where np.configuration_id in ($ids)"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Int64,1}(results[:,2])
    pl_returns = join(GetProfits(false, true), results, on = :configuration_id)

    #Box Plot
    #groups = by(pl_returns, [:encoding], df -> [df])
    #general_boxplot(groups, "Encoding Size", "IS Encoding P&L Box", :profit, "IS P&L", true, colourSetChoice, 16)

    #Median Plot

    encoding_groups = by(pl_returns, [:encoding], df -> aggregation(df[:profit]))
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    trace = scatter(;x=encoding_groups[:encoding],
                     y=encoding_groups[:x1],
                     name="Encoding",
                    marker = Dict(:line => Dict(:width => 2, :color => colors[1]), :color=>colors[1]))
    push!(data, trace)

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> IS P&L (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
        , font = Dict(:size => 18)
         )

    savefig(plot(data, l), string("./GraphOutputDir/IS Encoding P&L ", string(aggregation), ".html"))
end

function PL_SAE_Encoding_SizeLines(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, learning_rate = nothing, aggregation = maximum)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
                case when substr(layer_sizes, 0, instr(layer_sizes, ',')) == '$noneSize' then 0
                else cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT) end encoding,
                tp.learning_rate
            from network_parameters np
                inner join training_parameters tp on tp.configuration_id = np.configuration_id and tp.category = 'FFN-OGD'
                inner join dataset_config dc on dc.configuration_id = np.configuration_id
            where np.configuration_id in ($ids)
            order by cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT)"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Int64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    pl_returns = join(GetProfits(testDatabase, false), results, on = :configuration_id)

    groups = by(pl_returns, [:encoding, :learning_rate], df -> aggregation(df[:profit]))

    encoding_groups = by(groups, [:learning_rate], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets["ActualPL"], 8)

    for i in 1:size(encoding_groups,1)
        trace = scatter(;x=encoding_groups[i,2][:encoding],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " OGD Learning Rate"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> P&L (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, "Encoding PL Learning Rates", string(aggregation), ".html"))
end

#MSE BoxPlots############################################################################

function MSE_SAE_Encoding_Size_Deltas_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
                dc.configuration_id,
                dc.deltas,
                layer_sizes,
                min(training_cost) cost
        from dataset_config dc
            inner join network_parameters np on np.configuration_id = dc.configuration_id
            inner join epoch_records er on er.configuration_id = dc.configuration_id
        where dc.configuration_id in ($ids)
        group by dc.configuration_id, dc.deltas, layer_sizes
        having cost not null
        order by deltas"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{String,1}(results[:,2])
    results[:,3] = Array{String,1}(results[:,3])
    results[:,4] = Array{Float64,1}(results[:,4])


    results[:encoding] =  map(s -> parse(Int64, s), (ascii.(map(l -> split(l, ',')[1], results[:,:layer_sizes]))))

    groups = by(results, [:deltas, :encoding], df -> mean(df[:cost]))

    node_groups = by(groups, [:deltas], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in [1,3,2]
        trace = scatter(;x=node_groups[i,2][:encoding],
                        y=node_groups[i,2][:x1],
                        name=string(node_groups[i,1], " Deltas"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Median MSE </br> </b>"))
        , xaxis = Dict(:dtick => 5, :title => string("<b> Encoding Size </b>"), :showticklabels => true )
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " Encoding Size Deltas MSE.html"))
end

function OGDMSE_SAE_Encoding_SizeLines(config_ids, noneSize = -1, file_prefix = "", colourSetChoice = "", testDatabase = false, learning_rate = nothing, aggregation = maximum)

    ids = TransformConfigIDs(config_ids)

    query = "select np.configuration_id,
                case when substr(layer_sizes, 0, instr(layer_sizes, ',')) == '$noneSize' then 0
                else cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT) end encoding,
                tp.learning_rate,
                er.training_cost
            from network_parameters np
                inner join training_parameters tp on tp.configuration_id = np.configuration_id and tp.category = 'FFN-OGD'
                inner join epoch_records er on er.configuration_id = np.configuration_id and er.category = 'OGD'
            where np.configuration_id in ($ids)
                and training_cost is not null
            order by cast(substr(layer_sizes, 0, instr(layer_sizes, ',')) as INT)"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Int64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    results[:,4] = Array{Float64,1}(results[:,4])

    groups = by(results, [:encoding, :learning_rate], df -> aggregation(df[:training_cost]))

    encoding_groups = by(groups, [:learning_rate], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets["ActualPL"], 8)

    for i in 1:size(encoding_groups,1)
        trace = scatter(;x=encoding_groups[i,2][:encoding],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " OGD Learning Rate"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> OGD MSE (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>")))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, "Encoding OGD MSE Learning Rates", string(aggregation), ".html"))
end

function MSE_ActivationsEncodingSizes(config_ids, encoding_size = nothing, file_prefix = "", colourSetChoice = "", testDatabase = false)

    function general_boxplot2(layer_groups, prefix, fn, variable_name)

        y_vals = layer_groups[5,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[5,1]))
        data = [trace]

        #for i in 2:size(layer_groups,1)
        for i in (6, 7, 8, 9, 10, 1, 2, 3, 4)
            y_vals = layer_groups[i,2][:,variable_name]
            trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
            push!(data, trace)
        end
        plot(data)
        l = Layout(width = 1200, height = 600, margin = Dict(:b => 120, :l => 100)
            , yaxis = Dict(:title => string("<b>", "MSE", "<br> </b>"))
            , xaxis = Dict(:title => string("<b>", "Encoding-Activation", "<br> </b>"))
            , font = Dict(:size => 18))

        savefig(plot(data, l), string("./GraphOutputDir/", fn, ".html"))
    end

    ids = TransformConfigIDs(config_ids)

    query = "select
            er.configuration_id,
            min(training_cost) cost,
            np.output_activation,
            np.encoding_activation,
            experiment_set_name,
            min(er.learning_rate) lr,
            dc.scaling_function,
            np.layer_sizes
        from epoch_records er
        inner join configuration_run cr on cr.configuration_id = er.configuration_id
        inner join dataset_config dc on dc.configuration_id = er.configuration_id
        inner join network_parameters np on np.configuration_id = er.configuration_id
        where er.configuration_id in ($ids)
        and er.category = 'SAE-SGD-Init'
        and training_cost is not null
        and output_activation not like 'Relu%'
        and scaling_function not like 'Standardize%'
        group by er.configuration_id
        having cost < 1000
        order by cost desc"

    results = RunQuery(query, testDatabase)

    #By encoding; output activation; primary activation; encoding layer & network size
    results[:network] =  map(t -> replace(t[(findin(t, ",")[1]+1):end], ",", "x"), Array(results[:layer_sizes]))
    results[:primary_activation] =  ascii.(Array(map(e -> split(split(e, " ")[end], "_")[1], results[:experiment_set_name])))
    results[:encoding_size] = Array(map(n -> split(n, "x")[end], results[:network]))

    results[:config_group] = string.(results[:encoding_size]
                                    , "-"
                                    ,replace.(results[:primary_activation], "Activation", ""))

    filename = string(file_prefix, "SAE Activations And Encoding Sizes Min MSE")

    if encoding_size != nothing
        filtered_indices = Array(results[:encoding_size] .== string(encoding_size))
        filtered_results = results[filtered_indices, :]
        filename = string(filename, " Encoding " , encoding_size)
    else
        filtered_results = results
    end

    groups = by(filtered_results, [:config_group], df -> [df])

    general_boxplot2(groups, " ", filename, :cost)
end

function MSE_LayerSizesLines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
            er.configuration_id,
            min(training_cost) cost,
            np.layer_sizes
        from epoch_records er
        inner join configuration_run cr on cr.configuration_id = er.configuration_id
        inner join network_parameters np on np.configuration_id = er.configuration_id
        where er.configuration_id in ($ids)
            and er.category = 'SAE-SGD-Init'
            and training_cost is not null
        group by er.configuration_id
        having cost < 1000
        order by layer_sizes"

    results = RunQuery(query, testDatabase)

    results[:LayerSize] = map(i -> parse(Int32, split(get(results[i, :layer_sizes]), ",")[2]), 1:size(results,1))
    results[:NumberLayers] = map(i -> size(split(get(results[i, :layer_sizes]), ","),1) - 2, 1:size(results,1))

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{String,1}(results[:,3])

    groups = by(results, [:LayerSize, :NumberLayers], df -> median(df[:cost]))

    node_groups = by(groups, [:LayerSize], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in 1:size(node_groups,1)
        trace = scatter(;x=node_groups[i,2][:NumberLayers],y=node_groups[i,2][:x1], name=string(node_groups[i,1], " Hidden Layer Nodes"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE (median) </br> </b>"))
        , xaxis = Dict(:dtick => 1.0, :title => string("<b> Number of Layers </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, "Network Size MSE.html"))
end

function MSE_LearningRateReg_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
            er.configuration_id,
            min(training_cost) cost,
            tp.learning_rate learning_rate,
            tp.l1_lambda lambda
        from epoch_records er
        inner join configuration_run cr on cr.configuration_id = er.configuration_id
        inner join training_parameters tp on tp.configuration_id = er.configuration_id and tp.category = 'SAE'
        where er.configuration_id in ($ids)
            and er.category = 'SAE-SGD-Init'
            and training_cost is not null
        group by er.configuration_id
        having cost < 1000
        order by tp.learning_rate"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    results[:,4] = Array{Float64,1}(results[:,4])

    groups = by(results, [:lambda, :learning_rate], df -> median(df[:cost]))

    node_groups = by(groups, [:lambda], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in 1:size(node_groups,1)
        trace = scatter(;x=node_groups[i,2][:learning_rate],
                        y=node_groups[i,2][:x1],
                        name=string(node_groups[i,1], " Lambda"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE </br> </b>"))
        , xaxis = Dict(:dtick => 1.0, :title => string("<b> Learning Rate </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " Learning Rate Lambda MSE.html"))
end

function MSE_LearningRateInit_Lines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select
            er.configuration_id,
            min(training_cost) cost,
            tp.min_learning_rate learning_rate,
            np.initialization
        from epoch_records er
        inner join configuration_run cr on cr.configuration_id = er.configuration_id
        inner join network_parameters np on np.configuration_id = er.configuration_id
        inner join training_parameters tp on tp.configuration_id = er.configuration_id and tp.category = 'SAE'
        where er.configuration_id in ($ids)
            and er.category = 'SAE-SGD-Init'
            and training_cost is not null
        group by er.configuration_id
        having cost < 1000
        order by tp.learning_rate"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    results[:,4] = Array{String,1}(results[:,4])

    groups = by(results, [:initialization, :learning_rate], df -> median(df[:cost]))

    node_groups = by(groups, [:initialization], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()

    colors = colors = ColorBrewer.palette(colourSets["ActualMSE"], 8)

    for i in 1:size(node_groups,1)
        trace = scatter(;x=node_groups[i,2][:learning_rate],
                        y=node_groups[i,2][:x1],
                        name=string(node_groups[i,1], " Init"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE </br> </b>"))
        , xaxis = Dict(:dtick => 1.0, :title => string("<b> Learning Rate </b>"))
        , font = Dict(:size => default_fontsize))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " Learning Rate Init MSe.html"))
end

function MSE_LayerSizes(config_ids, encoding_size = nothing, file_prefix = "",  colourSetChoice = "", testDatabase = false, ordering = nothing)

    ids = TransformConfigIDs(config_ids)

    query = "select
            er.configuration_id,
            min(training_cost) cost,
            layer_sizes
        from epoch_records er
        inner join configuration_run cr on cr.configuration_id = er.configuration_id
        inner join network_parameters np on np.configuration_id = er.configuration_id
        where er.configuration_id in ($ids)
            and er.category = 'SAE-SGD-Init'
            and training_cost is not null
        group by er.configuration_id
        having cost < 1000
        order by layer_sizes"

    if ordering != nothing
        ordering = map(i -> string("Layers: ", i), ordering)
    end

    results = RunQuery(query, testDatabase)
    results[:layers] = Array(map(t -> string("Layers: ", t[4:(findin(t, ",")[end]-1)]), Array(results[:layer_sizes])))

    groups = by(results, [:layers], df -> [df])
    groups = OrderDataFrame(groups, ordering)

    general_boxplot(groups, "Network Layers ", string(file_prefix, "SAE Layer Sizes Min MSE"), :cost, "MSE", false, "SyntheticMSE")
end

function MSE_LearningRate_MaxMin(config_ids, init, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    init_clause = init == nothing ? "" : " and initialization like '%$init%'"

    lr_msequery = "select
                    tp.configuration_id,
                    min(testing_cost) cost,
                    replace(('' || cast(tp.learning_rate as text) || '-' ||  cast(min_learning_rate as text)), '1.0e-05', '0.00001') learning_rates
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                $init_clause
                group by tp.configuration_id, tp.learning_rate
                having min(testing_cost) is not null
                order by learning_rates"

    r = RunQuery(lr_msequery)

    MSEBoxplot(lr_msequery, :learning_rates, "Max-Min Learning Rates", string(file_prefix, "SAE Max Learning Rate Min Test MSE"), String, colourSetChoice, testDatabase, true, default_fontsize)
end

function MSE_LearningRate_MaxMin_Lines(config_ids, init, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    init_clause = init == nothing ? "" : " and initialization like '%$init%'"

    lr_msequery = "select
                    tp.configuration_id,
                    min(testing_cost) cost,
                    replace(('' || cast(tp.learning_rate as text) || '-' ||  cast(min_learning_rate as text)), '1.0e-05', '0.00001') learning_rates,
                    layer_sizes
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                $init_clause
                group by tp.configuration_id, tp.learning_rate, layer_sizes
                having min(testing_cost) is not null
                order by learning_rates"

        results = RunQuery(lr_msequery, testDatabase)
        results[:,1] = Array{Int64,1}(results[:,1])
        results[:,2] = Array{Float64,1}(results[:,2])
        results[:,3] = Array{String,1}(results[:,3])
        results[:,4] = Array{String,1}(results[:,4])

        results[:encoding] = results[:layer_sizes]#map(i -> parse(Int64, ascii(i[end])), split.(results[:layer_sizes], ','))

        groups = by(results, [:encoding, :learning_rates], df -> aggregation(df[:cost]))


        encoding_groups = by(groups, [:learning_rates], df -> [df])
        data = Array{PlotlyBase.GenericTrace,1}()
        colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

        for i in 1:size(encoding_groups, 1)
            trace = scatter(;x=encoding_groups[i,2][:encoding],
                             y=encoding_groups[i,2][:x1],
                             name=string(encoding_groups[i,1],
                             " "),
                            marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
            push!(data, trace)
        end

        l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
            , yaxis = Dict(:title => string("<b> MSE (", string(aggregation), ")</br> </b>"))
            , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
            , font = Dict(:size => default_fontsize)
             )

        savefig(plot(data, l), string("./GraphOutputDir/SAE MSE Learning Rates by Encoding ", string(aggregation), ".html"))
end

function MSE_Lambda1(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select tp.configuration_id, min(testing_cost) cost,
                            (cast(tp.l1_lambda as string) || ' lambda') l1_lambda
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.l1_lambda
                having cost not null"

    MSEBoxplot(lr_msequery, :l1_lambda, "L1 Lambda ", string(file_prefix, "SAE L1 Reg Min Test MSE"), String, colourSetChoice, testDatabase)
end

function MSE_EpochCycle(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select tp.configuration_id,
                            min(testing_cost) cost,
                            cast(epoch_cycle_max as string) || ' Epochs' epoch_cycle_max
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, epoch_cycle_max
                having cost not null
                order by epoch_cycle_max"

    MSEBoxplot(query, :epoch_cycle_max, "Learning Rate Epoch Cycles ", string(file_prefix, "SAE MSE Epoch Cycles"), String, colourSetChoice, testDatabase)
end

function MSE_EpochCycle_LayersLines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select tp.configuration_id,
                            min(testing_cost) cost,
                            cast(epoch_cycle_max as string) || ' Epochs' epoch_cycle_max,
                            layer_sizes
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, epoch_cycle_max, layer_sizes
                having cost not null
                order by epoch_cycle_max"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{String,1}(results[:,3])
    results[:,4] = Array{String,1}(results[:,4])

    results[:encoding] = map(i -> parse(Int64, ascii(i[end])), split.(results[:layer_sizes], ','))

    groups = by(results, [:encoding, :epoch_cycle_max], df -> aggregation(df[:cost]))


    encoding_groups = by(groups, [:epoch_cycle_max], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in 1:size(encoding_groups, 1)
        trace = scatter(;x=encoding_groups[i,2][:encoding],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " "),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
        , font = Dict(:size => default_fontsize)
         )

    savefig(plot(data, l), string("./GraphOutputDir/SAE MSE Epoch Cycles by Encoding ", string(aggregation), ".html"))
end

function MSE_Reg_EncodingLines(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false, aggregation = minimum)

    ids = TransformConfigIDs(config_ids)

    query = "select tp.configuration_id,
                            min(testing_cost) cost,
                            l1_lambda,
                            layer_sizes
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, l1_lambda, layer_sizes
                having cost not null
                order by epoch_cycle_max"

    results = RunQuery(query, testDatabase)
    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{Float64,1}(results[:,3])
    results[:,4] = Array{String,1}(results[:,4])

    results[:encoding] = map(i -> ascii(i[end]), split.(results[:layer_sizes], ','))

    groups = by(results, [:encoding, :l1_lambda], df -> aggregation(df[:cost]))


    encoding_groups = by(groups, [:encoding], df -> [df])
    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in [5, 1,2,3,4]
        trace = scatter(;x=encoding_groups[i,2][:l1_lambda],
                         y=encoding_groups[i,2][:x1],
                         name=string(encoding_groups[i,1],
                         " Encoding Size"),
                        marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> L1 Lambda </b>"))
        , font = Dict(:size => default_fontsize)
         )

    savefig(plot(data, l), string("./GraphOutputDir/SAE MSE l1_lambda by Encoding ", string(aggregation), ".html"))

end

function MSE_Init(config_ids, encoding_size, file_prefix = "", colourSetChoice = "", testDatabase = false)
    ids = TransformConfigIDs(config_ids)

    encoding_clause = encoding_size == nothing ? "" : " and layer_sizes like '%,$encoding_size'"

    init_query = string("select er.configuration_id,
                                min(testing_cost) cost,
                                case when initialization like '%Xavier%' then 'Xavier'
                                when initialization like '%He%' then 'He'
                                when initialization like '%DC%' then 'He-Adj'
                                else null end init
                            from epoch_records er
                            inner join network_parameters np on np.configuration_id = er.configuration_id
                            where np.category = \"SAE\"
                                and testing_cost is not null
                            and er.configuration_id in ($ids)
                            $encoding_clause
                            group by er.configuration_id, init
                            ")

    MSEBoxplot(init_query, :init, "Initialization", string(file_prefix, " SAE Inits Min Test MSE"), String, colourSetChoice, testDatabase, true, 14)
end

function MSE_Deltas(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    delta_query = string("select er.configuration_id, min(testing_cost) cost, ('[' || deltas || '] Horizons') deltas
                        from epoch_records er
                        inner join network_parameters np on np.configuration_id = er.configuration_id
                        inner join dataset_config dc on dc.configuration_id = er.configuration_id
                        where np.category = 'SAE'
                            and er.configuration_id in ($ids)
                        group by er.configuration_id, deltas
                        having cost not null")

    results = RunQuery(delta_query, testDatabase)

    results[:,1] = Array{Int64,1}(results[:,1])
    results[:,2] = Array{Float64,1}(results[:,2])
    results[:,3] = Array{String,1}(results[:,3])
    groups = by(results, [:deltas], df -> [df])

    ordering = groups[[1,3,2],1]

    MSEBoxplot(delta_query, :deltas, "", string(file_prefix, "SAE Delta MSE"), String, colourSetChoice, testDatabase, false, 18, ordering)
end

function MSE_Min_EncodingSize_Activation(config_ids, activation_choice = "encoding", file_prefix = "", colourSetChoice = "", testDatabase = false, aggregation = minimum)

    #Exclude Standardize
    #Linear output only
    #maxcost < 1000

    activation_clause = activation_choice == "encoding" ?
                        "replace(encoding_activation, 'Activation', '')" :
                        "replace(substr(layer_activations, 0, instr(layer_activations, ',')), 'Activation', '')"

    ids = TransformConfigIDs(config_ids)

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    layer_sizes,
                    $activation_clause activation
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id

            where er.configuration_id in ($ids)
                and er.category = \"SAE-SGD-Init\"
                and training_cost is not null
                and scaling_function not like 'Standardize%'
                and output_activation not like 'Relu%'
            group by er.category,er.configuration_id, layer_sizes, activation
            having cost < 1000
            "

    results = RunQuery(query, testDatabase)
    results[:encoding_size] = map(i -> parse(Int32, split(i, ',')[end]), Array(results[:layer_sizes]))

    groups = by(results, [:encoding_size, :activation], df -> aggregation(Array(df[:cost])))
    line_groups = by(groups, [:activation], df -> [df])

    data = Array{PlotlyBase.GenericTrace,1}()
    colors = colors = ColorBrewer.palette(colourSets[colourSetChoice], 8)

    for i in 1:size(line_groups,1)
        trace = scatter(;x=line_groups[i, 2][:encoding_size],
                         y=line_groups[i,2][:x1],
                         name=string(get(line_groups[i,1])),
                         marker = Dict(:line => Dict(:width => 2, :color => colors[i+1]), :color=>colors[i+1]))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> MSE (", string(aggregation), ")</br> </b>"))
        , xaxis = Dict(:title => string("<b> Encoding Size </b>"))
        , font = Dict(:size => 18))

    savefig(plot(data, l), string("./GraphOutputDir/", file_prefix, " Encoding by Activation MSE.html"))

end

function MSE_Activation_Scaling_Filters(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    replace((substr(layer_activations, 0, instr(layer_activations, ',')) || '-' ||
                            np.encoding_activation || '-' ||
                            np.output_activation || '-' ||
                            dc.scaling_function
                    ), 'Activation', '') act_scaling
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, act_scaling
            having cost < $maxCost
            "

    MSEBoxplot(query, :act_scaling, "", string(file_prefix, "SAE ScalingOutput Min MSE"), String, colourSetChoice, testDatabase, false)
end

function MSE_OutputActivation_Scaling_Filters(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    replace(np.output_activation || '-' || dc.scaling_function, 'Activation', '') act_scaling
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, act_scaling
            having cost < $maxCost
            "

    MSEBoxplot(query, :act_scaling, "", string(file_prefix, "SAE ScalingOutput Min MSE"), String, colourSetChoice, testDatabase, false)
end

function MSE_Output_Activation(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    replace(output_activation, 'Activation', ' Output') output_activation
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, output_activation
            having cost < $maxCost
            "

    MSEBoxplot(query, :output_activation, "", string(file_prefix, "Output Activation Activation Min MSE"), String, colourSetChoice, testDatabase, true)
end

function MSE_Encoding_Activation(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    encoding_activation
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, encoding_activation
            having cost < $maxCost
            "

    MSEBoxplot(query, :encoding_activation, "", string(file_prefix, "Encoding Activation Activation Min MSE"), String, colourSetChoice, testDatabase, false)
end

function MSE_Hidden_Activation(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    replace((substr(layer_activations, 0, instr(layer_activations, ','))), 'Activation', '') act_scaling
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, act_scaling
            having cost < $maxCost
            "

    MSEBoxplot(query, :act_scaling, "", string(file_prefix, "Hidden Layer Activation Min MSE"), String, colourSetChoice, testDatabase, false)
end

function MSE_Scaling_Filters(config_ids, includeStandardize, maxCost, excludeReluOutput, layerSize, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    andClause = includeStandardize ? " and true " : " and scaling_function not like 'Standardize%' "
    andReluClause = excludeReluOutput ? " and output_activation not like 'Relu%' " : " and true "
    andSizeClause = layerSize == nothing ? " and true " : " and layer_sizes like '%,$layerSize'"

    query = "select er.configuration_id,
                    min(training_cost) cost,
                    scaling_function
            from epoch_records er
            inner join configuration_run cr on cr.configuration_id = er.configuration_id
            inner join dataset_config dc on dc.configuration_id = er.configuration_id
            inner join network_parameters np on np.configuration_id = er.configuration_id
            where er.configuration_id in ($ids)
            and er.category = \"SAE-SGD-Init\"
            and training_cost is not null
            $andClause
            $andReluClause
            $andSizeClause
            group by er.category,er.configuration_id, scaling_function
            having cost < $maxCost
            "

    MSEBoxplot(query, :scaling_function, "", string(file_prefix, "SAE ScalingOutput Min MSE"), String, colourSetChoice, testDatabase, true)
end

function MSE_Pretraining(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    query = "select er.configuration_id, min(er.testing_cost) cost,
                (cast(ifnull(max(er2.epoch_number), 0) as string) || ' ' || 'Epochs' ) pre_training_epochs

            from epoch_records er
            left join epoch_records er2 on er.configuration_id = er2.configuration_id and er2.category = 'RBM-CD'
            inner join training_parameters tp on tp.configuration_id = er.configuration_id and tp.category = 'RBM-CD'
            where er.category like \"SAE-SGD%\"
                and er.configuration_id in ($ids)
            group by er.configuration_id
            having cost not null"

    MSEBoxplot(query, :pre_training_epochs, "Pretraining Epochs ", string(file_prefix, "SAE Pre-training Learning Rates epochs Min Test MSE"), String, colourSetChoice, testDatabase)
end

function MSE_Denoising(config_ids, file_prefix = "", colourSetChoice = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    dn_mse_query = "select tp.configuration_id,
                            min(testing_cost) cost,
                            (' ' || cast(denoising_variance as string)) denoising_variance
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                and denoising_variance >= 0.000001
                group by tp.configuration_id, denoising_variance
                having training_cost not null
                order by denoising_variance"

    MSEBoxplot(dn_mse_query, :denoising_variance, "Denoising Variance", string(file_prefix, "Denoising Variance Min MSE"), String, colourSetChoice, testDatabase)
end

##Heatmaps######################################################################

function GenericHeatmap(results, filename, testDatabase = false)

    var_one = names(results)[2]
    var_two = names(results)[3]

    results[:,1] = Array(results[:,1])
    results[:,2] = Array(results[:,2])
    results[:,3] = Array(results[:,3])
    results = join(GetProfits(testDatabase), results, on = :configuration_id)

    comb_pl = by(results, [var_one, var_two], df -> mean(df[:profit]))

    var_one_order = Array(unique(comb_pl[var_one]))
    var_two_order = Array(unique(comb_pl[var_two]))

    vals = Array{Float64,2}(fill(NaN, length(var_one_order), length(var_two_order)))

    for r in 1:size(comb_pl, 1)
        l_index = findfirst(var_one_order, comb_pl[r,1])
        r_index = findfirst(var_two_order, comb_pl[r,2])
        vals[l_index, r_index] = comb_pl[r,3]
    end

    xaxis_label = string(var_one)
    yaxis_label = string(var_two)

    l = Layout(width = 1000, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b>", yaxis_label, "<br> </b>"))
        , xaxis = Dict(:title => string("<b>", xaxis_label, "<br> </b>"), :showticklabels => true ))


    yvals = map(i -> string(": ", i), var_two_order)
    xvals = map(i -> string(": ", i), var_one_order)
    trace = heatmap(z = vals, y = yvals, x = xvals, colorscale ="Portland")
    data = [trace]
    savefig(plot(data, l), string("./GraphOutputDir/", filename, ".html"))
end

function PL_Heatmap_LearningRate_MaxEpochs(config_ids, file_prefix = "", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    results_query = string("select tp1.configuration_id, tp1.max_epochs, tp.learning_rate
                            from training_parameters tp1
                            inner join training_parameters tp on tp.configuration_id = tp1.configuration_id and tp.category = 'FFN-OGD'
                            where tp1.configuration_id in ($ids)
                            and tp1.category = 'FFN'
                            order by tp1.max_epochs, tp.learning_rate
                            ")
    results = RunQuery(results_query)

    GenericHeatmap(results, string(file_prefix, "PL Heatmap Learning Rate - Max Epochs"), testDatabase)
end

function PL_Heatmap_NetworkSize_DataAggregation(config_ids, file_prefix ="", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    results_query = string("select np.configuration_id,
                                    layer_sizes,
                                   ('Deltas ' || (case when deltas = '1,5,20' then '1: '
                                        when deltas = '5,20,60' then '2: '
                                        when deltas = '10,20,60' then '3: ' end)
                                        || deltas) deltas

                            from network_parameters np
                            inner join dataset_config dc on np.configuration_id = dc.configuration_id
                            where np.configuration_id in ($ids)
                            order by layer_sizes, deltas
                            ")
    results = RunQuery(results_query)

    results[:number_layers] = map(i -> size(split(get(results[:layer_sizes][i]),','),1)-2, 1:size(results,1))

    new_results = results[[:configuration_id, :deltas, :number_layers]]

    GenericHeatmap(new_results, string(file_prefix, "PL Heatmap Network Size - Deltas"), testDatabase)
end

function PL_Heatmap_LayerSize_DataAggregation(config_ids, file_prefix ="", testDatabase = false)

    ids = TransformConfigIDs(config_ids)

    results_query = string("select np.configuration_id,
                                    case when layer_sizes like '%60%' then '1: 60'
                                        when layer_sizes like '%120%' then '2: 120'
                                        when layer_sizes like '%240%' then '3: 240' end layer_sizes,

                                   ('Deltas ' || (case when deltas = '1,5,20' then '1: '
                                        when deltas = '5,20,60' then '2: '
                                        when deltas = '10,20,60' then '3: ' end)
                                        || deltas) deltas

                            from network_parameters np
                            inner join dataset_config dc on np.configuration_id = dc.configuration_id
                            where np.configuration_id in ($ids)
                            order by  deltas, layer_sizes
                            ")
    results = RunQuery(results_query)

    #results[:layer_sizes] = map(i -> split(get(results[:layer_sizes][i]),',')[2], 1:size(results,1))

    new_results = results[[:configuration_id, :deltas, :layer_sizes]]

    GenericHeatmap(new_results, string(file_prefix, "PL Heatmap Layer Size - Deltas"), testDatabase)
end

##Data Distributions############################################################

function GenerateDeltaDistribution(delta_values, variances, time_steps, dataset, dataseed, file_prefix = "", colourSetChoice = "")


    data_config = DatasetConfig(dataseed, #75,
                                "none",
                                time_steps, #5000,  #timesteps
                                [0,0,0], #delta aggregatios
                                [1.0], #process split (for SAE/SGD & OGD)
                                [1.0], #validation set split
                                [1], #prediction step
                                variances, #
                                LimitedNormalizeData) #scaling function

    colours = ColorBrewer.palette(colourSets[colourSetChoice], 8)
    opacity_value = 0.5

    std_dict = Dict()
    mean_dict = Dict()


    data_config.deltas = delta_values[1]
    training_data = PrepareData(data_config, dataset, nothing)[1]
    vals = mapreduce(i -> training_data.training_input[:, i], vcat, 1:size(training_data.training_input,2))

    trace = histogram(;x=vals, name=string(string(delta_values[1]), " Horizons"), marker = Dict(:color=>colours[1]), opacity=opacity_value)
    mean_dict[string(delta_values[1])] = mean(vals)
    std_dict[string(delta_values[1])] = std(vals)

    data = [trace]

    for d in 2:size(delta_values,1)

        data_config.deltas = delta_values[d]
        training_data = PrepareData(data_config, dataset, nothing)[1]
        vals = mapreduce(i -> training_data.training_input[:, i], vcat, 1:size(training_data.training_input,2))

        trace = histogram(;x=vals, name=string(string(delta_values[d]), " Horizons"), marker = Dict(:color=>colours[d]), opacity=opacity_value)

        #trace = bar(;x=vals, name=string(string(delta_values[d]), " Data Windows"), marker = Dict(:color=>colours[d]), opacity=opacity_value)
        mean_dict[string(delta_values[d])] = mean(vals)
        std_dict[string(delta_values[d])] = std(vals)

        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Number of Observations </br> </b>"))
        , xaxis = Dict(:title => string("<b> Log Difference</b>"))
        , barmode="overlay"
        , font = Dict(:size => 18))

    fig = Plot(data, l)
    savefig(plot(fig), string("./GraphOutputDir/", file_prefix, " Aggregation Distributions.html"))

    return (mean_dict, std_dict, fig)
end

##Price Line Plots##############################################################

function PlotSynthetic6(dataseed)
    dataseed = 75
    var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))

    names = ("Strong Upward High",
            "Strong Upward Low",
            "Upward High",
            "Upward Low",
            "Strong Downward High",
            "Strong Downward")


    ds = GenerateDataset(dataseed, 5000, var_pairs)


    for r in size(ds, 1):-1:1
        for c in 1:size(ds, 2)
            ds[r,c] = ds[r,c]./ds[1,c]
        end
    end

    data = [scatter(;x=1:5001,y=ds[:,1], name=names[1])]

    for i in 2:size(ds, 2)
        trace = scatter(;x=1:5001,y=ds[:,i], name=names[i])
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Stock Value </br> </b>"))
        , xaxis = Dict(:title => string("<b> Time Step </b>")))

    savefig(plot(data, l), string("./GraphOutputDir/PlotSynthetic6.html"))
end

function PlotSynthetic10(dataseed)
    dataseed = 75
    var_pairs = ((0.9,0.5),
                (0.7, 0.2),
                (0.05, 0.5),
                (0.05, 0.4),
                (0.04, 0.1),
                (0.02, 0.15),
                (0.01, 0.05),
                (-0.1, 0.2),
                (-0.4, 0.15),
                (-0.8, 0.55))

    names = ("Strong Upward High",
            "Strong Upward Low",
            "Upward High",
            "Upward High",
            "Upward Low",
            "Sideways Low",
            "Sideways Low",
            "Downwards Low",
            "Strong Downward Low",
            "Strong Downward High")


    ds = GenerateDataset(dataseed, 5000, var_pairs)

    for r in size(ds, 1):-1:1
        for c in 1:size(ds, 2)
            ds[r,c] = ds[r,c]./ds[1,c]
        end
    end

    data = [scatter(;x=1:5001,y=ds[:,1], name=names[1])]

    for i in 2:size(ds, 2)
        trace = scatter(;x=1:5001,y=ds[:,i], name=names[i])
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Stock Value </br> </b>"))
        , xaxis = Dict(:title => string("<b> Time Step </b>")))

    savefig(plot(data, l), string("./GraphOutputDir/PlotSynthetic10.html"))
end

function PlotJSEPlots()
    PlotJSEPrices([:ACL, :AGL, :AMS, :AOD, :BAW, :BIL, :BVT, :CFR, :CRH, :DDT], "Scaling10")
    PlotJSEPrices([:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT], "Actual10")
    PlotJSEPrices([:AGL, :ACL], "AGLACL")
    PlotJSEPrices([:AGL], "AGL")
end

function PlotJSEPrices(stock_names, file_suffix)

    jsedata = ReadJSETop40Data()

    dataset = jsedata[:, stock_names]

    data = [scatter(;x=1:size(dataset, 1),y=dataset[:,1], name=string("Stock ", string(names(dataset)[1])))]

    for i in 2:size(dataset, 2)
        trace = scatter(;x=1:size(dataset, 1),y=dataset[:,i], name=string("Stock ", string(names(dataset)[i])))
        push!(data, trace)
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Stock Value </br> </b>"))
        , xaxis = Dict(:title => string("<b> Time Step </b>")))

    savefig(plot(data, l), string("./GraphOutputDir/Plot",file_suffix,".html"))
end

function StockPricePlot(dataseed)
    var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
    ds = GenerateDataset(dataseed, 5000, var_pairs)

    price_plot = plot(Array(ds), name = ["stock 1" "stock 2" "stock 3" "stock 4" "stock 5" "stock 6"])
    savefig(price_plot, "./GraphOutputDir/PriceGraphs", export_ext)
end

function RecreateStockPrices(config_names)
    configs = mapreduce(x->string(x, ","), string, collect(keys(config_names)))[1:(end-1)]
    best_query = string("select * from prediction_results where configuration_id in ($configs)")
    best_results = RunQuery(best_query)
    best_groups = by(best_results, [:stock], df -> [df])

    for i in 1:size(best_groups,1)
        timesteps = best_groups[i,2][:time_step]
        config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

        #actual = cumsum(config_groups[1,2][:actual])
        #predicted_one = cumsum(config_groups[1,2][:predicted])
        #predicted_two = cumsum(config_groups[2,2][:predicted])

        actual = (config_groups[1,2][:actual])
        predicted_one = (config_groups[1,2][:predicted])
        predicted_two = (config_groups[2,2][:predicted])

        stock_name = get(best_groups[i,1])

        t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
        t1 = scatter(;y=predicted_one, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][1])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))
        t2 = scatter(;y=predicted_two, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][2])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))

        recreation_plots = [t0, t1, t2]
        filename = string("recreation_", stock_name, "_", collect(keys(config_names))[1])
        savefig(plot(recreation_plots), string("./GraphOutputDir/", filename, export_ext))

    end
end

function RecreateStockPricesSingle(config_names)
    configs = mapreduce(x->string(x, ","), string, collect(keys(config_names)))[1:(end-1)]
    best_query = string("select * from prediction_results where configuration_id in ($configs)")
    best_results = RunQuery(best_query)
    best_groups = by(best_results, [:stock], df -> [df])

    for i in 1:size(best_groups,1)
        timesteps = best_groups[i,2][:time_step]
        config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

        actual = (config_groups[1,2][:actual])
        predicted_one = (config_groups[1,2][:predicted])

        stock_name = get(best_groups[i,1])

        t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
        t1 = scatter(;y=predicted_one, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][1])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))

        recreation_plots = [t0, t1]
        filename = string("recreation_", stock_name, "_", collect(keys(config_names))[1])
        savefig(plot(recreation_plots), string("./GraphOutputDir/", filename, export_ext))

    end
end

function RecreateStockPricesMany(config_names)
    configs = mapreduce(x->string(x, ","), string, collect(keys(config_names)))[1:(end-1)]
    best_query = string("select * from prediction_results where configuration_id in ($configs)")
    best_results = RunQuery(best_query)
    best_groups = by(best_results, [:stock], df -> [df])

    for i in 1:size(best_groups,1)
        timesteps = best_groups[i,2][:time_step]
        config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

        #actual = cumsum(config_groups[1,2][:actual])
        #predicted_one = cumsum(config_groups[1,2][:predicted])
        #predicted_two = cumsum(config_groups[2,2][:predicted])

        stock_name = get(best_groups[i,1])
        actual = (config_groups[1,2][:actual])

        predictions = map(i -> Array(config_groups[i,2][:predicted]), 1:size(config_groups,1))
        traces = map(i ->scatter(;y=predictions[i], x = timesteps, name=string(stock_name, "_predicted_",
        config_names[get(config_groups[1][i])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i)), 1:length(predictions))

        t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
        push!(traces, t0)
        #recreation_plots = [t0, traces]
        filename = string("recreation_", stock_name)
        savefig(plot(traces), string("./GraphOutputDir/", filename, export_ext))

    end
end

##MMS Strategy Plots############################################################

function PlotConfusion()

    confusion_results = RunQuery("select * from config_confusion")
    #x0 = Array(confusion_results[:,:all_trades_percentage])
    #trace = histogram(;x=x0, name="Experiment Configurations", marker=Dict(:color => "mediumseagreen"))

    confusion_results[:grouping] = round.(Array(confusion_results[:all_trades_percentage]),2)
    one_groups = by(confusion_results, [:grouping], df -> size(df, 1))
    one_groups[:x1] = one_groups[:x1]./size(confusion_results, 1).*100
    opacity_value = 0.8
    br_max = maximum(one_groups[:x1]) + 0.1
    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Strategy P\&L [n=", size(confusion_results,1),"]"),
                        opacity=opacity_value,
                        marker=Dict(:color => "forestgreen"),
                        xbins=Dict(:size=>0.001))


    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Configuration </br> </b>"))
        , xaxis = Dict(:title => string("<b> Percentage of Correct Trades </b>"))
        , font = Dict(:size => 18))

    data = [trace_one]
    savefig(plot(data, l), string("./GraphOutputDir/Confusion Distribution.html"))
end

function SharpeRatiosPDF()

    sr_data = RunQuery("Select configuration_id, ifnull(sharpe_ratio, 0.0) sharpe_ratio from config_oos_sharpe_ratio_cost")

    sr_data[:grouping] = round.(Array(sr_data[:sharpe_ratio]),2)
    one_groups = by(sr_data, [:grouping], df -> size(df, 1))
    one_groups[:x1] = one_groups[:x1]./size(sr_data, 1).*100
    opacity_value = 0.8
    br_max = maximum(one_groups[:x1]) + 0.1
    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Strategy P\&L [n=", size(sr_data,1),"]"),
                        opacity=opacity_value,
                        marker=Dict(:color => "indianred"),
                        xbins=Dict(:size=>0.001))

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Configurations </br> </b>"))
        , xaxis = Dict(:title => string("<b> Sharpe Ratio </b>"))
        , font = Dict(:size => 18))

    data = [trace_one]
    savefig(plot(data, l), string("./GraphOutputDir/Sharpe Ratios.html"))
end

function AllProfitsPDF(original_prices, cost = false, benchmark_yval = 1500)

    profits = cost ? TotalProfitsCost : GetProfits()

    maxprof = maximum(profits[:profit])
    config_id = Int64.(profits[Bool.(Array(profits[:profit] .== maxprof)),:][:configuration_id][1])

    results = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(results[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end

    results = RunQuery("select * from prediction_results where configuration_id = $config_id")

    num_predictions = get(maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions + 1

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)

    var_name = cost ? parse("cumulative_profit_observed_benchmark_fullcosts") : parse("cumulative_profit_observed_benchmark")
    benchmark_profit = strategyreturns[end,var_name]


    profits[:grouping] = round.(Array(profits[:profit]./10), 0)*10
    one_groups = by(profits, [:grouping], df -> size(df, 1))
    one_groups[:x1] = one_groups[:x1]./size(profits, 1).*100
    opacity_value = 0.8
    br_max = maximum(one_groups[:x1]) + 0.1
    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Strategy P\&L [n=", size(profits,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))


    x0 = Array(profits[:,:profit])

    xvals = [benchmark_profit,benchmark_profit]
    yvals = [0, br_max]
    bm  = scatter(;x=xvals,y=yvals, name="Benchmark", marker = Dict(:color=>"orange"))

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Configurations</br> </b>"))
        , xaxis = Dict(:title => string("<b> P&L Observed </b>"))
        , font = Dict(:size => 18))

    data = [trace_one, bm]
    savefig(plot(data, l), string("./GraphOutputDir/Profits PDF", string(cost), ".html"))
end

function GenericStrategyResultPlot(strategyreturns, columns, filename)
    timesteps = size(strategyreturns, 1)
    traceplots = map(c -> scatter(;y=strategyreturns[c], x = timesteps, name=string(c), mode ="lines"), columns)
    savefig(plot(traceplots), string("./GraphOutputDir/", filename, ".html"))
end

function WriteStrategyGraphs(config_id, strategyreturns)

    cumulative_profits = [:cumulative_profit_observed, :cumulative_profit_observed_fullcosts, :cumulative_profit_observed_benchmark, :cumulative_profit_observed_benchmark_fullcosts]
    cumulative_rates = [:cumulative_observed_rate, :cumulative_expected_rate, :cumulative_benchmark_rate]
    cumulative_rates_fullcosts = [:cumulative_observed_rate_fullcost, :cumulative_expected_rate_fullcost, :cumulative_benchmark_rate_fullcost]

    GenericStrategyResultPlot(strategyreturns, cumulative_profits, string(config_id, "_CumulativeProfits"))
    GenericStrategyResultPlot(strategyreturns, cumulative_rates, string(config_id, "_CumulativeRates"))
    GenericStrategyResultPlot(strategyreturns, cumulative_rates_fullcosts, string(config_id, "_CumulativeRatesFullCost"))
end

function ConfigStrategyOutput(config_id, original_prices)

    results = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(results[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end

    results = RunQuery("select * from prediction_results where configuration_id = $config_id")

    num_predictions = get(maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions + 1

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)
    strategyreturns[end,:cumulative_profit_observed_benchmark]

    WriteStrategyGraphs(config_id, strategyreturns)
    ConfusionMatrix(config_id, stockreturns)
end

function BestStrategyGraphs(config_ids, dataset)

    mini = minimum(config_ids)
    maxi = maximum(config_ids)
    subset = GetProfits()[Array(GetProfits()[:configuration_id] .>= mini) & Array(GetProfits()[:configuration_id] .<= maxi), :]
    maxp = maximum(subset[:profit])
    indices = Array{Bool}(subset[:profit] .== maxp)
    cid = subset[indices,:][:configuration_id][1]
    ConfigStrategyOutput(cid, dataset)
    config_names = Dict(cid=>"best")
    RecreateStockPricesSingle(config_names)
end

function OGD_MSE_vs_PL()

    ogdpl_data = RunQuery("select training_cost, total_pl
    from clusters c
    inner join epoch_records er on er.configuration_id = c.configuration_id and category = 'OGD'
    inner join config_oos_pl pl on pl.configuration_id = c.configuration_id
    where training_cost < 0.02")

    cor(Array(ogdpl_data[:training_cost]),Array(ogdpl_data[:total_pl]))

    ogdpl_data[:,1] = Array(ogdpl_data[:,1])
    ogdpl_data[:,2] = Array(ogdpl_data[:,2])
    ogdpl_data[:rounded_mse] = round.(Array(ogdpl_data[:,1]),3)

    groups = by(ogdpl_data, [:rounded_mse], df -> mean(df[:total_pl]))

    trace1 = scatter(;x = Array(ogdpl_data[:training_cost]), y = Array(ogdpl_data[:total_pl]), mode = "markers", name = "Observations")
    trace2 = scatter(;x = Array(groups[:rounded_mse]), y = Array(groups[:x1]), mode = "line", name = "P&L Mean")

    l = Layout(width = 900, height = 600, margin = Dict(:b => 140, :l => 100)
        , yaxis = Dict(:title => string("<b> P\&L <br> </b>"))
        , xaxis = Dict(:title => string("<b> OGD MSE </b><br><br><br><i> Sample Size: ", size(ogdpl_data,1), "</i>")
                     , :showticklabels => true)
        , font = Dict(:size => 18))


    fig = Plot([trace1, trace2], l)
    savefig(plot(fig), string("./GraphOutputDir/OGD MSE vs PL.html"))
end

##CSCV Plots ####################################################################

function PlotCombinationSizes()

    splitPoints = [2,4,8,10,12,14,16,18,20,22]
    combSizes = Array{Int32,1}()

    for splits in splitPoints
        nrows = 3888
        length = Int64.(floor(nrows/splits))
        ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
        all = Set(ranges)
        m = Int64.(splits/2)
        training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
        set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)
        push!(combSizes, size(set_pairs,1))
    end

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Number of Combinations </br> </b>"))
        , xaxis = Dict(:title => string("<b> Number of Splits </b>"))
        , font = Dict(:size => default_fontsize))

    xvals = splitPoints
    yvals = combSizes

    trace = scatter(;x=xvals,y=yvals, name="Combinations", bargap=0.1, marker = Dict(:line => Dict(:width => 2, :color => "darkred"), :color=>"darkred"))
    data = [trace]
    savefig(plot(data, l), string("./GraphOutputDir/Combinations by S.html"))
end

function PlotPBOBySplits()

    #This is manually populated from the CSCV outputs in CSCV.jl

    xvals = (2, 4, 8, 12, 16)
    yvals = (50,
            16.666666666666666,
            07.142857142857142,
            02.4891774891774902,
            01.7171717171717137)

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> PBO % </br> </b>"))
        , xaxis = Dict(:title => string("<b> Number of Splits </b>"))
        , font = Dict(:size => default_fontsize))


    trace = scatter(;x=xvals,y=yvals, name="Combinations",
                    bargap=0.1,
                    marker = Dict(:line => Dict(:width => 2, :color => "darkred"), :color=>"darkred"))
    data = [trace]
    savefig(plot(data, l), string("./GraphOutputDir/PBO by S.html"))
end

##DSR Plots#####################################################################

function ClusterOGDMSEPlot()

    clusterOneResults = RunQuery("select training_cost
                                    from epoch_records p
                                    inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 1
                                    inner join config_oos_sharpe_ratio_cost s on s.configuration_id = c.configuration_id and sharpe_ratio is not null
                                    where category = 'OGD'
                                    --and training_cost is not null
                                    and training_cost < 0.05")
    clusterTwoResults = RunQuery("select training_cost
                                from epoch_records p
                                inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 0
                                inner join config_oos_sharpe_ratio_cost s on s.configuration_id = c.configuration_id and sharpe_ratio is not null
                                where category = 'OGD'
                                --and training_cost is not null
                                and training_cost < 0.05")

    clusterOneResults[:grouping] = round.(Array(clusterOneResults[:training_cost]), 3)
    clusterTwoResults[:grouping] = round.(Array(clusterTwoResults[:training_cost]), 3)

    one_groups = by(clusterOneResults, [:grouping], df -> size(df, 1))
    two_groups = by(clusterTwoResults, [:grouping], df -> size(df, 1))

    one_groups[:x1] = one_groups[:x1]./size(clusterOneResults, 1).*100
    two_groups[:x1] = two_groups[:x1]./size(clusterTwoResults, 1).*100

    opacity_value = 0.8

    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Cluster One [n=14397]"), opacity=opacity_value, xbins=Dict(:size=>0.0001))
    trace_two = bar(;y=two_groups[:x1], x=two_groups[:grouping], name=string("Cluster Two [n=7253]"), opacity=opacity_value, xbins=Dict(:size=>0.0001))

    data = [trace_one, trace_two]#, bm]

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Trials in Cluster </br> </b>"))
        , xaxis = Dict(:title => string("<b> OGD MSE </b>"))
        , barmode="overlay"
        , font = Dict(:size => default_fontsize))

    fig = Plot(data, l)
    savefig(plot(fig), string("./GraphOutputDir/Cluster MSE Distributions.html"))
end

function ClusterDistributionPercAll(cluster_ids)

    configuration_num = get(RunQuery("select count(distinct(configuration_id)) from config_oos_sharpe_ratio_cost")[1,1])
    highestSR =get(RunQuery("select max(sharpe_ratio) from config_oos_sharpe_ratio_cost")[1,1])
    opacity_value = 0.8

    c_id = cluster_ids[1]
    clusterOneResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = $c_id where sharpe_ratio is not null")
    clusterOneResults[:grouping] = round.(Array(clusterOneResults[:sharpe_ratio]), 2)
    one_groups = by(clusterOneResults, [:grouping], df -> size(df, 1))
    one_groups[:x1] = one_groups[:x1]./configuration_num.*100#  size(clusterOneResults, 1).*100
    maximums = [maximum(one_groups[:x1])]
    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Cluster 1 [n=", size(clusterOneResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
    data = [trace_one]

    for c in 2:size(cluster_ids,1)

        c_id = cluster_ids[c]

        clusterResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = $c_id where sharpe_ratio is not null")
        clusterResults[:grouping] = round.(Array(clusterResults[:sharpe_ratio]), 2)
        groups = by(clusterResults, [:grouping], df -> size(df, 1))
        groups[:x1] = groups[:x1]./configuration_num.*100#size(clusterTwoResults, 1).*100
        push!(maximums, maximum(groups[:x1]))
        trace = bar(;y=groups[:x1], x=groups[:grouping], name=string("Cluster $c [n=", size(clusterResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
        push!(data, trace)
    end

    br_max = maximum(maximums) + 0.1
    bm  = scatter(;x=[highestSR, highestSR],y=[0, br_max], name="Highest SR", marker = Dict(:color=>"green"))

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Trials </br> </b>"))
        , xaxis = Dict(:title => string("<b> Sharpe Ratio </b>"))
        , barmode="overlay"
        , font = Dict(:size => default_fontsize))

    fig = Plot(data, l)
    savefig(plot(fig), string("./GraphOutputDir/Cluster Distributions Perc of All.html"))
end

function ClusterDistributionPercCluster(cluster_ids)

    highestSR =get(RunQuery("select max(sharpe_ratio) from config_oos_sharpe_ratio_cost")[1,1])
    opacity_value = 0.8

    c_id = cluster_ids[1]
    clusterOneResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = $c_id where sharpe_ratio is not null")
    clusterOneResults[:grouping] = round.(Array(clusterOneResults[:sharpe_ratio]), 2)
    one_groups = by(clusterOneResults, [:grouping], df -> size(df, 1))
    cluster_num = size(clusterOneResults, 1)
    one_groups[:x1] = one_groups[:x1]./cluster_num.*100#  size(clusterOneResults, 1).*100
    maximums = [maximum(one_groups[:x1])]
    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Cluster 1 [n=", size(clusterOneResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
    data = [trace_one]

    for c in 2:size(cluster_ids,1)

        c_id = cluster_ids[c]

        clusterResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = $c_id where sharpe_ratio is not null")
        clusterResults[:grouping] = round.(Array(clusterResults[:sharpe_ratio]), 2)
        groups = by(clusterResults, [:grouping], df -> size(df, 1))
        cluster_num = size(clusterResults, 1)
        groups[:x1] = groups[:x1]./cluster_num.*100#size(clusterTwoResults, 1).*100
        push!(maximums, maximum(groups[:x1]))
        trace = bar(;y=groups[:x1], x=groups[:grouping], name=string("Cluster $c [n=", size(clusterResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
        push!(data, trace)
    end

    br_max = maximum(maximums) + 0.1
    bm  = scatter(;x=[highestSR, highestSR],y=[0, br_max], name="Highest SR", marker = Dict(:color=>"green"))

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Trials in Cluster </br> </b>"))
        , xaxis = Dict(:title => string("<b> Sharpe Ratio </b>"))
        , barmode="overlay"
        , font = Dict(:size => default_fontsize))

    fig = Plot(data, l)
    savefig(plot(fig), string("./GraphOutputDir/Cluster Distributions Perc of Cluster.html"))
end

function PrintClusterAnalysis(cluster_ids)

    for c in cluster_ids
        clusterOneResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p
                                    inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = $c
                                    where sharpe_ratio is not null")
        println(string("Cluster $c Skewness: ", string(skewness(Array(clusterOneResults[:sharpe_ratio])))))
        println(string("Cluster $c Kurtosis: ", string(kurtosis(Array(clusterOneResults[:sharpe_ratio])))))
        println(string("Cluster $c Mean: ", string(mean(Array(clusterOneResults[:sharpe_ratio])))))
        println(string("Cluster $c Variance: ", string(var(Array(clusterOneResults[:sharpe_ratio])))))
        println(string("Cluster $c Std Dev: ", string(std(Array(clusterOneResults[:sharpe_ratio])))))
        println(string("Cluster $c Samples: ", string(size(Array(clusterOneResults[:sharpe_ratio])))))
    end
end

################################################################################

function ReadOOSProfits()

    plresults = RunQuery("select * from config_oos_pl")

    pl_df = DataFrame()
    pl_df[:configuration_id] = Array(plresults[:configuration_id])
    pl_df[:profit] = Array(plresults[:total_pl])

    return pl_df
end

function ReadISProfits()

    plresults = RunQuery("select * from config_is_pl")

    pl_df = DataFrame()
    pl_df[:configuration_id] = Array(plresults[:configuration_id])
    pl_df[:profit] = Array(plresults[:total_pl])

    return pl_df
end

function ReadProfitsCost()
    plresults = RunQuery("select * from config_oos_pl_cost")

    pl_df = DataFrame()
    pl_df[:configuration_id] = Array(plresults[:configuration_id])
    pl_df[:profit] = Array(plresults[:total_pl])

    return pl_df
end

function ReadTestProfits()

    #This data is from a previous database which was generated with different code. The results are primarily for synthetic data (none of the Actual10 data results are from this).

    ln = BSON.load(string("./ProfitVals.bson"))
    return DataFrame(configuration_id = Array{Int64,1}(ln[:profits][:,1]), profit = ln[:profits][:,2])
end

function GetProfits(testDatabase = false, in_sample = false)
    if testDatabase == true
        return TotalProfits_Test
    elseif in_sample == true
        return TotalProfitsIS
    else
        return TotalProfitsOOS
    end
end

TotalProfits_Test = ReadTestProfits()
TotalProfitsOOS = ReadOOSProfits() #OOS no Cost
TotalProfitsIS = ReadISProfits() #IS no Cost
TotalProfitsCost = ReadProfitsCost() #OOS Cost

end
