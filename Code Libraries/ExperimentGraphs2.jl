module ExperimentGraphs2

#workspace()
#push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

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

using PlotlyJS


export FFN_LR_Sched_BxProfit, OGD_EncodingSizes_Profits_Bx, OGD_L1Reg_BxProfit, OGD_NetworkSize_Profits_Bx, OGD_Init_Profits_Bx, SAE_LayerSizes_MinMSE, SAE_EncodingSizes_MinMSE, SAE_Deltas_MinTest_MxMSE, SAE_Init_MinTest_MxMSE, SAE_LREpochs_MinTest_BxMSE, RecreateStockPricesSingle, BestStrategyGraphs, OGD_DataDeltas_Profits_Bx, SAEProfitBoxPlot, OGD_DataVariances_Profits_Bx, OGD_NetworkVariances_Profits_Bx,SAE_Lambda1_MinTest_BxMSE, Denoising_BxMSE, OGD_ValidationSet_Profits_bx, SAE_MaxLR_MinTest_BxMSE, FFN_LR_BxProfit, OGD_LR_AvgTrain_BxMSE, OGD_LR_BxProfit, OGD_Activations_Profits_Bx, OGD_SAE_Selection_Profits_bx, OGD_NetworkSizeOutputActivation_Profits_Bx, SAE_ActivationsNetworkSizes_MinMSE, SAE_ActivationsEncodingSizes_MinMSE

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end

################################################################################
##Profit Records

function UpdateTotalProfits(config_ids, over_ride, dataset)

    #Original Setup
    #TotalProfits = DataFrame()
    #TotalProfits[:configuration_id] = []
    #TotalProfits[:profit] = []

    TotalProfits = BSON.load("ProfitVals.bson")[:profits]

    current_configs = TotalProfits[:,1]
    if over_ride
        needed_configs = config_ids
    else
        needed_configs = collect(setdiff(Set(config_ids), Set(current_configs)))
    end

    println(length(needed_configs))

    for c in needed_configs
        println(c)
        tic()
        profits = GenerateTotalProfit(c, dataset)

        index = findin(TotalProfits[:,1], c)
        if length(index) > 0
            TotalProfits[index[1],2] = profits
        else
            TotalProfits = cat(1, TotalProfits, [c profits])
        end
        println(string(toc()))
    end

    profit_array = Array(TotalProfits)
    file_name = string("ProfitVals.bson")
    values = Dict(:profits => profit_array)
    bson(file_name, values)
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
    #return ascii(split(split(set_name, "_")[1])[end])
    return ascii(filter(l -> contains(l, "x"), split(set_name, " "))[1])
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
    l = Layout(width = 1000, height = 600, margin = Dict(:b => 150))
    plot(data,l)
    savefig(plot(data,l), string("/users/joeldacosta/desktop/", filename, ".html"))
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

function OGD_ScalingOutputActivation_Profits_Bx(config_ids)

    minid = minimum(config_ids)
    maxid = maximum(config_ids)

    query = "select cr.configuration_id,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) || '-' || output_activation || '-' || scaling_function) scaling_methodology
            from configuration_run cr
            inner join dataset_config dc on cr.configuration_id = dc.configuration_id
            inner join network_parameters np on np.configuration_id = cr.configuration_id
            where cr.configuration_id between $minid and $maxid
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :scaling_methodology, " ", "Scaling Methodolgy Profits", String, NullTransform)
end

#config transform done

function FFN_LR_Sched_BxProfit(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_query = "select tp.configuration_id, (cast(learning_rate as text) || '-' ||  cast(min_learning_rate as text)) learning_rates
                from training_parameters tp
                inner join configuration_run cr on cr.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids) and category = 'FFN'
                order by tp.configuration_id desc"

    ProfitBoxplot(lr_query, :learning_rates, "FFN Learning Rates Schedules", "FFN LR-Schedule Profits", String, NullTransform)
end

function SAEProfitBoxPlot(config_ids)

    ids = TransformConfigIDs(config_ids)

    sae_query = string("select configuration_id, sae_config_id from configuration_run
                            where configuration_id in ($ids)")
    sae_results = RunQuery(sae_query)

    sae_results[:,1] = Array{Int64,1}(sae_results[:,1])
    sae_results[:,2] = Array{Int64,1}(sae_results[:,2])
    sae_returns = join(TotalProfits, sae_results, on = :configuration_id)
    groups = by(sae_returns, [:sae_config_id], df -> [df])

    sae_boxplot(groups, "SAE Profit Boxplots", :profit)
end

function OGD_L1Reg_BxProfit(config_ids)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    l1_lambda regval
                from training_parameters tp
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""

    ProfitBoxplot(lr_query, :regval, "L1 Lambda ", "OGD L1 Reg Effects on Profits", Float64, NullTransform)
end

function OGD_ValidationSetMaxEpochs_Profits_bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                (tp.max_epochs ||'-'||
                case when experiment_set_name like '%CVSplit_1.0%' then 0
                     when experiment_set_name like '%CVSplit_0.9%' then 10
                     when experiment_set_name like '%CVSplit_0.8%' then 20
                     when experiment_set_name like '%CVSplit_0.7%' then 30
                     when experiment_set_name like '%CVSplit_0.6%' then 40
                end) Epochs_Validation_Percentage_Size
            from configuration_run cr
            inner join training_parameters tp on cr.configuration_id = tp.configuration_id
            where cr.configuration_id in ($ids)
            and tp.category = 'FFN'
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :Epochs_Validation_Percentage_Size, "Epochs-ValidationSet ", "Max SGD Epochs and Validation Set Effects on Profits", String, NullTransform)
end

function OGD_ValidationSet_Profits_bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                case when experiment_set_name like '%CVSplit_1.0%' then 0
                     when experiment_set_name like '%CVSplit_0.9%' then 10
                     when experiment_set_name like '%CVSplit_0.8%' then 20
                     when experiment_set_name like '%CVSplit_0.7%' then 30
                     when experiment_set_name like '%CVSplit_0.6%' then 40
                     when experiment_set_name like '%CVSplit_0.5%' then 50
                     when experiment_set_name like '%CVSplit_0.4%' then 60
                     when experiment_set_name like '%CVSplit_0.3%' then 70
                     when experiment_set_name like '%CVSplit_0.2%' then 80
                     when experiment_set_name like '%CVSplit_0.1%' then 90
                end Validation_Percentage_Size
            from configuration_run cr
            where configuration_id in ($ids)
            order by cr.configuration_id desc"


    r = RunQuery(query)

    ProfitBoxplot(query, :Validation_Percentage_Size, "Validation Set Percentage ", "Validation Set Effects on Profits", Float64, NullTransform)
end

function OGD_LR_BxProfit(config_ids)

    ids = TransformConfigIDs(config_ids)
    lr_query = "select tp.configuration_id,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) ||'-'|| tp.learning_rate) activation_learningrate
                from training_parameters tp
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN-OGD\""
    ProfitBoxplot(lr_query, :activation_learningrate, "OGD Learning Rate", "OGD LR Profits", String, NullTransform)
end

function FFN_LR_BxProfit(config_ids)
    ids = TransformConfigIDs(config_ids)
    lr_query = "select
                    tp.configuration_id,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) ||'-'|| tp.learning_rate) activation_learningrate
                from training_parameters tp
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN\""
    ProfitBoxplot(lr_query, :activation_learningrate, "FFN Learning Rate", "FFN LR Profits", String, NullTransform)
end

function OGD_Activations_Profits_Bx(config_ids)

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

    ProfitBoxplot(query, :networkconfig, " ", "Network Activations Profits", String, NullTransform)
end

function OGD_NetworkSizeOutputActivation_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
            (
            substr(layer_activations, 1,  instr(layer_activations, ',')-11)
            --|| '-' || output_activation
            || '-' ||
            substr(layer_sizes, instr(layer_sizes, ','), length(layer_sizes))
            ) networkconfig
            from configuration_run cr
            inner join  network_parameters np on cr.configuration_id = np.configuration_id
            where cr.configuration_id in($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :networkconfig, " ", "Network Size Profits", String, NullTransform)
end

function OGD_NetworkSize_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
            'Layers' || '-' || (substr(layer_sizes, instr(layer_sizes, ',')+1, length(layer_sizes))) network_structure
        from network_parameters
        where configuration_id in ($ids)
        order by network_structure"

    ProfitBoxplot(query, :network_structure, " ", "Network Size Profits", String, NullTransform)
end

function OGD_EncodingSizes_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
                    substr(layer_sizes, 0, instr(layer_sizes, ',')) encoding
            from network_parameters
            where configuration_id in ($ids)
            order by encoding"

    ProfitBoxplot(query, :encoding, "Encoding ", "Encoding Size Profits", String, NullTransform)

end

function OGD_DataVariances_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
            variation_values
            from dataset_config dc
            where configuration_id in($ids)"

    ProfitBoxplot(query, :variation_values, " ", "Data Variances Profits", String, NullTransform)
end

function OGD_DataDeltas_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
            deltas
            from dataset_config dc
            where configuration_id in($ids)"

    ProfitBoxplot(query, :deltas, " ", "Data Deltas Profits", String, NullTransform)
end

function OGD_Init_Profits_Bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select configuration_id,
            initialization
            from network_parameters
            where configuration_id in($ids)"

    ProfitBoxplot(query, :initialization, " ", "Init Profits", String, NullTransform)
end

function OGD_SAE_Selection_Profits_bx(config_ids)

    ids = TransformConfigIDs(config_ids)

    query = "select cr.configuration_id,
                case when experiment_set_name like '%MAPE%' then 'MAPE'
            else 'MSE'
            end SAE_Selection
            from configuration_run cr
            where configuration_id in ($ids)
            order by cr.configuration_id desc"

    ProfitBoxplot(query, :SAE_Selection, "Method ", "SAE Selection Method Effects on Profits", String, NullTransform)
end

################################################################################
#MSE BoxPlots

##Config Transforms done

function SAE_ActivationsNetworkSizes_MinMSE(config_ids, encoding_size = nothing)

    function general_boxplot2(layer_groups, prefix, fn, variable_name)

        y_vals = layer_groups[1,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
        data = [trace]

        for i in 2:size(layer_groups,1)
            y_vals = layer_groups[i,2][:,variable_name]
            trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
            push!(data, trace)
        end
        plot(data)
        l = Layout(width = 1500, height = 500, margin = Dict(:b => 120))
        savefig(plot(data, l), string("/users/joeldacosta/desktop/", fn, ".html"))
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

    results = RunQuery(query)

    #By encoding; output activation; primary activation; encoding layer & network size
    results[:network] =  map(t -> replace(t[(findin(t, ",")[1]+1):(findin(t, ",")[end])], ",", "x"), Array(results[:layer_sizes]))
    results[:primary_activation] =  ascii.(Array(map(e -> split(split(e, " ")[end], "_")[1], results[:experiment_set_name])))
    results[:encoding_size] = Array(map(n -> split(n, "x")[end], results[:network]))

    results[:config_group] = string.(
                                    results[:network], "-"
                                    #results[:encoding_size]
                                    , "-"
                                    ,replace.(results[:primary_activation], "Activation", "")
                                    #, "-",
                                    #replace.(Array(results[:encoding_activation]), "Activation", ""), "-",
                                    #replace.(Array(results[:output_activation]), "Activation", ""), "-"
                                    #,Array(results[:scaling_function])
                                    #, "-"
                                    #,"-", Array(results[:lr])


    )

    #filtered_indices = Array(results[:primary_activation] .!= "SigmoidActivation") & Array(results[:encoding_size] .== "5")

    filename = "SAE Activations And Network Sizes Min MSE"

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

function SAE_ActivationsEncodingSizes_MinMSE(config_ids, encoding_size = nothing)

    function general_boxplot2(layer_groups, prefix, fn, variable_name)

        y_vals = layer_groups[1,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
        data = [trace]

        for i in 2:size(layer_groups,1)
            y_vals = layer_groups[i,2][:,variable_name]
            trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
            push!(data, trace)
        end
        plot(data)
        l = Layout(width = 1500, height = 500, margin = Dict(:b => 120))
        savefig(plot(data, l), string("/users/joeldacosta/desktop/", fn, ".html"))
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

    results = RunQuery(query)

    #By encoding; output activation; primary activation; encoding layer & network size
    results[:network] =  map(t -> replace(t[(findin(t, ",")[1]+1):end], ",", "x"), Array(results[:layer_sizes]))
    results[:primary_activation] =  ascii.(Array(map(e -> split(split(e, " ")[end], "_")[1], results[:experiment_set_name])))
    results[:encoding_size] = Array(map(n -> split(n, "x")[end], results[:network]))

    results[:config_group] = string.(
                                    #results[:network], "-",
                                    results[:encoding_size]
                                    , "-"
                                    ,replace.(results[:primary_activation], "Activation", "")
                                    #, "-",
                                    #replace.(Array(results[:encoding_activation]), "Activation", ""), "-",
                                    #replace.(Array(results[:output_activation]), "Activation", ""), "-"
                                    #,Array(results[:scaling_function])
                                    #, "-"
                                    #,"-", Array(results[:lr])


    )

    #filtered_indices = Array(results[:primary_activation] .!= "SigmoidActivation") & Array(results[:encoding_size] .== "5")

    filename = "SAE Activations And Encoding Sizes Min MSE"

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

function SAE_EncodingSizes_MinMSE(config_ids, encoding_size = nothing)

    function general_boxplot2(layer_groups, prefix, fn, variable_name)

        y_vals = layer_groups[1,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
        data = [trace]

        for i in 2:size(layer_groups,1)
            y_vals = layer_groups[i,2][:,variable_name]
            trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
            push!(data, trace)
        end
        plot(data)
        l = Layout(width = 1500, height = 500, margin = Dict(:b => 120))
        savefig(plot(data, l), string("/users/joeldacosta/desktop/", fn, ".html"))
    end

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
        order by cost desc"

    results = RunQuery(query)
    results[:encoding_size] = Array(map(n -> split(n, ",")[end], results[:layer_sizes]))
    filename = "SAE Activations And Encoding Sizes Min MSE"
    groups = by(results, [:encoding_size], df -> [df])
    general_boxplot2(groups, " ", filename, :cost)
end

function SAE_LayerSizes_MinMSE(config_ids, encoding_size = nothing)

    function general_boxplot2(layer_groups, prefix, fn, variable_name)

        y_vals = layer_groups[1,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
        data = [trace]

        for i in 2:size(layer_groups,1)
            y_vals = layer_groups[i,2][:,variable_name]
            trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
            push!(data, trace)
        end
        plot(data)
        l = Layout(width = 1500, height = 500, margin = Dict(:b => 120))
        savefig(plot(data, l), string("/users/joeldacosta/desktop/", fn, ".html"))
    end

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

    results = RunQuery(query)

    results[:layers] = Array(map(t -> string("Layers - ", t[1:(findin(t, ",")[end]-1)]), Array(results[:layer_sizes])))

    filename = "SAE Layer Sizes Min MSE"
    groups = by(results, [:layers], df -> [df])
    general_boxplot2(groups, " ", filename, :cost)
end

function SAE_MaxLR_MinTest_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select
                    tp.configuration_id,
                    min(testing_cost) cost,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) ||'-'|| tp.learning_rate) activation_learningrate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.learning_rate
                having min(testing_cost) is not null"

    r = RunQuery(lr_msequery)

    MSEBoxplot(lr_msequery, :activation_learningrate, "SAE Max LR", "SAE Max Learning Rate Min Test MSE", String, NullTransform)
end

function SAE_LREpochs_MinTest_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select
                    tp.configuration_id,
                    min(testing_cost) cost,
                    epoch_cycle_max
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.epoch_cycle_max
                having min(testing_cost) is not null"

    r = RunQuery(lr_msequery)

    MSEBoxplot(lr_msequery, :epoch_cycle_max, "SAE LR Epoch Cycle", "SAE Learning Rate Epoch Cycle Min Test MSE", Int64, NullTransform)
end

function OGD_LR_AvgTrain_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    ogd_mse_query = "select
                    tp.configuration_id,
                    avg(training_cost) cost,
                    (substr(layer_activations, 1,  instr(layer_activations, ',')-11) ||'-'|| tp.learning_rate) activation_learningrate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                inner join network_parameters np on np.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category = \"FFN-OGD\"
                    and er.category = \"OGD\"
                group by tp.configuration_id, tp.learning_rate
                having training_cost not null"

    MSEBoxplot(ogd_mse_query, :activation_learningrate, "OGD LR", "OGD Learning Rate Avg Train MSE", String, NullTransform)
end

function SAE_Lambda1_MinTest_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select tp.configuration_id, min(testing_cost) cost, tp.l1_lambda
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.l1_lambda
                having cost not null"

    MSEBoxplot(lr_msequery, :l1_lambda, "Lambda1 ", "SAE L1 Reg Min Test MSE", Float64, NullTransform)
end

function SAE_MinLR_MinTest_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select tp.configuration_id,
                            min(testing_cost) cost,
                            tp.min_learning_rate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.min_learning_rate"

    MSEBoxplot(lr_msequery, :min_learning_rate, "SAE Min LR", "SAE Min Learning Rate Min Test MSE", Float64, NullTransform)
end

function SAE_Init_MinTest_MxMSE(config_ids)
    ids = TransformConfigIDs(config_ids)

    init_query = string("select er.configuration_id, min(testing_cost) cost, initialization init
                        from epoch_records er
                        inner join network_parameters np on np.configuration_id = er.configuration_id
                        where np.category = \"SAE\"
                            and er.configuration_id in ($ids)
                        group by er.configuration_id, init
                        having cost not null")

    MSEBoxplot(init_query, :init, "Init", "SAE Inits Min Test MSE", String, NullTransform)
end

function SAE_Deltas_MinTest_MxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    delta_query = string("select er.configuration_id, min(testing_cost) cost, deltas
                        from epoch_records er
                        inner join network_parameters np on np.configuration_id = er.configuration_id
                        inner join dataset_config dc on dc.configuration_id = er.configuration_id
                        where np.category = \"SAE\"
                            and er.configuration_id in ($ids)
                        group by er.configuration_id, deltas
                        having cost not null")

    MSEBoxplot(delta_query, :deltas, "Delta", "SAE Delta MSE", String, NullTransform)
end


#####

function SAE_Scaling_MinTest_BxMSE(config_ids)

    minid = minimum(config_ids)
    maxid = maximum(config_ids)

    query = "select  tp.configuration_id,
                    min(testing_cost) cost,
                    (case when experiment_set_name like '%Limited%' then \"LimitedStandardize\" else \"Standardize\" end || '-' ||
                    case when cr.experiment_set_name like '%LinearActivation%' then 'LinearActivation' else 'ReluActivation' end) scaling

            from training_parameters tp
            inner join epoch_records er on er.configuration_id = tp.configuration_id
            inner join configuration_run cr on cr.configuration_id = tp.configuration_id
            where tp.configuration_id between $minid and $maxid
                and tp.category = \"SAE\"
            group by tp.configuration_id, scaling"

    MSEBoxplot(query, :scaling, "Scaling Ltd ", "SAE Scaling Limitation Min Test MSE", String, NullTransform)
end

function OGD_ScalingOutput_BxMSE(config_ids)

    minid = minimum(config_ids)
    maxid = maximum(config_ids)

    query = "select er.configuration_id, training_cost cost,
        (sf.scaling_method || \"-\" ||
        case when cr.experiment_set_name like '%LinearActivation%' then 'LinearActivation'
        else 'ReluActivation'
        end) scaling_method
    from epoch_records er
    inner join configuration_run cr on cr.configuration_id = er.configuration_id
    inner join scaling_functions sf on sf.configuration_id = er.configuration_id
    where er.configuration_id between $minid and $maxid
    and category = \"OGD\"
    and training_cost is not null"

    p = MSEBoxplot(query, :scaling_method, "Scaling Method ", "OGD Scaling Limitation Min MSE", String, NullTransform)
end

function SAE_ScalingOutput_BxMSE(config_ids)

    minid = minimum(config_ids)
    maxid = maximum(config_ids)

    query = "select er.configuration_id, min(training_cost) cost,
        min((dc.scaling_function || '-' ||
        case when cr.experiment_set_name like '%LinearActivation%' then 'LinearActivation'
        when cr.experiment_set_name like '%SigmoidActivation%' then 'SigmoidActivation'
        else 'ReluActivation'
        end)) scaling_method
    from epoch_records er
    inner join configuration_run cr on cr.configuration_id = er.configuration_id
    inner join dataset_config dc on dc.configuration_id = er.configuration_id
    where er.configuration_id between $minid and $maxid
    and er.category = \"SAE-SGD-Init\"
    and training_cost is not null
    group by er.configuration_id"

    results = RunQuery(query)

    p = MSEBoxplot(query, :scaling_method, "Scaling Method ", "SAE ScalingOutput Min MSE", String, NullTransform)
end

function SAE_ScalingLimited_MinTest_BxMSE(min_config)

    query = "select  tp.configuration_id,
                    min(testing_cost) cost,
                    case when experiment_set_name like '%Limited%' then 1 else 0 end limited
            from training_parameters tp
            inner join epoch_records er on er.configuration_id = tp.configuration_id
            inner join configuration_run cr on cr.configuration_id = tp.configuration_id
            where tp.configuration_id >= $min_config
                and tp.category = \"SAE\"
            group by tp.configuration_id, limited"

    MSEBoxplot(query, :limited, "Scaling Ltd ", "SAE Scaling Limitation Min Test MSE", Float64, NullTransform)
end

function SAE_Pretraining_MinTest_BxMSE(config_ids)

    mc = minimum(config_ids)
    xc = maximum(config_ids)

    #ifnull(max(er2.epoch_number), 0) pre_training_epochs

    query = "select er.configuration_id, min(er.testing_cost) cost,
                (cast(ifnull(max(er2.epoch_number), 0) as string) || '-' || cast(tp.learning_rate as string)) pre_training_epochs

            from epoch_records er
            left join epoch_records er2 on er.configuration_id = er2.configuration_id and er2.category = 'RBM-CD'
            inner join training_parameters tp on tp.configuration_id = er.configuration_id and tp.category = 'RBM-CD'
            where er.category like \"SAE-SGD%\"
                and er.configuration_id between $mc and $xc
            group by er.configuration_id
            having cost not null"

    MSEBoxplot(query, :pre_training_epochs, "Pretraining Epochs ", "SAE Pre-training Learning Rates epochs Min Test MSE", String, NullTransform)
end

function SAE_MinLR_MinTest_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    lr_msequery = "select tp.configuration_id, min(testing_cost) cost, tp.min_learning_rate
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id >= $min_config
                    and tp.category like \"SAE%\"
                    and er.category like \"SAE%\"
                group by tp.configuration_id, tp.min_learning_rate"

    MSEBoxplot(lr_msequery, :min_learning_rate, "SAE Min LR", "SAE Min Learning Rate Min Test MSE", Float64, NullTransform)
end

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

function Layer_MinTest_MxMSE(min_config, category)
    layer_msequery = string("select er.configuration_id, min(testing_cost) cost, experiment_set_name
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category like \"$category%\"
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

    layer_msequery = string("select er.configuration_id, tp.learning_rate, min(testing_cost) min_test_cost, experiment_set_name
                            from epoch_records er
                            inner join configuration_run cr on cr.configuration_id = er.configuration_id
                            inner join training_parameters tp on tp.configuration_id = er.configuration_id
                            where er.configuration_id >= $min_config
                            and er.category = \"FFN-SGD\"
                            and tp. category = \"FFN\"
                            group by er.configuration_id, sae_config_id, tp.learning_rate
                            ")
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

function StockPricePlot(dataseed)
    var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
    ds = GenerateDataset(dataseed, 5000, var_pairs)

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
        savefig(plot(recreation_plots), string("/users/joeldacosta/desktop/", filename, ".html"))

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
        savefig(plot(recreation_plots), string("/users/joeldacosta/desktop/", filename, ".html"))

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
        savefig(plot(traces), string("/users/joeldacosta/desktop/", filename, ".html"))

    end
end

################################################################################
##MMS Strategy Plots

function AllProfitsPDF(min_config)
    indexes = Array{Bool}(TotalProfits[:configuration_id] .> min_config)

    x0 = Array(TotalProfits[indexes,:profit])
    trace = histogram(;x=x0, name="Configuration Profits")
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/Profits PDF.html"))
end

function GenericStrategyResultPlot(strategyreturns, columns, filename)
    timesteps = size(strategyreturns, 1)
    traceplots = map(c -> scatter(;y=strategyreturns[c], x = timesteps, name=string(c), mode ="lines"), columns)
    savefig(plot(traceplots), string("/users/joeldacosta/desktop/", filename, ".html"))
end

function WriteStrategyGraphs(config_id, strategyreturns)
    #daily_rates = [:daily_rates_observed, :daily_rates_observed_fullcosts]
    cumulative_profits = [:cumulative_profit_observed, :cumulative_profit_observed_fullcosts, :cumulative_profit_observed_benchmark, :cumulative_profit_observed_benchmark_fullcosts]
    cumulative_rates = [:cumulative_observed_rate, :cumulative_expected_rate, :cumulative_benchmark_rate]
    cumulative_rates_fullcosts = [:cumulative_observed_rate_fullcost, :cumulative_expected_rate_fullcost, :cumulative_benchmark_rate_fullcost]

    #GenericStrategyResultPlot(strategyreturns, daily_rates, string(config_id, "_DailyRates"))
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

    #actuals = original_prices[start_t:finish_t, :]
    #all(round.(Array(results[:actual]),4) .== round.(Array(actuals),4))

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)

    WriteStrategyGraphs(config_id, strategyreturns)
    ConfusionMatrix(config_id, stockreturns)

end

function ConfusionMatrix(config_id, stockreturns)
    trades = mapreduce(i -> stockreturns[i,2][:,[:trade, :trade_benchmark]], vcat, 1:size(stockreturns,1))

    actual =   trades[:,2]
    model =    trades[:,1]

    df = DataFrame(confusmat(2, actual .+ 1, model .+ 1))
    names!(df, [:model_no_trade, :model_trade])

    writetable(string("/users/joeldacosta/desktop/", config_id, "_confusion.csv"), df)
end

###############################################################################
##New Plots

function Denoising_BxMSE(config_ids)

    ids = TransformConfigIDs(config_ids)

    dn_mse_query = "select tp.configuration_id, avg(testing_cost) cost, denoising_variance
                from training_parameters tp
                inner join epoch_records er on er.configuration_id = tp.configuration_id
                where tp.configuration_id in ($ids)
                group by tp.configuration_id, denoising_variance
                having training_cost not null"

    MSEBoxplot(dn_mse_query, :denoising_variance, "DN Variance", "Denoising Variance Min MSE", Float64, NullTransform)
end

function BestStrategyGraphs(config_ids, dataset)

    mini = minimum(config_ids)
    maxi = maximum(config_ids)
    subset = TotalProfits[Array(TotalProfits[:configuration_id] .>= mini) & Array(TotalProfits[:configuration_id] .<= maxi), :]
    maxp = maximum(subset[:profit])
    indices = Array{Bool}(subset[:profit] .== maxp)
    cid = subset[indices,:][:configuration_id][1]
    ConfigStrategyOutput(cid, dataset)
    config_names = Dict(cid=>"best")
    RecreateStockPricesSingle(config_names)
end

###############################################################################
##General Plots

#jsedata = ReadJSETop40Data()
#agldataset = jsedata[:, [:AGL]]
#config_ids = 30059:30418
#UpdateTotalProfits(config_ids, false, agldataset)

#TotalProfits = ReadProfits()


#AllProfitsPDF(3704)

end
