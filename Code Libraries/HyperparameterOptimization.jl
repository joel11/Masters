push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

module HyperparameterOptimization

export HyperparameterRangeSearch, GraphHyperparameterResults, ChangeMaxLearningRate, ChangeL1Reg, ChangeL2Reg, ChangeMinibatchSize, GenerateGridBasedParameterSets, GetDataConfig, GetSAENetwork, GetFFNNetwork, GetSAETraining, GetFFNTraining, GetOGDTraining, GetOGDHOTraining, ChangeLayers, ChangeInit, GetRBMTraining, ChangeMaxEpochs

using NetworkTrainer, TrainingStructures, StoppingFunctions, CostFunctions
#using Plots
#plotlyjs()

function ChangeInit(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_Init_" , split(string(val), ".")[end])
    get_function(parameters).initialization = val
    return parameters
end

function ChangeMaxLearningRate(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MaxLearningRate_" , string(val))
    get_function(parameters).max_learning_rate = val
    return parameters
end

function ChangeL2Reg(get_function,parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_L2Reg_" , string(val))
    get_function(parameters).l2_lambda = val
    return parameters
end

function ChangeL1Reg(get_function,parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_L1Reg_" , string(val))
    get_function(parameters).l1_lambda = val
    return parameters
end

function ChangeMinibatchSize(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MinibatchSize_" , string(val))
    get_function(parameters).minibatch_size = val
    return parameters
end

function ChangeLayers(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_LayerSizes_" , string(val[1]))
    get_function(parameters).layer_sizes = val[2]
    get_function(parameters).layer_activations = val[3]

    return parameters
end

function ChangeMaxEpochs(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MaxEpoch_" , string(val[1]))
    get_function(parameters).max_epochs = val
    return parameters
end



function GetDataConfig(experiment_config)
    return experiment_config.data_config
end

function GetSAENetwork(experiment_config)
    return experiment_config.sae_network
end

function GetFFNNetwork(experiment_config)
    return experiment_config.ffn_network
end

function GetSAETraining(experiment_config)
    return experiment_config.sae_sgd
end

function GetRBMTraining(experiment_config)
    return experiment_config.rbm_cd
end

function GetFFNTraining(experiment_config)
    return experiment_config.ffn_sgd
end

function GetOGDTraining(experiment_config)
    return experiment_config.ogd
end

function GetOGDHOTraining(experiment_config)
    return experiment_config.ogd_ho
end

function GenerateGridBasedParameterSets(vps, base_parameters)
    first = vps[1]
    #firstvals = length(first[3]) > 1 ? first[3] : [first[3]]
    one_samples = map(vp -> first[2].(first[1], deepcopy(base_parameters), vp), first[3])
    combos = length(first[3]) > 1 ?  one_samples : [one_samples]
    if length(vps) > 1
        for i in 2:length(vps)
            combos = mapreduce(current_sample -> mapreduce(y -> (vps[i][2](vps[i][1], deepcopy(current_sample), y)), vcat, vps[i][3]), vcat, combos)
        end
    end
    return combos
end

function HyperparameterRangeSearch(dataset, network_parameters, rbm_parameters, base_ffn_parameters, attribute_change_function, values)
    results = []

    for i in values

        attribute_change_function(base_ffn_parameters, i)
        network, rbm_records, ffn_records = TrainEncoderRBNMFFNNetwork(dataset, network_parameters, rbm_parameters, base_ffn_parameters)

        run_times = map(x -> x.run_time, ffn_records)
        test_accuracy = map(x -> x.test_accuracy, ffn_records)
        test_cost = map(x -> x.test_cost, ffn_records)
        epoch_numbers = map(x -> x.epoch_number, ffn_records)

        push!(results, (i, epoch_numbers, test_cost, test_accuracy, run_times))
    end

    return (results)
end

function GraphHyperparameterResults(results, output_dir, file_name, val_name)

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    #Run Times
    max_y_runtimes = maximum(reduce(vcat, map(x -> x[5], results)))*1.1
    min_y_times =  minimum(reduce(vcat, map(x -> x[5], results)))*0.9

    runtime_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    runtime_series = map(x -> x[5], results)

    runtime_plot = plot(runtime_series, ylims = (min_y_times, max_y_runtimes), labels = runtime_labels, ylabel = "Run Times", xlabel = "Epoch")

    #Cost
    cost_max_y = maximum(reduce(vcat, map(x -> x[3], results)))*1.1
    cost_min_y =  minimum(reduce(vcat, map(x -> x[3], results)))*0.9

    cost_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    cost_series = map(x -> x[3], results)

    cost_plot = plot(cost_series, ylims = (cost_min_y, cost_max_y), labels = cost_labels, ylabel = "Costs", xlabel = "Epoch")

    #Accuracy

    accuracy_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    accuracy_series = map(x -> x[4], results)

    accuracy_plot = plot(accuracy_series, ylims = (0, 1.1), labels = accuracy_labels, ylabel = "Accuracy", xlabel = "Epoch")

    savefig(plot(cost_plot, accuracy_plot, runtime_plot, layout = 3, size = (1000,1000)), string(output_dir , file_name, ".html"))
end

end
