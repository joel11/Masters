module ConfigGenerator

export ChangeLearningRateCycle, ChangeVariations,ChangeDeltas, ChangeTrainingSplits, ChangeIsDenoising, ChangeDenoisingVariance, ChangeEncodingActivation, ChangeOutputActivation,ChangeScalingFunction, ChangeMinLearningRate, ChangeMinMaxLearningRate, ChangeMaxLearningRate, ChangeL1Reg, ChangeL2Reg, ChangeMinibatchSize, GenerateGridBasedParameterSets, GetDataConfig, GetSAENetwork, GetFFNNetwork, GetSAETraining, GetFFNTraining, GetOGDTraining, GetOGDHOTraining, ChangeLayers, ChangeInit, GetRBMTraining, ChangeMaxEpochs

function ChangeInit(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_Init_" , split(string(val), ".")[end])
    get_function(parameters).initialization = val
    return parameters
end

function ChangeSAENetwork(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_SAE_" , split(string(val), ".")[end])
    get_function(parameters).sae_config_id = val
    return parameters
end

function ChangeIsDenoising(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_denoising_" , split(string(val), ".")[end])
    get_function(parameters).is_denoising = val
    return parameters
end

function ChangeDenoisingVariance(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_variance_" , split(string(val), ".")[end])
    get_function(parameters).denoising_variance = val
    return parameters
end

function ChangeOutputActivation(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_output_activation_" , split(string(val), ".")[end])
    get_function(parameters).output_activation = val
    return parameters
end

function ChangeEncodingActivation(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_encoding_activation_" , split(string(val), ".")[end])
    get_function(parameters).encoding_activation = val
    return parameters
end

function ChangeMaxLearningRate(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MaxLearningRate_" , string(val))
    get_function(parameters).max_learning_rate = val
    return parameters
end

function ChangeTrainingSplits(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_CVSplit_" , string(val))
    get_function(parameters).training_splits = val
    return parameters
end

function ChangeMinLearningRate(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MinLearningRate_" , string(val))
    get_function(parameters).min_learning_rate = val
    return parameters
end

function ChangeLearningRateCycle(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_LRCycle_" , string(val))
    get_function(parameters).epoch_cycle_max = val
    return parameters
end

function ChangeMinMaxLearningRate(get_function, parameters,val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_MinMaxLearningRate_" , string(val))
    get_function(parameters).min_learning_rate = val[1]
    get_function(parameters).max_learning_rate = val[2]
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

function ChangeScalingFunction(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_Scaling_" , string(val))
    get_function(parameters).scaling_function = val
    return parameters
end

function ChangeDeltas(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_Deltas_" , string(val))
    println(val)
    get_function(parameters).deltas = val
    return parameters
end

function ChangeVariations(get_function, parameters, val)
    parameters.experiment_set_name = string(parameters.experiment_set_name , "_Variations_" , string(val))
    get_function(parameters).variation_values = val
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

end
