function GenerateActivationFunctions(number_layers)
    activation_functions = Array{Function,1}()
    for in in 1:number_layers
        push!(activation_functions, SigmoidActivation)
    end
    return (activation_functions)
end

#=function TrainEncoderRBNMFFNNetwork(dataset::DataSet, network_parameters::NetworkParameters, rbm_parameters::TrainingParameters, ffn_parameters::TrainingParameters)

    encoder_data = dataset#CreateEncoderDataset(dataset)
    rbm_network, rbm_records = TrainRBMNetwork(encoder_data, network_parameters, rbm_parameters)
    sgd_records = RunSGD(encoder_data, rbm_network, ffn_parameters)

    return (rbm_network, rbm_records, sgd_records)
end=#


function CalculateLambdaErrors(network::NeuralNetwork, activations::Array{Array{Float64,2}}, training_output::Array{Float64,2}, cost_function)
    #error = activations[end] - training_output
    z_vals = CalculateZVal(network.layers[end].weights, activations[(end-1)])
    #derivative_activations = FunctionDerivatives[layers[end].activation](z_vals)
    lambda_L =  cost_function.Delta(activations[end], training_output, z_vals, network.layers[end].activation)# derivative_activations)

    lambdas = Array{Array{Float64,2}}(length(network.layers))#[lambda_L])
    lambdas[1] = lambda_L
    #lambdas = [lambda_L]
    #lambdas = Array{Array{Float64,2}}([lambda_L])

    for i in (length(network.layers)-1):-1:1
        z_vals = CalculateZVal(network.layers[i].weights,activations[(i)])

        z_der = FunctionDerivatives[network.layers[i].activation](z_vals)
        output_act = lambdas[(length(network.layers)-i)] * network.layers[(i+1)].weights'[:, 2:end]
        lambda = output_act .* z_der
        lambdas[length(network.layers) - i + 1] = lambda
        #push!(lambdas, lambda)
    end

    reverse_lambdas = lambdas[(length(lambdas):-1:1)]
    return reverse_lambdas
    #return lambdas
end


function RunSAEConfigurationTest(ep, dataset)

    srand(ep.seed)

    ################################################################################
    #a. Record all config
    config_id = RecordExperimentConfig(ep)

    ################################################################################
    #b. Prepare data accordingly
    data_raw = dataset == nothing ? GenerateDataset(ep.data_config.data_seed, ep.data_config.steps, ep.data_config.variation_values) : dataset
    data_splits = SplitData(data_raw, ep.data_config.process_splits)
    processed_data = map(x -> ProcessData(x, ep.data_config.deltas, ep.data_config.prediction_steps), data_splits)
    #saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], ep.data_config.training_splits), processed_data)
    #saesgd_data = NormalizeDatasetForTanh(CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits))
    saesgd_data = NormalizeDatasetForSigmoid(CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits))
    #saesgd_data = CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits)
    ################################################################################
    #c. Run training, and record all epochs

    ## SAE Training & Encoding
    training_objects = (ep.rbm_pretraining == true ? (TrainRBMSAE(config_id, "SAE-SGD-RBM", saesgd_data, ep.sae_network, ep.rbm_cd, ep.sae_sgd))
                                              : (TrainInitSAE(config_id, "SAE-SGD-Init", saesgd_data, ep.sae_network, ep.sae_sgd)))


    full_network = training_objects[end]
    sgd_records = training_objects[(end-1)]
    actual_data = saesgd_data.testing_input

    ffdata = Feedforward(full_network, actual_data)
    reconstructed_data = ffdata[end]
    #data_pair = (actual_data, reconstructed_data)
    data_pair = (DenormalizatData(actual_data, saesgd_data.scaling_min, saesgd_data.scaling_max)
                , DenormalizatData(reconstructed_data, saesgd_data.scaling_min, saesgd_data.scaling_max))

    return (config_id, ep.experiment_set_name, data_pair, sgd_records, ffdata)
end
function DenormalizatData(data, min, max)
    diff = max - min
    newds = deepcopy(data)
    newds = newds .* diff .+ min
    return newds
end

function NormalizeDatasetForTanh(dataset)

    maxval = max(maximum(dataset.training_input)
                ,maximum(dataset.testing_input)
                ,maximum(Array(dataset.training_output))
                ,maximum(Array(dataset.testing_output)))

    newds = deepcopy(dataset)

    newds.training_input = newds.training_input ./ maxval
    newds.testing_input = newds.testing_input ./ maxval

    training_output = Array(newds.training_output) ./ maxval
    testing_output = Array(newds.testing_output) ./ maxval

    for n in 1:length(names(newds.training_output))
        newds.training_output[names(newds.training_output)[n]] = training_output[:, n]
    end

    for n in 1:length(names(newds.testing_output))
        newds.testing_output[names(newds.testing_output)[n]] = testing_output[:, n]
    end

    return newds
end

function NormalizeDatasetForSigmoid(dataset)

    minval = min(minimum(dataset.training_input)
                ,minimum(dataset.testing_input)
                ,minimum(Array(dataset.training_output))
                ,minimum(Array(dataset.testing_output)))

    maxval = max(maximum(dataset.training_input)
                ,maximum(dataset.testing_input)
                ,maximum(Array(dataset.training_output))
                ,maximum(Array(dataset.testing_output)))

    den = maxval - minval

    newds = deepcopy(dataset)
    newds.scaling_min = minval
    newds.scaling_max = maxval


    newds.training_input = (newds.training_input .- minval) ./ den
    newds.testing_input = (newds.testing_input .- minval) ./ den

    training_output = (Array(newds.training_output) .- minval) ./ den
    testing_output = (Array(newds.testing_output) .- minval) ./ den

    for n in 1:length(names(newds.training_output))
        newds.training_output[names(newds.training_output)[n]] = training_output[:, n]
    end

    for n in 1:length(names(newds.testing_output))
        newds.testing_output[names(newds.testing_output)[n]] = testing_output[:, n]
    end

    return newds
end
