module ExperimentProcessTrainFFN

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess
using DataJSETop40
using BSON

using ExperimentGraphs

export RunFFNExperiment



function RunNLayerFFNTest(layer_sizes, sae_choices,
                            network_hidden_layer_activation,
                            experiment_set_name,
                            dataset_name,
                            dataset,
                            network_initialization_functions,
                            network_output_activation,
                            sgd_max_learning_rates,
                            sgd_min_learning_rates,
                            sgd_learning_rate_epoch_length,
                            sgd_minibatch_size,
                            sgd_learning_rate_max_epochs,
                            sgd_l1_lambda,
                            sgd_validation_set_split,
                            sgd_denoising_enabled,
                            sgd_denoising_variance,
                            ogd_learning_rates)

    srand(2)

    function GenerateBaseFFNConfig(set_name, dataset, sae_config_id)

        srand(2)
        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [OutputSize(encoder); map(x -> layer_sizes[x], 1:length(layer_sizes)); output_size]
        #layers = [30; map(x -> layer_sizes[x], 1:length(layer_sizes)); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, network_hidden_layer_activation)
        end

        ffn_net_par = NetworkParameters("FFN",
                                        layers, #layer_sizes
                                        activations, #layer_activations
                                        network_initialization_functions[1], #Init
                                        network_output_activation, #Output Activation
                                        nothing) #Encoding Activation

        ffn_sgd_par = TrainingParameters("FFN",
                                        0.01, #max_learning_rate
                                        0.01, #min_learning_rate
                                        100,  #epoch_cycle_max
                                        32, #minibatch_sizes
                                        1000, #max_epochs
                                        (0.0001, 100), #stopping_parameters
                                        NonStopping, #stopping_function
                                        0.0, #l1_lambda
                                        MeanSquaredError(), #cost_function
                                        [1.0], #training_splits
                                        false, #is_denoising #TODO SET TO FALSE
                                        0.0) #denoising_variance

        ogd_par = OGDTrainingParameters("FFN-OGD",
                                        0.05,
                                        true,
                                        MeanSquaredError(),
                                        0)

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string(experiment_set_name, string(layer_sizes))

    vps = []

    push!(vps, (GetFFNNetwork, ChangeInit, network_initialization_functions))
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, sgd_max_learning_rates))
    push!(vps, (GetFFNTraining, ChangeMinLearningRate, sgd_min_learning_rates))
    push!(vps, (GetFFNTraining, ChangeLearningRateCycle, sgd_learning_rate_epoch_length))
    push!(vps, (GetFFNTraining, ChangeMinibatchSize, sgd_minibatch_size))
    push!(vps, (GetFFNTraining, ChangeMaxEpochs, sgd_learning_rate_max_epochs))
    push!(vps, (GetFFNTraining, ChangeL1Reg, sgd_l1_lambda))
    push!(vps, (GetFFNTraining, ChangeIsDenoising, sgd_denoising_enabled))
    push!(vps, (GetFFNTraining, ChangeDenoisingVariance, sgd_denoising_variance))
    push!(vps, (GetFFNTraining, ChangeIsDenoising, sgd_denoising_enabled))
    push!(vps, (GetOGDTraining, ChangeMaxLearningRate, ogd_learning_rates))

    combos = []
    for s in sae_choices
        sae_setname = string(set_name, " SAE ", s)
        sae_combos = GenerateGridBasedParameterSets(vps, GenerateBaseFFNConfig(sae_setname, dataset, s))
        for c in sae_combos
            push!(combos, c)
        end
    end

    i = 1
    for ep in combos
        println(string("$i/", length(combos)))
        RunFFNConfigurationTest(ep, dataset)
        i = i + 1
    end

end

function RunFFNExperiment(experiment_set_name, sae_choices,
                            dataset_name,
                            dataset,
                            network_initialization_functions,
                            network_hidden_layer_activation,
                            network_output_activation,
                            network_layer_sizes,
                            sgd_max_learning_rates,
                            sgd_min_learning_rates,
                            sgd_learning_rate_epoch_length,
                            sgd_minibatch_size,
                            sgd_learning_rate_max_epochs,
                            sgd_l1_lambda,
                            sgd_validation_set_split,
                            sgd_denoising_enabled,
                            sgd_denoising_variance,
                            ogd_learning_rates)

    for l in network_layer_sizes
            RunNLayerFFNTest(       l,
                                        sae_choices,
                                        network_hidden_layer_activation,
                                        experiment_set_name,
                                        dataset_name,
                                        dataset,
                                        network_initialization_functions,
                                        network_output_activation,
                                        sgd_max_learning_rates,
                                        sgd_min_learning_rates,
                                        sgd_learning_rate_epoch_length,
                                        sgd_minibatch_size,
                                        sgd_learning_rate_max_epochs,
                                        sgd_l1_lambda,
                                        sgd_validation_set_split,
                                        sgd_denoising_enabled,
                                        sgd_denoising_variance,
                                        ogd_learning_rates)
    end
end

end
