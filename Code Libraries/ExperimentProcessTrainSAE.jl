module ExperimentProcessTrainSAE

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

export  RunSAEExperiment

function RunNLayerReLUSAETest(encoding_layer, layer_sizes, network_hidden_layer_activation,
                                                            experiment_set_name,
                                                            dataset_name,
                                                            dataset,
                                                            data_sgd_validation_set_split,
                                                            data_sgd_ogd_split,
                                                            data_horizon_predictions,
                                                            data_scaling_function,
                                                            horizon_aggregation,
                                                            network_initialization_functions,
                                                            network_output_activation,
                                                            network_encoding_activation,
                                                            sgd_max_learning_rates,
                                                            sgd_min_learning_rates,
                                                            sgd_learning_rate_epoch_length,
                                                            sgd_minibatch_size,
                                                            sae_sgd_max_epochs,
                                                            sgd_l1_lambda,
                                                            sgd_validation_set_split,
                                                            sgd_denoising_enabled,
                                                            sgd_denoising_variance)

    srand(2)

    function GenerateBaseSAEConfig(set_name, datasetname)
        srand(2)

        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))

        data_config = DatasetConfig(ds,
                                    dataset_name,
                                    1,  #timesteps
                                    horizon_aggregation, #horizon aggregations
                                    data_sgd_ogd_split, #process split (for SAE/SGD & OGD)
                                    [0.8, 1.0], #validation set split
                                    data_horizon_predictions, #prediction step
                                    (1,1), #var pairs
                                    data_scaling_function) #scaling function

        #layers = [(length(var_pairs)*length(data_config.deltas))]
        layers = [1*length(data_config.deltas)]

        for i in 1:length(layer_sizes)
            push!(layers, layer_sizes[i])
        end
        push!(layers, encoding_layer)

        activations = map(x -> network_hidden_layer_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE",
                                        layers,
                                        activations,
                                        #InitializationFunctions.DCUniformInit,
                                        InitializationFunctions.XavierGlorotUniformInit,
                                        network_output_activation, #output
                                        network_encoding_activation) #encoding

        sae_sgd_par = TrainingParameters("SAE",
                                        0.01,           #max learning rate
                                        0.01,         #min learning rate
                                        100,            #epoch cycle max
                                        32,             #minibatch size
                                        1000,           #max epochs
                                        (0.0001, 100),  #stopping parameters
                                        NonStopping,    #stopping function
                                        0.0,            #l1 lambda
                                        MeanSquaredError(), #cost_function
                                        sgd_validation_set_split,              #validation set split
                                        false,          #denoising enabled
                                        0.0)            #denoising level

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []


    push!(vps, (GetSAENetwork, ChangeInit, network_initialization_functions))
    push!(vps, (GetSAETraining, ChangeMaxLearningRate, sgd_max_learning_rates))
    push!(vps, (GetSAETraining, ChangeMinLearningRate, sgd_min_learning_rates))
    push!(vps, (GetSAETraining, ChangeLearningRateCycle, sgd_learning_rate_epoch_length))
    push!(vps, (GetSAETraining, ChangeMinibatchSize, sgd_minibatch_size))
    push!(vps, (GetSAETraining, ChangeMaxEpochs, sae_sgd_max_epochs))
    push!(vps, (GetSAETraining, ChangeL1Reg, sgd_l1_lambda))
    push!(vps, (GetSAETraining, ChangeIsDenoising, sgd_denoising_enabled))
    push!(vps, (GetSAETraining, ChangeDenoisingVariance, sgd_denoising_variance))

    set_name = string(experiment_set_name, string(layer_sizes), "x", encoding_layer, " ", split(string(network_hidden_layer_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, dataset_name))

    ################################################################################
    ##2a. Run Each SAE Configuration
    sae_results = map(ep -> RunSAEConfigurationTest(ep, dataset), combos)
    config_ids = map(x -> x[1], sae_results)

    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    return sae_results
end

function RunSAEExperiment(experiment_set_name,
                            dataset_name,
                            dataset,
                            data_sgd_validation_set_split,
                            data_sgd_ogd_split,
                            data_horizon_predictions,
                            data_scaling_function,
                            data_horizon_aggregations,
                            network_initialization_functions,
                            network_hidden_layer_activation,
                            network_output_activation,
                            network_encoding_activation,
                            network_layer_sizes,
                            network_encoding_layers,
                            sgd_max_learning_rates,
                            sgd_min_learning_rates,
                            sgd_learning_rate_epoch_length,
                            sgd_minibatch_size,
                            sae_sgd_max_epochs,
                            sgd_l1_lambda,
                            sgd_validation_set_split,
                            sgd_denoising_enabled,
                            sgd_denoising_variance)

    println("A")

    for l in network_layer_sizes
        println(l)
        for e in network_encoding_layers
            for d in data_horizon_aggregations
                RunNLayerReLUSAETest(e, l,  network_hidden_layer_activation,
                                            experiment_set_name,
                                            dataset_name,
                                            dataset,
                                            data_sgd_validation_set_split,
                                            data_sgd_ogd_split,
                                            data_horizon_predictions,
                                            data_scaling_function,
                                            d,
                                            network_initialization_functions,
                                            network_output_activation,
                                            network_encoding_activation,
                                            sgd_max_learning_rates,
                                            sgd_min_learning_rates,
                                            sgd_learning_rate_epoch_length,
                                            sgd_minibatch_size,
                                            sae_sgd_max_epochs,
                                            sgd_l1_lambda,
                                            sgd_validation_set_split,
                                            sgd_denoising_enabled,
                                            sgd_denoising_variance)
            end
        end
    end
end


end
