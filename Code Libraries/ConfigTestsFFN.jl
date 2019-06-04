workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
#using CSCV
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess
using DataJSETop40
using BSON

using ExperimentGraphs

function RunNLayerReLUFFNTest(layer_sizes, sae_configs, primary_activation)

    srand(2)

    function GenerateBaseFFNConfig(set_name, dataset, sae_config_id)

        srand(2)
        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [OutputSize(encoder); map(x -> layer_sizes[x], 1:length(layer_sizes)); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, primary_activation)
        end

        ffn_net_par = NetworkParameters("FFN",
            layers, #layer_sizes
            activations, #layer_activations
            InitializationFunctions.XavierGlorotNormalInit, #Init
            LinearActivation, #Output Activation
            nothing) #Encoding Activation

        ffn_sgd_par = TrainingParameters("FFN",
                                        0.001, #max_learning_rate
                                        0.0001, #min_learning_rate
                                        100,  #epoch_cycle_max
                                        20, #minibatch_size
                                        400, #max_epochs
                                        (0.0001, 100), #stopping_parameters
                                        NonStopping, #stopping_function
                                        0.0, #l1_lambda
                                        MeanSquaredError(), #cost_function
                                        [1.0], #training_splits
                                        false, #is_denoising
                                        0.0) #denoising_variance

        ogd_par = OGDTrainingParameters("FFN-OGD", 0.001, true, MeanSquaredError(), 0)

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Iteration4_2 FFN Tests ", string(layer_sizes))
    #jsedata = ReadJSETop40Data()
    dataset = nothing #jsedata[:, [:ACL, :AGL]] #nothing

    vps = []

    #push!(vps, (GetFFNNetwork, ChangeOutputActivation, (LinearActivation, LeakyReluActivation)))
    #push!(vps, (GetFFNNetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    #push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.00001, 0.0001, 0.001, 0.01)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.8, 1.0)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)))
    #push!(vps, (GetFFNTraining, ChangeMaxEpochs, (1, 5, 10, 50, 100)))
    push!(vps, (GetFFNTraining, ChangeL1Reg, (0, 0.1)))
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.01, 0.05, 0.1)))
    push!(vps, (GetFFNNetwork, ChangeInit, (XavierGlorotUniformInit, DCUniformInit)))
    push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.01, 0.05)))

    combos = []
    for s in sae_configs
        sae_setname = string(set_name, " SAE ", s)
        sae_combos = GenerateGridBasedParameterSets(vps, GenerateBaseFFNConfig(sae_setname, dataset, s))
        for c in sae_combos
            push!(combos, c)
        end
    end

    ffn_results = map(ep -> RunFFNConfigurationTest(ep, dataset), combos)

    PlotEpochs(map(x -> x[1], ffn_results), string(set_name, " Epochs"))
    #PlotGradientChangesCombined(ffn_results, 5, string(set_name," Combined Gradients"))
    #PlotActivations(ffn_results, string(set_name, " Activations"))
    #PlotOGDResults(ffn_results, string(set_name, " OGD Results"))
    return ffn_results
end

sae_choices = (18140,18914,18259,18311,19481, 20662,18917,18260,18314,18766,18119,18191,19343,18344,18751)

#RunNLayerReLUFFNTest((120, 60), sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 120), sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 120, 120), sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 90, 90, 60), sae_choices, LeakyReluActivation)

#RunNLayerReLUFFNTest((120, 90, 60), sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120), sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((120, 60), sae_choices, LeakyReluActivation)
