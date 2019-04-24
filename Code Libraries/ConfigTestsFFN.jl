workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

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

using ExperimentGraphs

function RunNLayerReLUFFNTest(layer_size, num_hidden, sae_configs, primary_activation)

    srand(1)

    function GenerateBaseFFNConfig(set_name, dataset, sae_config_id)

        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [OutputSize(encoder); map(x -> layer_size, 1:num_hidden); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, primary_activation)
        end

        ffn_net_par = NetworkParameters("FFN", layers, activations, InitializationFunctions.XavierGlorotNormalInit, "", "")
        ffn_sgd_par = TrainingParameters("FFN", 0.1, 0, 0,  20, 500, (0.0001, 100), NonStopping, 0.0, MeanSquaredError(), [1.0], false, 0.0)
        ogd_par = OGDTrainingParameters("FFN-OGD", 0.001, true, MeanSquaredError())

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Iteration2_1 Tests FFN ", num_hidden, " Layer ReLU ", num_hidden, "x", layer_size)
    #jsedata = ReadJSETop40Data()
    dataset = nothing #jsedata[:, [:ACL, :AGL]] #nothing

    vps = []

    push!(vps, (GetFFNNetwork, ChangeOutputActivation, (LinearActivation, ReluActivation)))
    #push!(vps, (GetFFNNetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.00001, 0.0001, 0.001, 0.01)))
    push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.0001, 0.001)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.8, 1.0)))


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

#mse_choices = (3560,3574,3580,3590,3613)
#mapes_choices = (3564,3575,3589,3601,3612)
#ltd_mse_choices = (3620,3626,3638,3650,3671)
#ltd_mapes_choices = (3624,3635,3649,3661,3673)

#input = 18
#Hidden = 40, 80
#Layers = 1, 3
#Activations = ReLU, Linear

all_saes  =  (3560,3574,3580,3590,3613, 3620,3626,3638,3650,3671)

RunNLayerReLUFFNTest(40, 1, all_saes, ReluActivation)
RunNLayerReLUFFNTest(40, 3, all_saes, ReluActivation)
RunNLayerReLUFFNTest(80, 1, all_saes, ReluActivation)
RunNLayerReLUFFNTest(80, 3, all_saes, ReluActivation)

RunNLayerReLUFFNTest(40, 1, all_saes, LinearActivation)
RunNLayerReLUFFNTest(40, 3, all_saes, LinearActivation)
RunNLayerReLUFFNTest(80, 1, all_saes, LinearActivation)
RunNLayerReLUFFNTest(80, 3, all_saes, LinearActivation)
