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

function RunNLayerReLUFFNTest(layer_size, num_hidden, sae_configs, primary_activation)

    srand(2)

    function GenerateBaseFFNConfig(set_name, dataset, sae_config_id)

        srand(2)
        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [OutputSize(encoder); map(x -> layer_size, 1:num_hidden); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, primary_activation)
        end

        ffn_net_par = NetworkParameters("FFN", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation, LinearActivation)
        ffn_sgd_par = TrainingParameters("FFN", 0.001, 0, 0,  20, 500, (0.0001, 100), NonStopping, 0.0, MeanSquaredError(), [1.0], false, 0.0)
        ogd_par = OGDTrainingParameters("FFN-OGD", 0.001, true, MeanSquaredError(), 0)

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Iteration3_15 FFN Validation 3 Test ", num_hidden, "x", layer_size)
    #jsedata = ReadJSETop40Data()
    dataset = nothing #jsedata[:, [:ACL, :AGL]] #nothing

    vps = []

    #push!(vps, (GetFFNNetwork, ChangeOutputActivation, (LinearActivation, LeakyReluActivation)))
    #push!(vps, (GetFFNNetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    #push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.00001, 0.0001, 0.001, 0.01)))
    #push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.0001, 0.001)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.8, 1.0)))
    push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)))
    #push!(vps, (GetFFNTraining, ChangeMaxEpochs, (1, 5, 10, 50, 100)))
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.001, 0.01)))
    #push!(vps, (GetFFNTraining, ChangeL1Reg, (0.1, 1.0, 10.0, 20.0, 40.0, 80.0, 120, 160)))

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

sae_choices = (17196, 17190, 17180, 17178)

RunNLayerReLUFFNTest(60, 2, sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest(60, 3, sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest(10, 2, sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest(40, 3, sae_choices, LeakyReluActivation)


#RunNLayerReLUFFNTest(40, 1, reg_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest(40, 3, reg_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest(80, 1, reg_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest(80, 3, reg_choices, LeakyReluActivation)








#mse_choices = (3560,3574,3580,3590,3613)
#mapes_choices = (3564,3575,3589,3601,3612)
#ltd_mse_choices = (3620,3626,3638,3650,3671)
#ltd_mapes_choices = (3624,3635,3649,3661,3673)
#all_saes  =  (3560,3574,3580,3590,3613, 3620,3626,3638,3650,3671)
#mapes_choices = (3564,3575,3589,3601,3612, 3624,3635,3649,3661,3673)
#leaky_saes = (8127, 8106, 8094, 8083, 8076)

#input = 18
#Hidden = 40, 80
#Layers = 1, 3
#Activations = ReLU, Linear

#relu_saes =         (8860, 8872, 8876, 8888, 8910)
#leakyrelu_saes =    (8800, 8812, 8816, 8828, 8850)

#RunNLayerReLUFFNTest(40, 1, relu_saes, ReluActivation)
#RunNLayerReLUFFNTest(40, 3, relu_saes, ReluActivation)
#RunNLayerReLUFFNTest(80, 1, relu_saes, ReluActivation)
#RunNLayerReLUFFNTest(80, 3, relu_saes, ReluActivation)

#RunNLayerReLUFFNTest(40, 1, leakyrelu_saes, LeakyReluActivation)
#RunNLayerReLUFFNTest(40, 3, leakyrelu_saes, LeakyReluActivation)
#RunNLayerReLUFFNTest(80, 1, leakyrelu_saes, LeakyReluActivation)
#RunNLayerReLUFFNTest(80, 3, leakyrelu_saes, LeakyReluActivation)

#RunNLayerReLUFFNTest(20, 1, all_saes, ReluActivation)
#RunNLayerReLUFFNTest(20, 2, all_saes, ReluActivation)
#RunNLayerReLUFFNTest(20, 3, all_saes, ReluActivation)

#RunNLayerReLUFFNTest(20, 1, all_saes, LinearActivation)
#RunNLayerReLUFFNTest(20, 2, all_saes, LinearActivation)
#RunNLayerReLUFFNTest(20, 3, all_saes, LinearActivation)
