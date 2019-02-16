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

function RunNLayerReLUFFNTest(sae_config_id, layer_size, num_hidden)

    srand(12345678912345678)

    function GenerateBaseFFNConfig(set_name, dataset)

        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [10; map(x -> layer_size, 1:num_hidden); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, ReluActivation)
        end
        activations[end] = LinearActivation

        ffn_net_par = NetworkParameters("FFN", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation)
        ffn_sgd_par = TrainingParameters("FFN", 3.0, Inf, 1,  20, 0.0, 1000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
        ogd_par = OGDTrainingParameters("FFN", 0.001, true, MeanSquaredError())

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("FFN ", num_hidden, " Layer ReLU ", num_hidden, "x", layer_size)
    dataset = nothing

    vps = []
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.0001, 0.01)))
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseFFNConfig(set_name, dataset))

    ffn_results = map(ep -> RunFFNConfigurationTest(ep, dataset), combos)

    PlotEpochs(map(x -> x[1], ffn_results), string(set_name, " Epochs"))
    PlotGradientChangesCombined(ffn_results, 5, string(set_name," Combined Gradients"))
    PlotActivations(ffn_results, string(set_name, " Activations"))
    PlotOGDResults(ffn_results, string(set_name, " OGD Results"))
    return ffn_results
end

results = RunNLayerReLUFFNTest(305, 100, 2)
