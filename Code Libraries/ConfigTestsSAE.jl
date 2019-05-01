workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

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

using ExperimentGraphs

export  RunReLUSAETest

function RunNLayerReLUSAETest(encoding_layer, layer_size, num_hidden, primary_activation, learning_rates)

    srand(2)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,  5000,  [1, 5, 20],  [0.6],  [0.8, 1.0],  [2], var_pairs, LimitedNormalizeData)

        layers = [(length(var_pairs)*length(data_config.deltas))]
        #layers = [10*length(data_config.deltas)]
        for i in 1:num_hidden
            push!(layers, layer_size)
        end
        push!(layers, encoding_layer)

        activations = map(x -> primary_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotUniformInit, LinearActivation, LinearActivation)
        sae_sgd_par = TrainingParameters("SAE", 0.1, 0, 0,  20, 1000, (0.0001, 100), NonStopping, 0.0, MeanSquaredError(), [0.8], true, 0.1)

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []

    #push!(vps, (GetSAENetwork, ChangeEncodingActivation, (LinearActivation, primary_activation)))
    #push!(vps, (GetSAENetwork, ChangeOutputActivation, (LinearActivation, primary_activation)))
    #push!(vps, (GetSAENetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    push!(vps, (GetSAETraining, ChangeMaxLearningRate, learning_rates))
    #push!(vps, (GetSAETraining, ChangeIsDenoising, (true, false)))
    #push!(vps, (GetSAETraining, ChangeDenoisingVariance, (0.1, 0.01, 0.001, 0.0001, 0.00000000001)))

    set_name = string("Iteration3 SAE LeakyRelu v2 ", num_hidden, "x", layer_size, "x", encoding_layer, " ", split(string(primary_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))
    #combos = [GenerateBaseSAEConfig(set_name, "Synthetic Set")]
    ################################################################################
    ##2a. Run Each SAE Configuration
    jsedata = ReadJSETop40Data()
    exp_data =  nothing #jsedata[:, [1, 4, 5, 7, 8, 9, 10, 11, 12, 14]] #jsedata[:, [:AGL, :ACL]]#nothing

    sae_results = map(ep -> RunSAEConfigurationTest(ep, exp_data), combos)
    #sae_results = map(ep -> RunSAEConfigurationTest(ep, nothing), combos)
    config_ids = map(x -> x[1], sae_results)

    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    prefix = string(num_hidden, " Layers ")
    PlotSAERecontructions(sae_results, string(set_name, "Recons ", prefix))
    #PlotEpochs(config_ids, string(set_name, "Epochs ", prefix))
    #PlotGradientChangesCombined(sae_results, 5, string(set_name,"Combined Gradients ", prefix))
    #PlotActivations(sae_results, string(set_name, "Activations ", prefix))

    return sae_results
end

#Input: 18
#Encoding: 15, 12, 9, 6, 3
#Sizes: 40, 80
#Layers:1, 2, 3

RunNLayerReLUSAETest(15, 40, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(15, 40, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(15, 40, 3, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(15, 80, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(15, 80, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(15, 80, 3, LeakyReluActivation, (0.05, 0.1))

RunNLayerReLUSAETest(12, 40, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(12, 40, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(12, 40, 3, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(12, 80, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(12, 80, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(12, 80, 3, LeakyReluActivation, (0.05, 0.1))

RunNLayerReLUSAETest(9, 40, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(9, 40, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(9, 40, 3, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(9, 80, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(9, 80, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(9, 80, 3, LeakyReluActivation, (0.05, 0.1))

RunNLayerReLUSAETest(6, 40, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(6, 40, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(6, 40, 3, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(6, 80, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(6, 80, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(6, 80, 3, LeakyReluActivation, (0.05, 0.1))

RunNLayerReLUSAETest(3, 40, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(3, 40, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(3, 40, 3, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(3, 80, 1, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(3, 80, 2, LeakyReluActivation, (0.05, 0.1))
RunNLayerReLUSAETest(3, 80, 3, LeakyReluActivation, (0.05, 0.1))
