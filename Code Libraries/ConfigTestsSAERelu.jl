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
        data_config = DatasetConfig(ds, datasetname,  5000,  [1, 5, 20],  [0.6],  [0.8, 1.0],  [2], var_pairs, NormalizeData)

        #layers = [(length(var_pairs)*length(data_config.deltas))]
        layers = [10*length(data_config.deltas)]
        for i in 1:num_hidden
            push!(layers, layer_size)
        end
        push!(layers, encoding_layer)

        activations = map(x -> primary_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotUniformInit, LinearActivation, LinearActivation)
        sae_sgd_par = TrainingParameters("SAE", 0.001, 0, 0,  20, 0.0, 500, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []

    push!(vps, (GetSAENetwork, ChangeEncodingActivation, (LinearActivation, primary_activation)))
    push!(vps, (GetSAENetwork, ChangeOutputActivation, (LinearActivation, primary_activation)))
    #push!(vps, (GetSAENetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    push!(vps, (GetSAETraining, ChangeMaxLearningRate, learning_rates))


    set_name = string("Linear Tests 2 Std ", num_hidden, "x", layer_size, "x", encoding_layer, " ", split(string(primary_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))
    #combos = [GenerateBaseSAEConfig(set_name, "Synthetic Set")]
    ################################################################################
    ##2a. Run Each SAE Configuration
    jsedata = ReadJSETop40Data()
    exp_data =  jsedata[:, [1, 4, 5, 7, 8, 9, 10, 11, 12, 14]] #jsedata[:, [:AGL, :ACL]]#nothing

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

#input: 30
#functions: Sigmoid, Relu, Linear
#Sizes & Encodings & Number Layers
#Encodings: 5; 15; 25
#Layer Size: 60; 120
#Num Layers: 1, 2, 3


RunNLayerReLUSAETest(5, 60, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(5, 60, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(5, 60, 3, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(5, 120, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(5, 120, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(5, 120, 3, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 60, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 60, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 60, 3, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 120, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 120, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(15, 120, 3, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 60, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 60, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 60, 3, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 120, 1, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 120, 2, SigmoidActivation, (0.01, 2.0))
RunNLayerReLUSAETest(25, 120, 3, SigmoidActivation, (0.01, 2.0))

#=
#RunNLayerReLUSAETest(5, 60, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 60, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 60, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(5, 60, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 60, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 60, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(5, 60, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 60, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 60, 3, LinearActivation, (0.001, 0.01))

#RunNLayerReLUSAETest(5, 120, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 120, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 120, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(5, 120, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 120, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 120, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(5, 120, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(5, 120, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(5, 120, 3, LinearActivation, (0.001, 0.01))

#RunNLayerReLUSAETest(15, 60, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 60, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 60, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(15, 60, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 60, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 60, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(15, 60, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 60, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 60, 3, LinearActivation, (0.001, 0.01))

#RunNLayerReLUSAETest(15, 120, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 120, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 120, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(15, 120, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 120, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 120, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(15, 120, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(15, 120, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(15, 120, 3, LinearActivation, (0.001, 0.01))

#RunNLayerReLUSAETest(25, 60, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 60, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 60, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(25, 60, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 60, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 60, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(25, 60, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 60, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 60, 3, LinearActivation, (0.001, 0.01))

#RunNLayerReLUSAETest(25, 120, 1, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 120, 1, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 120, 1, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(25, 120, 2, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 120, 2, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 120, 2, LinearActivation, (0.001, 0.01))
#RunNLayerReLUSAETest(25, 120, 3, SigmoidActivation, (0.001, 0.1, 1.0))
RunNLayerReLUSAETest(25, 120, 3, ReluActivation, (0.01, 0.05, 0.1, 0.25))
RunNLayerReLUSAETest(25, 120, 3, LinearActivation, (0.001, 0.01))
=#
