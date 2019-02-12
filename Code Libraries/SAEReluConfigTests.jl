#module SAEReluConfigTests

workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
using FinancialFunctions
using DatabaseOps
using HyperparameterOptimization
using ExperimentProcess
using DataJSETop40
using BSON

using ExperimentGraphs

export  RunReLUSAETest

function Run2LayerReLUSAETest(encoding_layer, layer_size)

    srand(12345678912345678)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,  5000,  [1,7,30],  [0.6],  [0.8, 1.0],  [2], var_pairs)

        input_size =  (length(var_pairs)*length(data_config.deltas))
        layers = [input_size, layer_size, layer_size, encoding_layer]
        activations = [ReluActivation, ReluActivation, ReluActivation]

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation)
        sae_sgd_par = TrainingParameters("SAE", 3.0, Inf, 1,  20, 0.0, 5000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []

    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.001, 0.005))) #0.0001, 0.001, 0.01, 0.1)))
    #push!(vps, (GetSAETraining, ChangeMinibatchSize, (10, 30)))
    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.0001, 0.001, 0.01)))
    push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.0001, 0.01)))
    #push!(vps, (GetSAENetwork, ChangeInit, (InitializationFunctions.XavierGlorotNormalInit, InitializationFunctions.NormalRandomInit)))
    #push!(vps, (GetSAENetwork, ChangeInit, (InitializationFunctions.XavierGlorotNormalInit, InitializationFunctions.HintonUniformInit, InitializationFunctions.HeUniformInit)))

    set_name = string("2 Layer ReLU ", layer_size, "x", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    sae_results = map(ep -> RunSAEConfigurationTest(ep, nothing), combos)
    config_ids = map(x -> x[1], sae_results)

    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    prefix = "2 Layer ReLU "
    PlotSAERecontructions(sae_results, string(prefix, "Recons ", set_name))
    PlotEpochs(config_ids, string(prefix, "Epochs ", set_name))
    PlotGradientChangesCombined(sae_results, 5, string(prefix,"Combined Gradients ", set_name))
    PlotActivations(sae_results, string(prefix, "Activations ", set_name))

    return sae_results
end

function Run1LayerReLUSAETest(encoding_layer, layer_size)

    srand(12345678912345678)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,  5000,  [1,7],  [0.6],  [0.8, 1.0],  [2], var_pairs)

        input_size =  (length(var_pairs)*length(data_config.deltas))
        layers = [input_size, layer_size, encoding_layer]
        activations = [ReluActivation, ReluActivation]

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation)
        sae_sgd_par = TrainingParameters("SAE", 3.0, Inf, 1,  20, 0.0, 2000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []

    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.001, 0.005))) #0.0001, 0.001, 0.01, 0.1)))
    #push!(vps, (GetSAETraining, ChangeMinibatchSize, (10, 30)))
    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.0001, 0.001, 0.01)))

    push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.0001, 0.01)))

    #push!(vps, (GetSAENetwork, ChangeInit, (InitializationFunctions.XavierGlorotNormalInit, InitializationFunctions.NormalRandomInit)))
    #push!(vps, (GetSAENetwork, ChangeInit, (InitializationFunctions.XavierGlorotNormalInit, InitializationFunctions.HintonUniformInit, InitializationFunctions.HeUniformInit)))

    set_name = string("1 Layer ReLU ", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    sae_results = map(ep -> RunSAEConfigurationTest(ep, nothing), combos)
    config_ids = map(x -> x[1], sae_results)

    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    prefix = "1 Layer ReLU "
    PlotSAERecontructions(sae_results, string(prefix, "Recons ", set_name))
    PlotEpochs(config_ids, string(prefix, "Epochs ", set_name))
    PlotGradientChangesCombined(sae_results, 5, string(prefix,"Combined Gradients ", set_name))
    PlotActivations(sae_results, string(prefix, "Activations ", set_name))

    return sae_results
end

#Run1LayerReLUSAETest(2, 25)
#Run2LayerReLUSAETest(2, 25)
#Run2LayerReLUSAETest(10, 25)
#Run2LayerReLUSAETest(8, 25)
#Run2LayerReLUSAETest(6, 25)
#Run2LayerReLUSAETest(4, 25)
#Run1LayerReLUSAETest(10, 25)
#Run1LayerReLUSAETest(8, 25)
#Run1LayerReLUSAETest(6, 25)
#Run1LayerReLUSAETest(4, 25)
#Run2LayerReLUSAETest(10, 40)
#Run2LayerReLUSAETest(8, 40)
#Run2LayerReLUSAETest(6, 40)
#Run2LayerReLUSAETest(4, 40)
#Run2LayerReLUSAETest(2, 40)
#Run1LayerReLUSAETest(10, 40)
#Run1LayerReLUSAETest(8, 40)
#Run1LayerReLUSAETest(6, 40)
#Run1LayerReLUSAETest(4, 40)
#Run1LayerReLUSAETest(2, 40)

#Run2LayerReLUSAETest(2, 40)
#Run2LayerReLUSAETest(4, 40)
#Run1LayerReLUSAETest(6, 40)
#Run1LayerReLUSAETest(8, 40)
Run2LayerReLUSAETest(10, 30)

#end
