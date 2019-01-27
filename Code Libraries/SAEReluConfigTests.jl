module SAEReluConfigTests

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

using ExperimentGraphs

export  RunReLUSAETest

function RunReLUSAETest(encoding_layer, layer_size)

    srand(12345678912345678)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,  5000,  [1],  [0.6],  [0.8, 1.0],  [1], var_pairs)

        input_size =  (length(var_pairs)*length(data_config.deltas))
        layers = [input_size, layer_size, layer_size, encoding_layer]
        activations = [ReluActivation, ReluActivation,  ReluActivation]

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.HeUniformInit, ReluActivation)
        sae_sgd_par = TrainingParameters("SAE", 3.0, Inf, 1,  15, 0.0, 1000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())


        return ExperimentConfig(seed, set_name, false, data_config, sae_net_par, nothing , sae_sgd_par, nothing, nothing, nothing, true)
    end

    ################################################################################
    ##1. Configuration Variations

    vps = []

    push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.00001, 1.0)))#0.0001, 0.001, 0.01, 0.1)))
    #push!(vps, (GetSAETraining, ChangeMinibatchSize, (10, 30)))

    set_name = string("ReLU ", layer_size, "x", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    sae_results = map(ep -> RunSAEConfigurationTest(ep, nothing), combos)
    config_ids = map(x -> x[1], sae_results)


    PlotSAERecontructions(sae_results, string("ReLU Recons ", set_name))
    PlotEpochs(config_ids, string("ReLU Epochs ", set_name))
    PlotGradientChangesCombined(sae_results, 5, string("ReLU Combined Gradients ", set_name))
    #PlotLayerActivations(sae)
end

RunReLUSAETest(5, 15)

end
