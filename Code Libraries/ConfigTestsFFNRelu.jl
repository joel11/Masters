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

function RunNLayerReLUFFNTest(layer_size, num_hidden, sae_configs)

    srand(1)

    function GenerateBaseFFNConfig(set_name, dataset, sae_config_id)

        seed = abs(Int64.(floor(randn()*100)))
        sae_network, data_config = ReadSAE(sae_config_id)
        encoder = GetAutoencoder(sae_network)

        output_size = dataset != nothing ? (size(dataset,2) * length(data_config.prediction_steps)) : (length(data_config.variation_values) * length(data_config.prediction_steps))

        layers = [OutputSize(encoder); map(x -> layer_size, 1:num_hidden); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, ReluActivation)
        end
        activations[end] = LinearActivation

        ffn_net_par = NetworkParameters("FFN", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation)
        ffn_sgd_par = TrainingParameters("FFN", 0.1, 0.00001, 1,  20, 0.0, 1000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.99])
        ogd_par = OGDTrainingParameters("FFN-OGD", 0.001, true, MeanSquaredError())

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Reduced Scaling FFN ", num_hidden, " Layer ReLU ", num_hidden, "x", layer_size)
    #jsedata = ReadJSETop40Data()
    dataset = nothing #jsedata[:, [:AGL]]

    vps = []

    push!(vps, (GetFFNNetwork, ChangeOutputActivation, (LinearActivation, ReluActivation)))
    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.1, 0.01, 0.00001)))
    #push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.00001, 0.001)))



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
    PlotGradientChangesCombined(ffn_results, 5, string(set_name," Combined Gradients"))
    PlotActivations(ffn_results, string(set_name, " Activations"))
    PlotOGDResults(ffn_results, string(set_name, " OGD Results"))
    return ffn_results
end

#Overfit Test
#sae_choices = (43)
#results = RunNLayerReLUFFNTest(40, 1, sae_choices)
#config_ids = map(r -> r[1], results)
#configs = mapreduce(x->string(x, "_"), string, config_ids)[1:(end-1)]
#ProfitPlots(config_ids, string("ProfitPlot_", configs))

#Limited
#limited_sae_choices = ( 318, 302, 286,190, 254)
limited_sae_choices = ( 318,  286, 254)
#Nonlimited
#normal_sae_choices = (317, 301, 285, 189, 253)
normal_sae_choices = (317,  285,  253)

#RunNLayerReLUFFNTest(40, 1, limited_sae_choices)
#RunNLayerReLUFFNTest(40, 2, limited_sae_choices)
#RunNLayerReLUFFNTest(40, 3, limited_sae_choices)

RunNLayerReLUFFNTest(80, 1, limited_sae_choices)
RunNLayerReLUFFNTest(80, 2, limited_sae_choices)
RunNLayerReLUFFNTest(80, 3, limited_sae_choices)

RunNLayerReLUFFNTest(40, 1, normal_sae_choices)
RunNLayerReLUFFNTest(40, 2, normal_sae_choices)
RunNLayerReLUFFNTest(40, 3, normal_sae_choices)

RunNLayerReLUFFNTest(80, 1, normal_sae_choices)
RunNLayerReLUFFNTest(80, 2, normal_sae_choices)
RunNLayerReLUFFNTest(80, 3, normal_sae_choices)

#RunNLayerReLUFFNTest(15, 1, sae_choices)
#RunNLayerReLUFFNTest(15, 2, sae_choices)
#RunNLayerReLUFFNTest(15, 3, sae_choices)
