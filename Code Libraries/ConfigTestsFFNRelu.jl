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
        ffn_sgd_par = TrainingParameters("FFN", 0.1, 0.00001, 1,  20, 0.0, 1000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [1.0])
        ogd_par = OGDTrainingParameters("FFN-OGD", 0.001, true, MeanSquaredError())

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Oscillation Tests FFN ", num_hidden, " Layer ReLU ", num_hidden, "x", layer_size)
    #jsedata = ReadJSETop40Data()
    dataset = nothing #jsedata[:, [:AGL]] #nothing

    vps = []

    #push!(vps, (GetFFNNetwork, ChangeOutputActivation, (LinearActivation, ReluActivation)))
    #push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.01, 0.00001)))
    push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.000001, 0.001)))



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

choices = (865)
RunNLayerReLUFFNTest(40, 2, choices)




using PlotlyJS


function RecreateStockPrices(config_names)
    configs = mapreduce(x->string(x, ","), string, collect(keys(config_names)))[1:(end-1)]
    best_query = string("select * from prediction_results where configuration_id in ($configs)")
    best_results = RunQuery(best_query)
    best_groups = by(best_results, [:stock], df -> [df])

    for i in 1:size(best_groups,1)
        timesteps = best_groups[i,2][:time_step]
        config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

        #actual = cumsum(config_groups[1,2][:actual])
        #predicted_one = cumsum(config_groups[1,2][:predicted])
        #predicted_two = cumsum(config_groups[2,2][:predicted])

        actual = (config_groups[1,2][:actual])
        predicted_one = (config_groups[1,2][:predicted])
        predicted_two = (config_groups[2,2][:predicted])

        stock_name = get(best_groups[i,1])

        t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
        t1 = scatter(;y=predicted_one, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][1])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))
        t2 = scatter(;y=predicted_two, x = timesteps, name=string(stock_name, "_predicted_", config_names[get(config_groups[1][2])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))

        recreation_plots = [t0, t1, t2]
        filename = string("recreation_", stock_name, "_", collect(keys(config_names))[1])
        savefig(plot(recreation_plots), string("/users/joeldacosta/desktop/", filename, ".html"))

    end
end
names = Dict(870 => "870", 871 => "871")
RecreateStockPrices(names)
