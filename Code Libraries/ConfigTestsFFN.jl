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
        #layers = [30; map(x -> layer_sizes[x], 1:length(layer_sizes)); output_size]

        activations = []
        for i in 1:(length(layers)-1)
            push!(activations, primary_activation)
        end

        ffn_net_par = NetworkParameters("FFN",
                                        layers, #layer_sizes
                                        activations, #layer_activations
                                        InitializationFunctions.DCUniformInit, #Init
                                        LinearActivation, #Output Activation
                                        nothing) #Encoding Activation

        ffn_sgd_par = TrainingParameters("FFN",
                                        0.01, #max_learning_rate
                                        0.0001, #min_learning_rate
                                        100,  #epoch_cycle_max
                                        32, #minibatch_size
                                        -1, #max_epochs #TODO Change this !
                                        (0.0001, 100), #stopping_parameters
                                        NonStopping, #stopping_function
                                        0.0, #l1_lambda
                                        MeanSquaredError(), #cost_function
                                        [1.0], #training_splits
                                        false, #is_denoising
                                        0.0) #denoising_variance

        ogd_par = OGDTrainingParameters("FFN-OGD", 0.01, true, MeanSquaredError(), 0)

        return FFNExperimentConfig(seed, set_name, false, data_config, sae_config_id, encoder, ffn_net_par, ffn_sgd_par, ogd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    set_name = string("Iteration10 FFN AGL PBO ", string(layer_sizes))
    jsedata = ReadJSETop40Data()

    dataset = jsedata[:, [:AGL]]#,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

    vps = []

    #push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.005, 0.01, 0.05)))
    #push!(vps, (GetFFNTraining, ChangeL1Reg, (0, 0.1, 0.5)))
    #push!(vps, (GetFFNTraining, ChangeDenoisingVariance, (0, 0.1)))

    push!(vps, (GetFFNTraining, ChangeMaxLearningRate, (0.5, 0.1, 0.01, 0.001)))
    #push!(vps, (GetFFNTraining, ChangeMinLearningRate, (0.001, 0.0001)))
    push!(vps, (GetFFNTraining, ChangeMaxEpochs, (10, 100, 1000, 2000)))

    push!(vps, (GetOGDTraining, ChangeMaxLearningRate, (0.01, 0.05, 0.1, 0)))

    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.4, 1.0)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.2, 0.6)))
    #push!(vps, (GetFFNTraining, ChangeTrainingSplits, (0.2, 0.4, 0.6, 1.0)))



    combos = []
    for s in sae_configs
        sae_setname = string(set_name, " SAE ", s)
        sae_combos = GenerateGridBasedParameterSets(vps, GenerateBaseFFNConfig(sae_setname, dataset, s))
        for c in sae_combos
            push!(combos, c)
        end
    end

    ffn_results = map(ep -> RunFFNConfigurationTest(ep, dataset), combos)

    #PlotEpochs(map(x -> x[1], ffn_results), string(set_name, " Epochs"))
    #PlotGradientChangesCombined(ffn_results, 5, string(set_name," Combined Gradients"))
    #PlotActivations(ffn_results, string(set_name, " Activations"))
    #PlotOGDResults(ffn_results, string(set_name, " OGD Results"))
    return ffn_results
end


#sae_choices = (1533, 1497, 1554, 1639, 147, 1534, 1468, 1501, 318, 1284, 1535, 1553, 1059, 1508, 333)
#nosae_choices = (1533,1534,1535)
#sae_ten_choices = (318, 1508, 1639)
sae_choices = (26)

RunNLayerReLUFFNTest((100),         sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((100, 100),    sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((200),         sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((200, 200),    sae_choices, LeakyReluActivation)

#for i in 1:10
RunNLayerReLUFFNTest((10),                          sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((10, 10),                      sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((10, 10, 10),                  sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((10, 10, 10, 10),              sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((20),                          sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((20, 20),                      sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((20, 20, 20),                  sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((20, 20, 20, 20),              sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((30),                          sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((30, 30),                      sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((30, 30, 30),                  sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((30, 30, 30, 30),              sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((40),                          sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((40, 40),                      sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((40, 40, 40),                  sae_choices, LeakyReluActivation)
RunNLayerReLUFFNTest((40, 40, 40, 40),              sae_choices, LeakyReluActivation)
#end


#RunNLayerReLUFFNTest((240),                 sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((240, 240),            sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120),                 sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 120),            sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((60),  sae_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((60, 60),  sae_choices, LeakyReluActivation)

#TODO Smarter exploration of these sizes based on results seen in 2 layers
#RunNLayerReLUFFNTest((60, 60, 60),  sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 120, 120),       sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((240, 240, 240),       sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((60, 60, 60, 60),  sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((120, 120, 120, 120),  sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((240, 240, 240, 240),  sae_ten_choices, LeakyReluActivation)
#RunNLayerReLUFFNTest((60, 60, 60, 60, 60),  sae_ten_choices, LeakyReluActivation)
