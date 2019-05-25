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

#var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))

function GenerateOneAssetConfig()
    return [(0.6, 0.05)]
end

function GenerateTwoAssetConfig()

    means = (0.05, 0.6)
    variances = (0.02, 0.4)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2])))
    end

    for v in variances
            push!(allcombos, ((means[1],v),(means[2], v)))
    end

    return allcombos
end

function GenerateThreeAssetConfig()

    means = (0.05, 0.2, 0.6)
    variances = (0.02, 0.1, 0.4)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2]),(m, variances[3])))
    end

    for v in variances
        push!(allcombos, ((means[1], v),(means[2], v),(means[3], v)))
    end

    return allcombos
end

function GenerateFourAssetConfig()

    means = (0.05, 0.2, 0.4, 0.8)
    variances = (0.02, 0.1, 0.3, 0.6)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2]),(m, variances[3]),(m, variances[4])))
    end

    for v in variances
        push!(allcombos, ((means[1],v),(means[2],v),(means[3],v),(means[4],v)))
    end

    return allcombos
end

function RunNLayerReLUSAETest(encoding_layer, layer_sizes, primary_activation, learning_rates)

    srand(2)

    function GenerateBaseSAEConfig(set_name, datasetname)
        srand(2)

        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5),
                     (0.7, 0.2),
                     (0.05, 0.4),
                     (0.05, 0.5),
                     (0.04, 0.1),
                     (0.02, 0.15),
                     (0.01, 0.05),
                     (-0.8, 0.55),
                     (-0.4, 0.15),
                     (-0.1, 0.2))

        data_config = DatasetConfig(ds, datasetname,
                                    5000,  #timesteps
                                    [1, 5, 20], #delta aggregatios
                                    [0.6], #process split (for SAE/SGD & OGD)
                                    [0.8, 1.0], #validation set split
                                    [5], #prediction step
                                    var_pairs,
                                    LimitedNormalizeData) #scaling function

        #layers = [(length(var_pairs)*length(data_config.deltas))]
        layers = [10*length(data_config.deltas)]

        for i in 1:length(layer_sizes)
            push!(layers, layer_sizes[i])
        end
        push!(layers, encoding_layer)

        activations = map(x -> primary_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations,
                                        InitializationFunctions.XavierGlorotUniformInit,
                                        LinearActivation, #output
                                        LinearActivation) #encoding

        sae_sgd_par = TrainingParameters("SAE",
                                        0.005,    #max learning rate
                                        0.0001,        #min learning rate
                                        100,        #epoch cycle max
                                        20,       #minibatch size
                                        400,      #max epochs
                                        (0.0001, 100), #stopping parameters
                                        NonStopping,   #stopping function
                                        0.0,           #l1 lambda
                                        MeanSquaredError(), #cost_function
                                        [0.8],              #validation set split
                                        false,  #denoising enabled
                                        0.0)    #denoising level

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations
    vps = []

    #push!(vps, (GetSAENetwork, ChangeEncodingActivation, (LinearActivation, primary_activation)))
    #push!(vps, (GetSAENetwork, ChangeOutputActivation, (LinearActivation, primary_activation)))
    #push!(vps, (GetSAETraining, ChangeIsDenoising, (true, false)))
    #push!(vps, (GetSAETraining, ChangeDenoisingVariance, (0.1, 0.01, 0.001, 0.0001, 0.00000000001)))
    #push!(vps, (GetSAETraining, ChangeL1Reg, (0.0001, 0.001, 0.01, 0.1, 1.0)))

    push!(vps, (GetSAETraining, ChangeMaxLearningRate, learning_rates))
    push!(vps, (GetSAETraining, ChangeLearningRateCycle, (100, 300)))
    push!(vps, (GetDataConfig, ChangeDeltas, ([1,5,20], [5,20,60], [10,20,60])))
    push!(vps, (GetSAENetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit, DCUniformInit)))


    set_name = string("Iteration4_1 SAE Tests ", string(layer_sizes), "x", encoding_layer, " ", split(string(primary_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))

    ################################################################################
    ##2a. Run Each SAE Configuration
    #jsedata = ReadJSETop40Data()
    exp_data =  nothing#jsedata[:, [1, 4, 5, 7, 8, 9, 10, 11, 12, 14]] #jsedata[:, [:AGL, :ACL]]#nothing

    sae_results = map(ep -> RunSAEConfigurationTest(ep, exp_data), combos)
    config_ids = map(x -> x[1], sae_results)

    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    prefix = string(string(layer_sizes), " Layers ")
    PlotSAERecontructions(sae_results, string(set_name, "Recons ", prefix))
    #PlotEpochs(config_ids, string(set_name, "Epochs ", prefix))
    #PlotGradientChangesCombined(sae_results, 5, string(set_name,"Combined Gradients ", prefix))
    #PlotActivations(sae_results, string(set_name, "Activations ", prefix))

    return sae_results
end



activation_function = LeakyReluActivation
layer_sizes = (90, 60)
learning_rates = (0.005, 0.01, 0.05, 0.1)

#RunNLayerReLUSAETest(25, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(20, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(15, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(10, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(5,  layer_sizes, activation_function, learning_rates)
