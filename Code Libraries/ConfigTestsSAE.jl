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

function RunNLayerReLUSAETest(encoding_layer, layer_size, num_hidden, primary_activation, learning_rates)

    srand(2)

    function GenerateBaseSAEConfig(set_name, datasetname)
        srand(2)

        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,
                                    5000,  #timesteps
                                    [1, 5, 20], #delta aggregatios
                                    [0.6], #process split (for SAE/SGD & OGD)
                                    [0.8, 1.0], #validation set split
                                    [5], #prediction step
                                    var_pairs,
                                    LimitedNormalizeData) #scaling function

        layers = [(length(var_pairs)*length(data_config.deltas))]
        #layers = [10*length(data_config.deltas)]
        #layers = [12]
        for i in 1:num_hidden
            push!(layers, layer_size)
        end
        push!(layers, encoding_layer)

        activations = map(x -> primary_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations,
                                        InitializationFunctions.XavierGlorotUniformInit,
                                        LinearActivation, #output
                                        LinearActivation) #encoding

        sae_sgd_par = TrainingParameters("SAE",
                                        0.005,    #max learning rate
                                        0,#0.00001,        #min learning rate
                                        0,#100,        #epoch cycle max
                                        20,       #minibatch size
                                        500,      #max epochs
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
    #push!(vps, (GetSAENetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit)))
    #push!(vps, (GetSAETraining, ChangeIsDenoising, (true, false)))
    #push!(vps, (GetSAETraining, ChangeDenoisingVariance, (0.1, 0.01, 0.001, 0.0001, 0.00000000001)))
    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, learning_rates))
    #push!(vps, (GetSAETraining, ChangeDenoisingVariance, (0.01, 0.05, 0.1, 0.15, 0.2, 0.25)))
    #push!(vps, (GetSAETraining, ChangeL1Reg, (0.0001, 0.001, 0.01, 0.1, 1.0)))

    push!(vps, (GetSAETraining, ChangeTrainingSplits, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)))
    push!(vps, (GetSAETraining, ChangeMaxLearningRate, learning_rates))
    #push!(vps, (GetSAETraining, ChangeLearningRateCycle, (100, 200)))
    #push!(vps, (GetDataConfig, ChangeDeltas, ([10,20,60], [1,10,20])))
    push!(vps, (GetDataConfig, ChangeVariations, GenerateFourAssetConfig()))

    set_name = string("Iteration3_14 SAE Validation Perc Test ", num_hidden, "x", layer_size, "x", encoding_layer, " ", split(string(primary_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic Set"))

    ################################################################################
    ##2a. Run Each SAE Configuration
    jsedata = ReadJSETop40Data()
    exp_data =  nothing#jsedata[:, [1, 4, 5, 7, 8, 9, 10, 11, 12, 14]] #jsedata[:, [:AGL, :ACL]]#nothing

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

activation_function = LeakyReluActivation
learning_rates = (0.005, 0.01)

RunNLayerReLUSAETest(12, 40, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(12, 40, 2, activation_function, learning_rates)
RunNLayerReLUSAETest(12, 80, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(12, 80, 2, activation_function, learning_rates)

RunNLayerReLUSAETest(9, 40, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(9, 40, 2, activation_function, learning_rates)
RunNLayerReLUSAETest(9, 80, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(9, 80, 2, activation_function, learning_rates)

RunNLayerReLUSAETest(6, 40, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(6, 40, 2, activation_function, learning_rates)
RunNLayerReLUSAETest(6, 80, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(6, 80, 2, activation_function, learning_rates)

RunNLayerReLUSAETest(3, 40, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(3, 40, 2, activation_function, learning_rates)
RunNLayerReLUSAETest(3, 80, 1, activation_function, learning_rates)
RunNLayerReLUSAETest(3, 80, 2, activation_function, learning_rates)



#RunNLayerReLUSAETest(15, 40, 1, activation_function, learning_rates)
#RunNLayerReLUSAETest(15, 40, 2, activation_function, learning_rates)
#RunNLayerReLUSAETest(15, 80, 1, activation_function, learning_rates)
#RunNLayerReLUSAETest(15, 80, 2, activation_function, learning_rates)


#RunNLayerReLUSAETest(1, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(2, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(3, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(4, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(5, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(6, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(8, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(10, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(1, 40, 2, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(2, 40, 2, LeakyReluActivation, (0.05))

#RunNLayerReLUSAETest(3, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(4, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(5, 40, 1, LeakyReluActivation, (0.05))


#RunNLayerReLUSAETest(2, 15, 1, LeakyReluActivation, (0.001, 0.05, 0.01))

#RunNLayerReLUSAETest(1, 10, 1, LeakyReluActivation, (0.05, 0.01))
#RunNLayerReLUSAETest(2, 10, 1, LeakyReluActivation, (0.05, 0.01))
#RunNLayerReLUSAETest(1, 10, 2, LeakyReluActivation, (0.05, 0.01))
#RunNLayerReLUSAETest(2, 10, 2, LeakyReluActivation, (0.05, 0.01))

#RunNLayerReLUSAETest(1, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(2, 20, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(3, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(4, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(5, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(6, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(8, 40, 1, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(10, 40, 1, LeakyReluActivation, (0.05))

#RunNLayerReLUSAETest(1, 20, 2, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(2, 20, 2, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(3, 20, 2, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(4, 20, 2, LeakyReluActivation, (0.05))
#RunNLayerReLUSAETest(5, 20, 2, LeakyReluActivation, (0.05))

#Input: 30
#Sizes: 60, 120
#Layers:1, 2
#Encoding: 25, 15, 5
#=
activation_function = LeakyReluActivation

RunNLayerReLUSAETest(25, 60, 1,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(25, 60, 2,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(25, 120, 1, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(25, 120, 2, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))

RunNLayerReLUSAETest(15, 60, 1,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(15, 60, 2,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(15, 120, 1, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(15, 120, 2, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))

RunNLayerReLUSAETest(5, 60, 1,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(5, 60, 2,  activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(5, 120, 1, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))
RunNLayerReLUSAETest(5, 120, 2, activation_function,  (0.0001, 0.001, 0.01, 0.1, 1.0))

=#
