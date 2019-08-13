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


function RunNLayerReLUSAETest(encoding_layer, layer_sizes, primary_activation)

    srand(2)

    function GenerateBaseSAEConfig(set_name, datasetname)
        srand(2)

        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))

        data_config = DatasetConfig(ds, datasetname,
                                    1,  #timesteps
                                    [1, 5, 20], #delta aggregatios
                                    [0.6], #process split (for SAE/SGD & OGD)
                                    [0.8, 1.0], #validation set split
                                    [5], #prediction step
                                    ((0,0)), #var pairs
                                    LimitedNormalizeData) #scaling function

        #layers = [(length(var_pairs)*length(data_config.deltas))]
        layers = [10*length(data_config.deltas)]

        for i in 1:length(layer_sizes)
            push!(layers, layer_sizes[i])
        end
        push!(layers, encoding_layer)

        activations = map(x -> primary_activation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations,
                                        InitializationFunctions.DCUniformInit,
                                        LinearActivation, #output
                                        LinearActivation) #encoding

        sae_sgd_par = TrainingParameters("SAE",
                                        0.01,    #max learning rate
                                        0.00001,        #min learning rate
                                        200,        #epoch cycle max
                                        32,       #minibatch size
                                        20000,      #max epochs
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


    #push!(vps, (GetSAENetwork, ChangeInit, (XavierGlorotUniformInit, HeUniformInit, DCUniformInit)))
    #push!(vps, (GetDataConfig, ChangeDeltas, ([1,5,20], [5,20,60], [10,20,60])))
    #push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.01, 0.1)))
    #push!(vps, (GetSAETraining, ChangeL1Reg, (0.5, 0.0)))

    #push!(vps, (GetSAETraining, ChangeL1Reg, (0.1, 0.0)))
    push!(vps, (GetSAETraining, ChangeL1Reg, (0.0)))
    push!(vps, (GetDataConfig, ChangeDeltas, ([1,5,20], [5,20,60], [10,20,60])))


    set_name = string("Iteration4 SAE Actual10 Test", string(layer_sizes), "x", encoding_layer, " ", split(string(primary_activation), ".")[2])
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Actual10 Set"))

    ################################################################################
    ##2a. Run Each SAE Configuration
    jsedata = ReadJSETop40Data()
    exp_data =  jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

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

layers = (240, 240, 240)
#RunNLayerReLUSAETest(25, layers, activation_function)
RunNLayerReLUSAETest(20, layers, activation_function)
RunNLayerReLUSAETest(15, layers, activation_function)
RunNLayerReLUSAETest(10, layers, activation_function)
RunNLayerReLUSAETest(5, layers, activation_function)




#layers = (240, 240, 240)


#TODO
#

#layers = (120, 90, 60)
#layers = (240)
#RunNLayerReLUSAETest(25, layers, activation_function)
#RunNLayerReLUSAETest(20, layers, activation_function)
#RunNLayerReLUSAETest(15, layers, activation_function)
#RunNLayerReLUSAETest(10, layers, activation_function)
#RunNLayerReLUSAETest(5, layers, activation_function)

#layers = (120, 120, 120, 120)
#RunNLayerReLUSAETest(25, layers, activation_function)
#RunNLayerReLUSAETest(20, layers, activation_function)
#RunNLayerReLUSAETest(15, layers, activation_function)
#RunNLayerReLUSAETest(10, layers, activation_function)
#RunNLayerReLUSAETest(5, layers, activation_function)

#layers = (240, 240)
#RunNLayerReLUSAETest(25, layers, activation_function)
#RunNLayerReLUSAETest(20, layers, activation_function)
#RunNLayerReLUSAETest(15, layers, activation_function)
#RunNLayerReLUSAETest(10, layers, activation_function)
#RunNLayerReLUSAETest(5, layers, activation_function)

#layers = (240, 240, 240)
#RunNLayerReLUSAETest(25, layers, activation_function)
#RunNLayerReLUSAETest(20, layers, activation_function)
#RunNLayerReLUSAETest(15, layers, activation_function)
#RunNLayerReLUSAETest(10, layers, activation_function)
#RunNLayerReLUSAETest(5, layers, activation_function)


#learning_rates = (0.005, 0.01, 0.05, 0.1)
#learning_rates = (0.01)

#RunNLayerReLUSAETest(2, (12,6), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (12), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (12,12), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (12,9), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (9,6), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (9,6,3), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (12,6,3), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (9,9,9), activation_function, learning_rates)
#RunNLayerReLUSAETest(2, (9,9), activation_function, learning_rates)

#RunNLayerReLUSAETest(1, (12,6), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (12), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (12,12), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (12,9), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (9,6), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (9,6,3), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (12,6,3), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (9,9,9), activation_function, learning_rates)
#RunNLayerReLUSAETest(1, (9,9), activation_function, learning_rates)

#RunNLayerReLUSAETest(25, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(20, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(15, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(10, layer_sizes, activation_function, learning_rates)
#RunNLayerReLUSAETest(5,  layer_sizes, activation_function, learning_rates)


#(12,6), (12), (12,12), (12,9), (9,6), (9,6,3),(12,6,3), (9,9,9), (9,9)
