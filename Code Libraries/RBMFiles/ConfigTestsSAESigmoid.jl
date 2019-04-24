#module SAEConfigTests

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


export RunSAEPreTrainingTest


function RunSAEPreTrainingTest(encoding_layer, layer_size, num_hidden)

    srand(1234567891234567)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))
        var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        data_config = DatasetConfig(ds, datasetname,  5000,  [1, 5, 20],  [0.6],  [0.8, 1.0],  [2], var_pairs, NormalizeData)

        #layers = [(length(var_pairs)*length(data_config.deltas))]
        layers = [2*length(data_config.deltas)]
        for i in 1:num_hidden
            push!(layers, layer_size)
        end
        push!(layers, encoding_layer)

        activations = map(x -> SigmoidActivation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit, SigmoidActivation)
        sae_sgd_par = TrainingParameters("SAE", 2.0, 1.0, 1, 20, 0.0, 1000, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])

        return SAEExperimentConfig(seed, set_name, false, data_config, sae_net_par, sae_sgd_par, nothing)
    end

    ################################################################################
    ##1. Configuration Variations

    vps = []

    push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.0000001, 0.00001, 0.001, 0.1, 1.0, 2.0))) #, 4.0))) #, 6.0)))
    push!(vps, (GetSAENetwork, ChangeOutputActivation, (SigmoidActivation, LinearActivation)))

    set_name =  string("SAE Sigmoid Output Activation Test 4 ", num_hidden, "x", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    jsedata = ReadJSETop40Data()
    exp_data = jsedata[:, [:ACL, :AGL]] #[:ACL, :AGL, :AMS, :CRH, :CFR , :SOL]]
    sae_results = map(ep -> RunSAEConfigurationTest(ep, exp_data), combos)
    #sae_results = map(ep -> RunSAEConfigurationTest(ep, nothing), combos)
    config_ids = map(x -> x[1], sae_results)


    for i in 1:length(config_ids)
        WriteSAE(config_ids[i], combos[i], sae_results[i][6])
    end

    prefix = string(num_hidden, " Layer Sigmoid ")
    PlotSAERecontructions(sae_results, string(prefix, "Recons ", set_name))
    PlotEpochs(config_ids, string(prefix, "Epochs ", set_name))
    PlotGradientChangesCombined(sae_results, 5, string(prefix,"Combined Gradients ", set_name))

    return sae_results
end


RunSAEPreTrainingTest(5, 15, 1)
RunSAEPreTrainingTest(5, 15, 2)
