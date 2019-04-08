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
        data_config = DatasetConfig(ds, datasetname,  5000,  [1, 7],  [0.6],  [0.8, 1.0],  [2], var_pairs, LimitedNormalizeData)

        layers = [(length(var_pairs)*length(data_config.deltas))]
        #layers = [1*length(data_config.deltas)]
        for i in 1:num_hidden
            push!(layers, layer_size)
        end
        push!(layers, encoding_layer)

        activations = map(x -> SigmoidActivation, 1:(length(layers)-1))

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit, LinearActivation)
        sae_sgd_par = TrainingParameters("SAE", 2.0, 1.0, 100, 20, 0.0, 300, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])

        rbm_cd = TrainingParameters("RBM-CD", 0.5, 3.0, 1, 20, 0.0, 1, (0.0001, 50), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])

        return SAEExperimentConfig(seed, set_name, true, data_config, sae_net_par, sae_sgd_par, rbm_cd)
    end

    ################################################################################
    ##1. Configuration Variations

    vps = []

    #push!(vps, (GetSAETraining, ChangeMaxEpochs, (100, 1000)))
    #push!(vps, (GetDataConfig, ChangeScalingFunction, (NormalizeData, StandardizeData)))
    #push!(vps, (GetSAETraining, ChangeMinibatchSize, (10, 30)))

    push!(vps, (GetDataConfig, ChangeScalingFunction, (NormalizeData, LimitedNormalizeData)))

    #push!(vps, (GetSAENetwork, ChangeInit, (InitializationFunctions.XavierGlorotNormalInit, InitializationFunctions.HeNormalInit, InitializationFunctions.HintonUniformInit)))

    push!(vps, (GetSAETraining, ChangeMinLearningRate, (2.0, 1.0, 0.5)))

    push!(vps, (GetRBMTraining, ChangeMaxLearningRate, (0.5, 1.0))) #, 2.0)))
    push!(vps, (GetRBMTraining, ChangeMaxEpochs, (0, 1, 10, 50))) # , 15)))

    set_name =  string("Sigmoid Test Set ", layer_size, "x", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "Synthetic"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    jsedata = ReadJSETop40Data()
    exp_data = nothing#jsedata[:, [:AGL]]#[:ACL, :AGL, :AMS, :CRH, :CFR , :SOL]]
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


RunSAEPreTrainingTest(9, 20, 1)
RunSAEPreTrainingTest(9, 40, 1)
RunSAEPreTrainingTest(9, 20, 2)
RunSAEPreTrainingTest(9, 40, 2)

RunSAEPreTrainingTest(6, 20, 1)
RunSAEPreTrainingTest(6, 40, 1)
RunSAEPreTrainingTest(6, 20, 2)
RunSAEPreTrainingTest(6, 40, 2)

RunSAEPreTrainingTest(3, 20, 1)
RunSAEPreTrainingTest(3, 40, 1)
RunSAEPreTrainingTest(3, 20, 2)
RunSAEPreTrainingTest(3, 40, 2)

#end
