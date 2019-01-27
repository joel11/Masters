module SAEConfigTests

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


export RunSAEPreTrainingTest


function RunSAEPreTrainingTest(encoding_layer, layer_size)

    srand(1234567891234567)

    function GenerateBaseSAEConfig(set_name, datasetname)
        seed = abs(Int64.(floor(randn()*100)))
        ds = abs(Int64.(floor(randn()*100)))

        all_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
        var_pairs =  all_pairs[1:end]

        data_config = DatasetConfig(ds, datasetname,  5000,  [1],  [0.6],  [0.8, 1.0],  [1], var_pairs)

        input_size =  (length(var_pairs)*length(data_config.deltas))
        output_size = (length(var_pairs)*length(data_config.prediction_steps))

        layers = [6, layer_size, layer_size, encoding_layer]
        activations = [SigmoidActivation, SigmoidActivation,  SigmoidActivation]

        sae_net_par = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit)
        sae_sgd_par = TrainingParameters("SAE", 3.0, 1.0, 300,  30, 0.0, 500, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

        rbm_pretraining = true
        rbm_cd = TrainingParameters("RBM-CD", 3.0, 3.0, 1, 30, 0.0, 200, (0.0001, 50), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())


        return ExperimentConfig(seed, set_name, rbm_pretraining, data_config, sae_net_par, nothing , sae_sgd_par, nothing, rbm_cd, nothing, true)
    end

    ################################################################################
    ##1. Configuration Variations

    vps = []

    push!(vps, (GetSAETraining, ChangeMaxLearningRate, (0.5, 1.0, 2.0, 4.0)))
    push!(vps, (GetRBMTraining, ChangeMaxLearningRate, (0.5, 1.0, 2.0, 4.0)))
    push!(vps, (GetRBMTraining, ChangeMaxEpochs, (1, 5, 15)))
    push!(vps, (GetRBMTraining, ChangeMinibatchSize, (10, 30)))

    set_name = string("Sigmoid ", layer_size, "x", layer_size, "x", encoding_layer)
    combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig(set_name, "JSETop40_1_2"))
    ################################################################################
    ##2a. Run Each SAE Configuration

    jsedata = ReadJSETop40Data()
    exp_data = jsedata[:, [:ACL, :AGL, :AMS, :CRH, :CFR , :SOL]]
    sae_results = map(ep -> RunSAEConfigurationTest(ep, exp_data), combos)
    config_ids = map(x -> x[1], sae_results)


    PlotSAERecontructions(sae_results, string("Recons ", set_name))
    PlotEpochs(config_ids, string("Epochs ", set_name))
    PlotGradientChangesCombined(sae_results, 5, string("Combined Gradients ", set_name))
end

end
