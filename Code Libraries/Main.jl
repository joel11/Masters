workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
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

##Experiment Process############################################################

#0. Create base config
#1. Create variations of base config that we want to test
#2. For each config variation
#   a. Record all config
#   b. Prepare data accordingly
#   c. Run training, and record all epochs
#   d. Record actual & prediction values of the stocks for the test section
#3. CSCV
#   a. Select the N configuration ids, and the actual & predicted values
#   b. Use trading strategy to generate M matrix
#   c. Run CSCV & PBO

################################################################################
##0. Base Configuration

function GenerateBaseSAEConfig()
    seed = abs(Int64.(floor(randn()*100)))
    ds = abs(Int64.(floor(randn()*100)))

    all_pairs = ((0.9, 0.15), (0.9, 0.4), (0.9, 0.25), (-0.9, 0.15), (-0.9, 0.4), (-0.9, 0.25), (0.2, 0.09), (0.2, 0.1), (0.2, 0.15))#bull; bear; stable
    var_pairs = (all_pairs[1], all_pairs[4])

    data_config = DatasetConfig(ds, "synthetic",  5000,  [1, 7, 30],  [0.6, 0.8],  [0.8, 1.0],  [7], var_pairs)
    sae_net_par = NetworkParameters("SAE", [6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    sae_sgd_par = TrainingParameters("SAE", 0.005, 10, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    return ExperimentConfig(seed, "Null Name", data_config, sae_net_par, nothing , sae_sgd_par, nothing, nothing, nothing)
end

function GenerateBaseExperimentConfig()

    seed = abs(Int64.(floor(randn()*100)))
    ds = abs(Int64.(floor(randn()*100)))

    all_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
    var_pairs =  all_pairs

    data_config = DatasetConfig(ds, "synthetic",  5000,  [1, 3, 7],  [0.6],  [0.8, 1.0],  [2], var_pairs)

    input_size =  (length(var_pairs)*length(data_config.deltas))
    output_size = (length(var_pairs)*length(data_config.prediction_steps))
    encoding_layer = 4

    sae_net_par = NetworkParameters("SAE", [input_size, 20, 20, encoding_layer],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    ffn_net_par = NetworkParameters("FFN", [encoding_layer, 40, 40, output_size] ,[ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.XavierGlorotNormalInit)

    rbm_cd = TrainingParameters("RBM-CD", 0.1, 30, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    sae_sgd_par = TrainingParameters("SAE", 0.1, 30, 0.0, 20, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_sgd_par = TrainingParameters("SGD", 0.01, 30, 0.0, 20, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    ogd_par = TrainingParameters("OGD", 0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    #holdout_ogd_par = TrainingParameters("OGD-HO",0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    rbm_pretraining = false

    return ExperimentConfig(seed, "Null Name", rbm_pretraining, data_config, sae_net_par, ffn_net_par, sae_sgd_par, ffn_sgd_par, rbm_cd, ogd_par)#, holdout_ogd_par)
end

base_config = GenerateBaseExperimentConfig()
#sae_config = GenerateBaseSAEConfig()

################################################################################
##1. Configuration Variations
vps = []
#push!(vps, (GetFFNTraining, ChangeLearningRate, (0.01)))
push!(vps, (GetSAETraining, ChangeLearningRate, (0.01, 0.2)))
#push!(vps, (GetSAETraining, ChangeL1Reg, (0.0, 0.1)))

set_name = "testset"
base_config.experiment_set_name = set_name
combos = GenerateGridBasedParameterSets(vps, base_config)

################################################################################
##2a. Run Each SAE Configuration

config_ids = map(ep -> RunSAEConfigurationTest(ep), combos)
println(config_ids)

################################################################################
##2b. Run Each Configuration
#config_ids = map(ep -> RunConfigurationTest(ep), combos)
#println(config_ids)

################################################################################
##3. Plot Results

using ExperimentGraphs
PlotResults(config_ids, "testresults")
PlotEpochs(config_ids, "testepochs")



################################################################################
################################################################################
################################################################################
