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

    data_config = DatasetConfig(ds, "synthetic",  5500,  [1, 7, 30],  [0.6, 0.8],  [0.8, 1.0],  [7], var_pairs)
    sae_net_par = NetworkParameters("SAE", [6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    sae_sgd_par = TrainingParameters("SAE", 0.005, 10, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    return ExperimentConfig(seed, "Null Name", data_config, sae_net_par, nothing , sae_sgd_par, nothing, nothing, nothing)
end

function GenerateBaseExperimentConfig()

    seed = abs(Int64.(floor(randn()*100)))
    ds = abs(Int64.(floor(randn()*100)))

    all_pairs = ((0.9, 0.15), (0.9, 0.4), (0.9, 0.25), (-0.9, 0.15), (-0.9, 0.4), (-0.9, 0.25), (0.2, 0.09), (0.2, 0.1), (0.2, 0.15))#bull; bear; stable
    var_pairs = (all_pairs[1], all_pairs[4])

    data_config = DatasetConfig(ds, "synthetic",  5500,  [1, 7, 30],  [0.6, 0.8],  [0.8, 1.0],  [7], var_pairs)

    sae_net_par = NetworkParameters("SAE", [6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    ffn_net_par = NetworkParameters("FFN", [4, 40, 40, 40, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.XavierGlorotNormalInit)

    rbm_cd = TrainingParameters("RBM-CD", 0.1, 10, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    sae_sgd_par = TrainingParameters("SAE", 0.005, 10, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_sgd_par = TrainingParameters("SGD", 0.00005, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ogd_par = TrainingParameters("OGD", 0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    holdout_ogd_par = TrainingParameters("OGD-HO",0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    return ExperimentConfig(seed, "Null Name", true, data_config, sae_net_par, ffn_net_par, sae_sgd_par, ffn_sgd_par, rbm_cd, ogd_par, holdout_ogd_par)
end

base_config = GenerateBaseExperimentConfig()
#sae_config = GenerateBaseSAEConfig()

################################################################################
##1. Configuration Variations
vps = []
push!(vps, (GetFFNTraining, ChangeMinibatchSize, (10, 30)))
#push!(vps, (GetSAETraining, ChangeLearningRate, (0.05, 0.1)))
#push!(vps, (GetSAETraining, ChangeL1Reg, (0.0, 0.1)))

base_config.experiment_set_name = "Second Run Experiment"
combos = GenerateGridBasedParameterSets(vps, base_config)

combos[1].seed = 6
combos[2].seed = 6
combos[1].data_config.data_seed = 192
combos[2].data_config.data_seed = 192

################################################################################
##2. Run Each Configuration
for ep in combos
    RunConfigurationTest(ep)
end

################################################################################
##3. CSCV
#ExperimentCSCVProcess()
