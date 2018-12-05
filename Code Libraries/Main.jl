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

function GenerateBaseExperimentConfig()

    seed = abs(Int64.(floor(randn()*100)))
    ds = abs(Int64.(floor(randn()*100)))

    all_pairs = ((0.9, 0.15), (0.9, 0.4), (0.9, 0.25), (-0.9, 0.15), (-0.9, 0.4), (-0.9, 0.25), (0.2, 0.09), (0.2, 0.1), (0.2, 0.15))#bull; bear; stable
    var_pairs = (all_pairs[1], all_pairs[4])

    data_config = DatasetConfig(ds, "synthetic",  5500,  [1, 7, 30],  [0.6, 0.8],  [0.8, 1.0],  [7], var_pairs)

    sae_net_par = NetworkParameters("SAE", [6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    ffn_net_par = NetworkParameters("FFN", [4, 40, 40, 40, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.XavierGlorotNormalInit)

    sae_sgd_par = TrainingParameters("SAE", 0.005, 10, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_sgd_par = TrainingParameters("SGD", 0.00005, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ogd_par = TrainingParameters("OGD", 0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    holdout_ogd_par = TrainingParameters("OGD-HO",0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    return ExperimentConfig(seed, data_config, sae_net_par, ffn_net_par, sae_sgd_par, ffn_sgd_par, ogd_par, holdout_ogd_par)
end

base_config = GenerateBaseExperimentConfig()

################################################################################
##1. Configuration Variations
vps = []
push!(vps, (GetFFNTraining, ChangeMinibatchSize, (1:2)))
push!(vps, (GetSAETraining, ChangeLearningRate, (1, 0.1, 0.01)))
push!(vps, (GetSAETraining, ChangeL1Reg, (0.5, 0.2)))

combos = GenerateGridBasedParameterSets(vps, base_config)

################################################################################
##2. Run Each Configuration
for ep in combos
    RunConfigurationTest(ep)
end

################################################################################
##3. CSCV
#ExperimentCSCVProcess()
