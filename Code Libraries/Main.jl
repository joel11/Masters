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
using DataJSETop40

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

function GenerateBaseSAEConfig(set_name, datasetname)
    seed = abs(Int64.(floor(randn()*100)))
    ds = abs(Int64.(floor(randn()*100)))

    all_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
    var_pairs =  all_pairs[1:end]

    data_config = DatasetConfig(ds, datasetname,  5000,  [2],  [0.6],  [0.8, 1.0],  [2], var_pairs)

    input_size =  (length(var_pairs)*length(data_config.deltas))
    output_size = (length(var_pairs)*length(data_config.prediction_steps))
    encoding_layer = 4

    sae_net_par = NetworkParameters("SAE", [input_size, 10,  encoding_layer],[ReluActivation,  LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
    sae_sgd_par = TrainingParameters("SAE", 0.005, 10, 0.0, 3000, (0.0001, 50), NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

    rbm_pretraining = false
    rbm_cd = TrainingParameters("RBM-CD", 0.1, 10, 0.0, 1, (0.0001, 50), GenValidationChangeReached, true, false, 0.0, 0.0, MeanSquaredError())


    return ExperimentConfig(seed, set_name, rbm_pretraining, data_config, sae_net_par, nothing , sae_sgd_par, nothing, rbm_cd, nothing, true)
end

################################################################################
##1. Configuration Variations
##Structures
input = 6
encoding = 4
layers =   (("8 - ReLU", [input, 8, encoding], [ReluActivation,  LinearActivation]),
            ("15 - ReLU", [input, 15, encoding], [ReluActivation,  LinearActivation]),
            ("30 - ReLU", [input, 30, encoding], [ReluActivation,  LinearActivation]),
            ("8x8 - ReLU", [input, 8, 8, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            ("15x15 - ReLU", [input, 15, 15, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            ("25x25 - ReLU", [input, 25, 25, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            ("40x40 - ReLU", [input, 40, 40, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            ("8x8x8 - ReLU", [input, 8, 8, 8, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation])
            #("8 - Sigmoid", [input, 8, encoding], [SigmoidActivation,  LinearActivation]),
            #("8x8 - Sigmoid", [input, 8, 8, encoding], [SigmoidActivation, SigmoidActivation,  LinearActivation]),
            #("8x8x8 - Sigmoid", [input, 8, 8, 8, encoding], [SigmoidActivation, SigmoidActivation, SigmoidActivation, LinearActivation]),
            #("15 - Sigmoid", [input, 15, encoding], [SigmoidActivation,  LinearActivation]),
            #("15x15 - Sigmoid", [input, 15, 15, encoding], [SigmoidActivation, SigmoidActivation,  LinearActivation])
            )

vps = []
push!(vps, (GetSAETraining, ChangeLearningRate, (0.1)))#, 0.05)))
push!(vps, (GetSAENetwork, ChangeLayers, layers))

combos = GenerateGridBasedParameterSets(vps, GenerateBaseSAEConfig("Xavier Init Plus Test_14", "JSETop40_1_2"))


################################################################################
##2a. Run Each SAE Configuration
jsedata = ReadJSETop40Data()
exp_data = jsedata[:, [:ACL, :AGL, :AMS, :CRH, :CFR , :SOL]]
sae_results = map(ep -> RunSAEConfigurationTest(ep, exp_data), combos)
config_ids = map(x -> x[1], sae_results)


using ExperimentGraphs
PlotSAERecontructions(sae_results, "Xavier Init_14")
PlotEpochs(config_ids, "Xavier Epochs_14")
