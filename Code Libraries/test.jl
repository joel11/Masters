
workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using DataJSETop40

jsedata = ReadJSETop40Data()
dataset =  jsedata[:, [:AGL ,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

data_config = DatasetConfig(2,
                            "tstset",
                            1,  #timesteps
                            #[1,5,20],
                            [5,20,60], #horizon aggregations
                            [0.6], #process split (for SAE/SGD & OGD)
                            [0.8, 1.0], #validation set split
                            [5], #prediction step
                            ((0,0)), #var pairs
                            LimitedNormalizeData) #scaling function

ffn_sgd_par = TrainingParameters("FFN",
                                0.01, #max_learning_rate
                                0.01, #min_learning_rate
                                100,  #epoch_cycle_max
                                32, #minibatch_sizes
                                1000, #max_epochs
                                (0.0001, 100), #stopping_parameters
                                NonStopping, #stopping_function
                                0.0, #l1_lambda
                                MeanSquaredError(), #cost_function
                                [1.0], #training_splits
                                false, #is_denoising #TODO SET TO FALSE
                                0.0) #denoising_variance



##Data Processing
saesgd_data, ogd_data = PrepareData(data_config, dataset, ffn_sgd_par)











saesgd_data2, ogd_data2 = PrepareData(data_config, dataset, ffn_sgd_par)
