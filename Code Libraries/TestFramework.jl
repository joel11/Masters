push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD

################################################################################

using MNISTData
using AutomatedTests
dataset = GenerateData()
#= Boiler plate
#AutomatedTests.RunTests(dataset)

#using OutputLibrary
#PlotRBMInputOutput(rbm_records, dataset.validation_input, 20, "/Users/joeldacosta/Desktop/")
#WriteFFNGraphs(ffn_records, "/Users/joeldacosta/Desktop/")
#output_dir = "/Users/joeldacosta/Desktop/plots/mnist_2/"
#WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)

#using HyperparameterOptimization
#vals = 0.1:0.5:3.1
#results = HyperparameterRangeSearch(dataset, network_parameters, rbm_parameters, ffn_parameters, ChangeLearningRate, vals)
#GraphHyperparameterResults(results, "/Users/joeldacosta/Desktop/", "lr_test_RlSgMSE", "lr") =#

##Pretraining Testing########################################################################
srand(1080)
cost_function = LoglikelihoodError()

network_parameters = NetworkParameters([784, 500, 500, 2000, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
    , InitializationFunctions.XavierGlorotUniformInit)
rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
prediction_acc = PredictionAccuracy(network, dataset)








#TrainAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, layer_functions, initialization, parameters, cost_function)

##RBM Testing ################################################################################
#parameters = TrainingParameters(0.1, 30, 0.0,  15, 0, NonStopping, true)
#layer = CreateRBMLayer(784, 500, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(dataset.training_input, dataset.validation_input, layer, parameters)
##test_set = dataset.validation_input[1:10,:]
##recon = ReconstructVisible(layer, test_set)
#############################################################################################
