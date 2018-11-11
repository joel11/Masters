workspace()
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


srand(9879)
network_parameters = NetworkParameters([784, 30, 10], [SigmoidActivation, SigmoidActivation], InitializationFunctions.HeNormalInit)
rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, true, true, 0.0, 0.0, MeanSquaredError())
ffn_parameters = TrainingParameters(0.1, 10, 0.0, 30, NonStopping, true, true, 0.0, 1.0, CrossEntropyError())
#network, rbm_records, ffn_records = TrainEncoderRBNMFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
#prediction_acc = PredictionAccuracy(network, dataset)
#9787
#5.0 9778

using HyperparameterOptimization
vals = 0.0:0.1:5.0
results = HyperparameterRangeSearch(dataset, network_parameters, rbm_parameters, ffn_parameters, ChangeL2Reg, vals)
GraphHyperparameterResults(results[1:(length(vals))], "/Users/joeldacosta/Desktop/", "lr2reg", "lr")


#############################################################################################

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
