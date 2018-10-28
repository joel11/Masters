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
network_parameters = NetworkParameters([784, 100, 10], [ReluActivation, SigmoidActivation], InitializationFunctions.HeNormalInit)
rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
ffn_parameters = TrainingParameters(0.5, 10, 0.0, 30, NonStopping, false, true, 0.0, 0.0, CrossEntropyError())
network, rbm_records, ffn_records = TrainEncoderRBNMFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
prediction_acc = PredictionAccuracy(network, dataset)

Pkg.update()
#/Applications/JuliaPro-0.6.2.1.app/Contents/Resources/pkgs-0.6.2.1/v0.6
 - build the package(s) and all dependencies with `Pkg.build("Cairo", "Homebrew", "Gtk")`
 # - build a single package by running its `deps/build.jl` script


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
