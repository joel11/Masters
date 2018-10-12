push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using Revise
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN
using AutomatedTests
using MNISTData


dataset = GenerateData()
AutomatedTests.RunTests(dataset)

srand(1234)
layer_sizes = [784, 100, 10]
layer_functions = [SigmoidActivation, SigmoidActivation]
parameters = TrainingParameters(0.5, 10, 0.0,  0, 30, NonStopping, true, true, 0.0, 5.0)
cost_function = LoglikelihoodError()
initialization = InitializationFunctions.XavierGlorotUniformInit

network, rbm_records, ffn_records = TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)
prediction_acc = PredictionAccuracy(network, dataset)

#using OutputLibrary
#reload("OutputLibrary")
#WriteFFNGraphs(ffn_records, "/Users/joeldacosta/Desktop/")



##RBM Testing ################################################################################
#parameters = TrainingParameters(0.1, 30, 0.0,  15, 0, NonStopping, true)
#layer = CreateRBMLayer(784, 500, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(dataset.training_input, dataset.validation_input, layer, parameters)
##test_set = dataset.validation_input[1:10,:]
##recon = ReconstructVisible(layer, test_set)
#############################################################################################

##Full Graphs Output#########################################################################
#using OutputLibrary
#reload("OutputLibrary")
#output_dir = "/Users/joeldacosta/Desktop/plots/mnist_2/"
#WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)
#############################################################################################

##AutoEncoder Testing########################################################################
#srand(1080)
#layer_sizes = [784, 500, 500, 2000, 10]
#layer_functions = [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
#cost_function = LoglikelihoodError()

#parameters = TrainingParameters(0.1, 30, 0.0, 0, 20, NonStopping, true)
#initialization = InitializationFunctions.XavierGlorotUniformInit

#network, rbm_records, ffn_records = TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)
#PredictionAccuracy(network, dataset)

#RBM 3, epoch 10: 9779
#RBM 1, epoch 10: 9763
#RBM 0, epoch 10: 9368
#RBM 0, epoch 20: 9675

#TrainAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, layer_functions, initialization, parameters, cost_function)
