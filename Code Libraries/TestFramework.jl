push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using Revise
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN


using MNIST
trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()
scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255

parameters = TrainingParameters(0.1, 30, 0.0,  3, 4, NonStopping)# GenValidationChangeReached(0.2))
training_data = scaled_training_data
validation_data = scaled_validation_data
initialization = InitializationFunctions.XavierGlorotUniformInit

#layer_sizes = [784, 400,200,100,50,25,6]
layer_sizes = [784, 400, 100, 10]
cost_function = MeanSquaredError



##RBM layer####################################################################

#parameters = TrainingParameters(0.1, 30, 0.0,  15, 4, NonStopping)# GenValidationChangeReached(0.2))
#layer = CreateRBMLayer(784, 500, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(scaled_training_data, scaled_validation_data, layer, parameters)
#test_set = validation_data[1:10,:]
#recon = ReconstructVisible(layer, test_set)


################################################################################



network, rbm_records, ffn_records =
TrainFFNNetwork(scaled_training_data, scaled_validation_data, layer_sizes, initialization, parameters, cost_function)
#TrainAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, initialization, parameters, cost_function)

##Full Graphs Output###########################################################
#using OutputLibrary
#reload("OutputLibrary")
#output_dir = "/Users/joeldacosta/Desktop/plots/mnist_2/"
#WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)
