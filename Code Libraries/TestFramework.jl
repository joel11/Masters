push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using RBM, NeuralNetworks, ActivationFunctions, InitializationFunctions, AutoEncoder, TrainingStructures, SGD, CostFunctions, StoppingFunctions, FFN
using OutputLibrary

using MNIST
trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()
scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255

parameters = TrainingParameters(0.1, 30, 0.0,  3, 10, NonStopping)# GenValidationChangeReached(0.2))
training_data = scaled_training_data
validation_data = scaled_validation_data
initialization = InitializationFunctions.XavierGlorotUniformInit

layer_sizes = [784, 400,200,100,50,25,6]
cost_function = MeanSquaredError

network, rbm_records, ffn_records =
CreateAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, initialization, parameters, cost_function)

##Full Graphs Output###########################################################

output_dir = "/Users/joeldacosta/Desktop/plots/mnist_1/"
WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)
