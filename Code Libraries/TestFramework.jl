push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
srand(1234)
using Revise
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN
using AutomatedTests

using MNIST
trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()

training_labels = fill(0.0, (10, length(traininglabels)))
validation_labels = fill(0.0, (10, length(validationlabels)))

i = 1

for i in 1:length(traininglabels)
    training_labels[Int64.(traininglabels[i])+1, i] = 1
end

for i in 1:length(validationlabels)
    validation_labels[Int64.(validationlabels[i])+1, i] = 1
end



scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255


training_data = scaled_training_data
validation_data = scaled_validation_data


#layer_sizes = [784, 400,200,100,50,25,6]

##RBM layer####################################################################

#parameters = TrainingParameters(0.1, 30, 0.0,  15, 4, NonStopping)# GenValidationChangeReached(0.2))
#layer = CreateRBMLayer(784, 500, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(scaled_training_data, scaled_validation_data, layer, parameters)
#test_set = validation_data[1:10,:]
#recon = ReconstructVisible(layer, test_set)

#3069
srand(1080)
cost_function = MeanSquaredError()
layer_sizes = [784, 300, 100,  10]
layer_functions = [ReluActivation, ReluActivation, SigmoidActivation]
parameters = TrainingParameters(0.1, 30, 0.0,  2, 10, NonStopping, true )# GenValidationChangeReached(0.2))
initialization = InitializationFunctions.XavierGlorotUniformInit

network, rbm_records, ffn_records =
TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

PredictionAccuracy(network, validation_data, validation_labels')

#TrainAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, layer_functions, initialization, parameters, cost_function)


##Full Graphs Output###########################################################
#using OutputLibrary
#reload("OutputLibrary")
#output_dir = "/Users/joeldacosta/Desktop/plots/mnist_2/"
#WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)
