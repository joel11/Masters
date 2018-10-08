push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
srand(1234)
using Revise
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN


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
initialization = InitializationFunctions.XavierGlorotUniformInit

#layer_sizes = [784, 400,200,100,50,25,6]

function PredictionAccuracy(network, validation_input, validation_labels)
    validation_pred = Feedforward(network, validation_input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == validation_labels[i,:], 1:size(validation_labels)[1])))
    return(correct_predictions)
end

##RBM layer####################################################################

#parameters = TrainingParameters(0.1, 30, 0.0,  15, 4, NonStopping)# GenValidationChangeReached(0.2))
#layer = CreateRBMLayer(784, 500, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(scaled_training_data, scaled_validation_data, layer, parameters)
#test_set = validation_data[1:10,:]
#recon = ReconstructVisible(layer, test_set)


################################################################################
#layer_sizes = [784, 500, 300, 100, 30, 10]
#layer_functions = [SigmoidActivation, SigmoidActivation,  SigmoidActivation,  SigmoidActivation,  SigmoidActivation]
#parameters = TrainingParameters(0.1, 30, 0.0,  2, 10, NonStopping)# GenValidationChangeReached(0.2))
#cost_function = CategoricalCrossEntropyError() 9728
#cost_function = MeanSquaredError() 3769

#9697
#cost_function = CategoricalCrossEntropyError()
#layer_sizes = [784, 500, 300, 100, 30, 10]
#layer_functions = [SigmoidActivation, SigmoidActivation,  SigmoidActivation,  SigmoidActivation,  SoftmaxActivation]
#parameters = TrainingParameters(0.1, 30, 0.0,  2, 10, NonStopping)# GenValidationChangeReached(0.2))

#5582
#cost_function = MeanSquaredError()
#layer_sizes = [784, 100, 30, 10]
#layer_functions = [SigmoidActivation,  SigmoidActivation,  SigmoidActivation]
#parameters = TrainingParameters(0.1, 30, 0.0,  2, 10, NonStopping)# GenValidationChangeReached(0.2))

#2.0 - 8310
#2.5 - 9230
srand(1080)
cost_function = MeanSquaredError()
layer_sizes = [784, 100,  10]
layer_functions = [SigmoidActivation,    SigmoidActivation]
parameters = TrainingParameters(2.3, 30, 0.0,  1, 10, NonStopping)# GenValidationChangeReached(0.2))

network, rbm_records, ffn_records =
TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

PredictionAccuracy(network, validation_data, validation_labels')

#e1network = ffn_records[1].network
#validation_estimations = Feedforward(e1network, scaled_validation_data)[end]
#oos_error = cost_function.CalculateCost(validation_output, validation_estimations)



#TrainAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, layer_functions, initialization, parameters, cost_function)



#


##Full Graphs Output###########################################################
#using OutputLibrary
#reload("OutputLibrary")
#output_dir = "/Users/joeldacosta/Desktop/plots/mnist_2/"
#WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)
