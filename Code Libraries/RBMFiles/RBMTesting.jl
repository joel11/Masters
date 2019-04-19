workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using MNIST
using TrainingStructures
export GenerateData
using RBM
using NeuralNetworks
using ActivationFunctions
using InitializationFunctions
using FunctionsStopping
using CostFunctions
using Plots
using DataProcessor
using SGD

plotlyjs()

function GenerateData()

    trainingset, trainingsetlabels = traindata()

    trainingdata = trainingset[:, 1:50000]
    traininglabels = trainingsetlabels[1:50000]

    testingdata = trainingset[:, 50001:60000]
    testinglabels = trainingsetlabels[50001:60000]

    validationdata, validationlabels = testdata()

    training_labels = fill(0.0, (10, length(traininglabels)))
    testing_labels = fill(0.0, (10, length(testinglabels)))
    validation_labels = fill(0.0, (10, length(validationlabels)))


    for i in 1:length(traininglabels)
        training_labels[Int64.(traininglabels[i])+1, i] = 1
    end

    for i in 1:length(testinglabels)
        testing_labels[Int64.(testinglabels[i])+1, i] = 1
    end

    for i in 1:length(validationlabels)
        validation_labels[Int64.(validationlabels[i])+1, i] = 1
    end

    scaled_training_data = (trainingdata')./255
    scaled_testing_data = (testingdata')/255
    scaled_validation_data = (validationdata')./255


    return (DataSet(nothing, scaled_training_data, scaled_testing_data, training_labels', testing_labels', nothing, nothing, nothing, nothing))
end


function PlotRBMInputOutput(rbm_records, validation, number_samples, file_name)
    function get_plot(data)
        return(heatmap(reshape(data, (28,28))))
    end

    samples = validation[rand(1:size(validation)[1], number_samples), :]
    biased_training_data = hcat(fill(1.0, size(samples,1)), Array(samples))

    layer = rbm_records[end].network.layers[1]
    output = ReconstructVisible(layer, biased_training_data)

    pairs = map(i -> (hcat(biased_training_data[i,2:end], output[i,2:end])), 1:number_samples)
    combos = reduce(hcat, pairs)
    plots = map(x -> get_plot(combos[:,x]), 1:size(combos)[2])
    savefig(plot(plots..., size = (800, 800)),  file_name)
end

function AddDecoder(network::NeuralNetwork, network_parameters::NetworkParameters)
    function ReverseLayer(layer::NetworkLayer, initialization)
        unbiased_weights_t = WeightsWithoutBias(layer)'
        bias_weights = initialization(1, size(unbiased_weights_t)[2])
        new_weights = vcat(bias_weights, unbiased_weights_t)
        return NetworkLayer(new_weights, layer.activation)
    end

    for i in length(network.layers):-1:1
        new_layer = ReverseLayer(network.layers[i], network_parameters.initialization)
        AddLayer(network, new_layer)
    end

    #network.layers[length(network_parameters.layer_activations)].activation = network_parameters.layer_activations[end]
end

function CreateEncoderDataset(dataset::DataSet)
    training_input = dataset.training_input
    testing_input = dataset.testing_input

    return DataSet(dataset.original_prices, training_input, testing_input, training_input, testing_input, nothing, nothing, nothing, nothing)
end


mnist_data = GenerateData()

#layer = NetworkLayer(784, 100, SigmoidActivation, XavierGlorotUniformInit)
config_id = 999999
category = "RBM Testing"
training_data = mnist_data.training_input
#PlotRBMInputOutput(records, mnist_data.testing_input, 20, "/users/joeldacosta/desktop/rbmoutputs.html")

encoder_data = CreateEncoderDataset(mnist_data)
layers = [784, 200, 50]
activations = map(x -> SigmoidActivation, 1:(length(layers)))
network_parameters = NetworkParameters("SAE", layers, activations, InitializationFunctions.XavierGlorotNormalInit, SigmoidActivation)
ffn_parameters = TrainingParameters("SAE", 0.5, 1.0, 100, 20, 0.0, 10, (0.0001, 100), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])
rbm_parameters = TrainingParameters("RBM-CD", 0.5, 0.0, 1, 20, 0.0, 2, (0.0001, 50), NonStopping, true, false, 0.0, 0.0, MeanSquaredError(), [0.8])

rbm_network, rbm_records = RBM.TrainRBMNetwork(config_id,encoder_data, network_parameters, rbm_parameters)
#rbm_network = NeuralNetwork(network_parameters.layer_sizes, network_parameters.layer_activations, network_parameters.initialization)

AddDecoder(rbm_network, network_parameters)

softmax = NetworkLayer(784, 10, SigmoidActivation, XavierGlorotUniformInit)
##TODO train this layer via RBM first?
AddLayer(rbm_network, softmax)


sgd_records = RunSGD(config_id, category, mnist_data, rbm_network, ffn_parameters)
