module AutoEncoder

using RBM, NeuralNetworks, ActivationFunctions, InitializationFunctions, AutoEncoder, TrainingStructures

export CreateAutoEncoder

function GenerateActivationFunctions(number_layers)
    activation_functions = Array{Function,1}()
    for in in 1:(number_layers-1)
        push!(activation_functions, SigmoidActivation)
    end
    push!(activation_functions, SigmoidActivation)
    return (activation_functions)
end

function ReverseLayer(layer::NetworkLayer, initialization)
    unbiased_weights_t = WeightsWithoutBias(layer)'
    bias_weights = initialization(1, size(unbiased_weights_t)[2])
    new_weights = vcat(bias_weights, unbiased_weights_t)
    return NetworkLayer(new_weights, layer.activation)
end

function CreateAutoEncoder(training_data, validation_data, layer_sizes::Array{Int64}, initialization::Function, parameters::TrainingParameters)
    activation_functions = GenerateActivationFunctions(length(layer_sizes))
    rbm_network, epoch_records = TrainRBMNetwork(training_data, validation_data, layer_sizes, activation_functions, initialization, parameters)
    AddDecoder(rbm_network, initialization)
    TrainBackprop(rbm_network)
    autoencoder = rbm_network#GetAutoencoder(rbm_network)
    return (autoencoder)
end

function AddDecoder(network::NeuralNetwork, initialization::Function)
    for i in length(network.layers):-1:1
        new_layer = ReverseLayer(network.layers[i], initialization)
        AddLayer(network, new_layer)
    end
end

function TrainBackprop(network::NeuralNetwork)
    return (network)
end

function GetAutoencoder(network::NeuralNetwork)
    layers = network.layers[1:(Int64.(length(network.layers)/2))]
    return (NeuralNetwork(layers))
end

end
