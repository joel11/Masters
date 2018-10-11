module NetworkTrainer

using RBM, NeuralNetworks, ActivationFunctions, InitializationFunctions, TrainingStructures, SGD, CostFunctions

export TrainAutoEncoder, TrainFFNNetwork

function TrainFFNNetwork(dataset::DataSet, layer_sizes::Array{Int64}, layer_functions, initialization::Function, parameters::TrainingParameters, cost_function)
    activation_functions = GenerateActivationFunctions(length(layer_sizes))
    rbm_network, rbm_records = TrainRBMNetwork(dataset, layer_sizes, activation_functions, initialization, parameters)

    ApplyActivationFunctions(rbm_network, layer_functions)

    sgd_records = RunSGD(dataset, rbm_network, parameters, cost_function)

    return (rbm_network, rbm_records, sgd_records)
end

function TrainAutoEncoder(dataset::DataSet, layer_sizes::Array{Int64}, layer_functions, initialization::Function, parameters::TrainingParameters, cost_function)
    activation_functions = GenerateActivationFunctions(length(layer_sizes))
    rbm_network, rbm_records = TrainRBMNetwork(dataset, layer_sizes, activation_functions, initialization, parameters)

    AddDecoder(rbm_network, initialization)
    ApplyActivationFunctions(rbm_network, layer_functions)

    rbm_network.layers[length(layer_functions)].activation = layer_functions[end]
    sgd_records = RunSGD(dataset, rbm_network, parameters, cost_function)
    autoencoder = rbm_network#GetAutoencoder(rbm_network)
    return (autoencoder, rbm_records, sgd_records)
end

function ApplyActivationFunctions(rbm_network, layer_functions)
    for l in 1:length(layer_functions)
        rbm_network.layers[l].activation = layer_functions[l]
    end
end

function GenerateActivationFunctions(number_layers)
    activation_functions = Array{Function,1}()
    for in in 1:number_layers
        push!(activation_functions, SigmoidActivation)
    end
    return (activation_functions)
end

function ReverseLayer(layer::NetworkLayer, initialization)
    unbiased_weights_t = WeightsWithoutBias(layer)'
    bias_weights = initialization(1, size(unbiased_weights_t)[2])
    new_weights = vcat(bias_weights, unbiased_weights_t)
    return NetworkLayer(new_weights, layer.activation)
end

function AddDecoder(network::NeuralNetwork, initialization::Function)
    for i in length(network.layers):-1:1
        new_layer = ReverseLayer(network.layers[i], initialization)
        AddLayer(network, new_layer)
    end
end

function GetAutoencoder(network::NeuralNetwork)
    layers = network.layers[1:(Int64.(length(network.layers)/2))]
    return (NeuralNetwork(layers))
end

end
