module NeuralNetworks

using ActivationFunctions, InitializationFunctions
export NetworkLayer, NeuralNetwork, InputSize, OutputSize, FirstLayer, LastLayer, InputSize, OutputSize, AddLayer, WeightsWithoutBias
export CopyLayer, CopyNetwork

type NetworkLayer
    weights::Array{Float64, 2}
    activation::Function

    function NetworkLayer(weights, activation)
        return new(weights, activation)
    end

    function NetworkLayer(input::Int64, output::Int64, activation::Function, weight_initialization::Function)
        return new(weight_initialization(input + 1, output), activation)
    end
end

function CopyLayer(network_layer::NetworkLayer)
    return NetworkLayer(copy(network_layer.weights), network_layer.activation)
end

type NeuralNetwork
    layers::Array{NetworkLayer}

    function NeuralNetwork(layers::Array{NetworkLayer})
        network = NeuralNetwork(layers[1])
        for i in 2:length(layers)
            AddLayer(network, layers[i])
        end
        return network
    end

    function NeuralNetwork(layer_sizes::Array{Int64}, activation::Function)
        return (NeuralNetwork(layer_sizes, activation, InitializationFunctions.HintonUniformInit))
    end

    function NeuralNetwork(layer_sizes::Array{Int64}, activation::Function, weight_initialization::Function)
        layers = Array{NetworkLayer}(0)
        for i in 1:(length(layer_sizes)-1)
            push!(layers, NetworkLayer(layer_sizes[i], layer_sizes[i+1], activation,weight_initialization))
        end
        return (NeuralNetwork(layers))
    end

    function NeuralNetwork(layer_sizes::Array{Int64}, activation::Array{Function}, weight_initialization::Function)
        layers = Array{NetworkLayer}(0)
        for i in 1:(length(layer_sizes)-1)
            push!(layers, NetworkLayer(layer_sizes[i], layer_sizes[i+1], activation[i],weight_initialization))
        end
        return (NeuralNetwork(layers))
    end

    function NeuralNetwork(layer::NetworkLayer)
        return new([layer])
    end
end

function CopyNetwork(neural_network::NeuralNetwork)
    return NeuralNetwork(map(x -> CopyLayer(x), neural_network.layers))
end

function InputSize(layer::NetworkLayer)
    return (size(layer.weights)[1])
end

function OutputSize(layer::NetworkLayer)
    return size(layer.weights)[2]
end

function FirstLayer(network::NeuralNetwork)
    return (network.layers[1])
end

function LastLayer(network::NeuralNetwork)
    return network.layers[end]
end

function InputSize(network::NeuralNetwork)
    return InputSize(FirstLayer(network))
end

function OutputSize(network::NeuralNetwork)
    return OutputSize(LastLayer(network))
end

function WeightsWithoutBias(layer::NetworkLayer)
    return(layer.weights[2:end,:])
end

function AddLayer(network::NeuralNetwork, layer::NetworkLayer)

    #FFN or RBM reversal
    if ((1+OutputSize(LastLayer(network))) == InputSize(layer)||(OutputSize(LastLayer(network))) == InputSize(layer))
        push!(network.layers, layer)
    else
        println(size(LastLayer(network).weights))
        println(size(layer.weights))
        throw(ErrorException("Error adding layer to network: Incorrect Output/Input dimensions."))
    end
end

end

#=Tests

layer = NetworkLayer(100, 50, ActivationFunctions.sigmoidActivation)
layer2 = NetworkLayer(50, 10, ActivationFunctions.sigmoidActivation)
layer3 = NetworkLayer(30, 10, ActivationFunctions.sigmoidActivation)
layer4 = NetworkLayer(10, 5, ActivationFunctions.sigmoidActivation)
network = NeuralNetwork(layer)
AddLayer(network, layer2)
AddLayer(network, layer4)

FirstLayer(network)
LastLayer(network)
InputSize(layer3)
OutputSize(layer3)

network2 = NeuralNetwork([layer, layer2,  layer3, layer4])
network3 = NeuralNetwork([layer, layer2,  layer4])

=#
