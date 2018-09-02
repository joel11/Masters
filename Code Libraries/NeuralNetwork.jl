module NeuralNetworks

import ActivationFunctions, InitializationFunctions

export ActivationFunctions, InitializationFunctions, NetworkLayer

type NetworkLayer
    weights::Array{Float64, 2}
    activation::Function

    function NetworkLayer(weights, activation)
        return new(weights, activation)
    end

    function NetworkLayer(input::Int64, output::Int64, activation)
        return new(InitializationFunctions.HintonUniformInit(input, output), activation)
    end
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

    function NeuralNetwork(layerSizes::Array{Int64}, activation::Function)
        layers = Array{NetworkLayer}(0)
        for i in 1:(length(layerSizes)-1)
            push!(layers, NetworkLayer(layerSizes[i], layerSizes[i+1], activation))
        end
        return(NeuralNetwork(layers))
    end

    function NeuralNetwork(layer::NetworkLayer)
        return new([layer])
    end
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

function AddLayer(network::NeuralNetwork, layer::NetworkLayer)
    if OutputSize(LastLayer(network)) == InputSize(layer)
        push!(network.layers, layer)
    else
        throw(ErrorException("Error adding layer to network: Incorrect Output/Input dimensions"))
    end
end

function Feedforward(network::NeuralNetwork, input)
    return(Feedforward(network.layers, input))
end

function Feedforward(layers::Array{NetworkLayer}, input)
    tempInput = input
    for i in 1:length(layers)
        tempInput = layers[i].activation(tempInput * layers[i].weights)
    end

    return tempInput
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
