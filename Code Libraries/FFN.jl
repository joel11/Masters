module FFN

using ActivationFunctions, InitializationFunctions, NeuralNetworks

export Feedforward

function Feedforward(network::NeuralNetwork,  input)
    return (Feedforward(network.layers, input))
end

function Feedforward(layers::Array{NetworkLayer},  input)
    current_vals = Array(input)
    layer_outputs = Array{Array{Float64,2},1}()
    push!(layer_outputs, current_vals)
    for i in 1:length(layers)
        bias_vals = hcat(fill(1.0, size(current_vals,1)), current_vals)
        current_vals = layers[i].activation(bias_vals * layers[i].weights)
        push!(layer_outputs, current_vals)
    end

    return layer_outputs
end

end
