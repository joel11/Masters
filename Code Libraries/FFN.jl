module FFN

using ActivationFunctions, InitializationFunctions, NeuralNetworks

export Feedforward, Feedforwad_Prealloc

function Feedforward(network::NeuralNetwork,  input)
    return (Feedforward(network.layers, input))
end

function Feedforward(layers::Array{NetworkLayer},  input)
    current_vals = Array{Float64,2}(input)
    layer_outputs = Array{Array{Float64,2},1}(length(layers)+1)
    layer_outputs[1] = current_vals
    #push!(layer_outputs, current_vals)
    for i in 1:length(layers)
        bias_vals = Array{Float64,2}(hcat(fill(1.0, size(current_vals,1)), current_vals))
        #layer_outputs[(i+1)] = layers[i].activation(bias_vals * layers[i].weights)
        current_vals = layers[i].activation(bias_vals * layers[i].weights)
        layer_outputs[(i+1)] = current_vals
    #    push!(layer_outputs, current_vals)
    end

    return layer_outputs
end

function Feedforwad_Prealloc(network::NeuralNetwork,  layer_outputs::Array{Array{Float64,2},1})
    Feedforwad_Prealloc(network.layers, layer_outputs)
end

function Feedforwad_Prealloc(layers::Array{NetworkLayer}, layer_outputs::Array{Array{Float64,2},1})

    for i in 1:length(layers)
        bias_vals = Array{Float64,2}(hcat(fill(1.0, size(layer_outputs[i],1)), layer_outputs[i]))
        current_vals = layers[i].activation(bias_vals * layers[i].weights)
        layer_outputs[(i+1)] = current_vals
    end

    nothing

end

end
