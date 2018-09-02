module FFN

push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using RBM, NeuralNetworks, ActivationFunctions, InitializationFunctions, AutoEncoder, TrainingStructures

export Feedforward


using MNIST
trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()
scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255

parameters = TrainingParameters(0.1, 20, 0.0,  1)
training_data = scaled_training_data
validation_data = scaled_validation_data
initialization = InitializationFunctions.XavierGlorotUniformInit
layer_sizes = [784, 30, 20, 5]

network =  CreateAutoEncoder(scaled_training_data, scaled_validation_data, layer_sizes, initialization, parameters)

function Feedforward(network::NeuralNetwork,  input::Array{Float64,2})
    return (Feedforward(network.layers, input))
end

function Feedforward(layers::Array{NetworkLayer},  input::Array{Float64,2})
    current_vals = input
    layer_outputs = Array{Array{Float64,2},1}()
    for i in 1:length(layers)
        bias_vals = hcat(fill(1.0, size(current_vals,1)), current_vals)
        current_vals = layers[i].activation(bias_vals * layers[i].weights)
        push!(layer_outputs, current_vals)
    end

    return layer_outputs
end

function CalculateZVal(weights, previous_activation)
    bias_vals = hcat(fill(1.0, size(previous_activation,1)), previous_activation)
    return (weights * bias_vals)
end

function CalculateLambdaErrors(network::NeuralNetwork, input::Array{Float64,2})
    layers = network.layers

    ff_vals = Feedforward(network, input)

    error = ff_vals[end] - input
    z_vals = CalculateZVal(layers[end], ff_vals[(end-1)])
    derivative_activation = Function_dictionary[layers[end].activation](z_vals)
    lambda_L = error .* derivative_activations

    lambdas = [lambda_L]

    for i in (length(layers)-1):-1:1
        output_act = layers[(i+1)]'*ff_vals[(i+1)]
        z_vals = CalculateZVal(layers[i], ff_vals[(i-1)])
        z_der = Function_dictionary[layers[i].activation](z_vals)
        lambda = output_act .* z_der
        push!(lambdas, lambda)
    end

    return(ReverseVector(lambdas))
end

function ReverseVector(vector)
    return(vector[(length(vector):-1:1)])
end


end
