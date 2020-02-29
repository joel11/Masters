module GradientFunctions

using Distributions
using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, CostFunctions, FFN

export CalculateLearningRate, ContrastiveDivergence1WeightUpdates, GradientDescentWeightUpdate, CalculateNewWeights

function CalculateL2Penalization(learning_rate, parameters, N)
    return  (1 - learning_rate * parameters.l2_lambda /N)
end

function CalculateL1Penalization(parameters, N, weights)
    l1pen = ((parameters.max_learning_rate * parameters.l1_lambda / N) .* sign.(weights))
    return (l1pen)
end

function GradientDescentWeightUpdate(network::NeuralNetwork, minibatch_input::Array{Float64,2}, minibatch_ouput::Array{Float64,2}, parameters, weight_updates::Array{Array{Float64,2}}, zero_activations::Array{Int64,2}, activations::Array{Array{Float64,2},1})

    function CalculateWeightChanges(activations::Array{Array{Float64,2},1}, lambdas::Array{Array{Float64,2},1}, weight_changes::Array{Array{Float64,2},1})

        for i in 1:length(lambdas)
            sample_size = size(activations[i], 1)
            acts_withbias = hcat(fill(1.0, sample_size), activations[i])
            weight_changes[i] = (acts_withbias' * lambdas[i]) ./ sample_size
            #push!(weight_changes, change_values)
        end

        nothing#return (weight_changes)
    end

    function CalculateZVal(weights::Array{Float64,2}, previous_activation::Array{Float64,2})
        return Array{Float64,2}(hcat(fill(1.0, size(previous_activation,1)), previous_activation) * weights)
    end

    function CalculateLambdaErrors(network::NeuralNetwork, activations::Array{Array{Float64,2}}, training_output::Array{Float64,2}, cost_function)
        z_vals = CalculateZVal(network.layers[end].weights, activations[(end-1)])
        lambdas = Array{Array{Float64,2}}(length(network.layers))
        lambdas[end] = cost_function.Delta(activations[end], training_output, z_vals, network.layers[end].activation)

        for i in (length(network.layers)-1):-1:1

            z_vals = CalculateZVal(network.layers[i].weights,activations[(i)])
            z_der = FunctionDerivatives[network.layers[i].activation](z_vals)

            output_act = lambdas[(i+1)] * network.layers[(i+1)].weights'[:, 2:end]
            lambdas[i] = output_act .* z_der
        end

        return lambdas
    end

    function CalculateZeroValues(activations, zero_activations)
        for i in 2:length(activations)
            zero_activations[(i-1)] = sum(activations[i] .== 0)
        end
    end


    activations[1] = minibatch_input
    Feedforwad_Prealloc(network, activations)

    lambdas = CalculateLambdaErrors(network, activations, minibatch_ouput, parameters.cost_function)

    CalculateWeightChanges(activations, lambdas, weight_updates)
    CalculateZeroValues(activations, zero_activations)

    nothing

end

function ContrastiveDivergence1WeightUpdates(minibatch_data, layer)
    #Calc Pos CD
    activations = minibatch_data * layer.weights
    activation_probabilities = layer.activation(activations)
    activation_probabilities[:,1] = 1 #Fix Bias
    hidden_states = activation_probabilities .> rand(Uniform(0, 1), size(activation_probabilities,1), size(activation_probabilities,2))
    pos_cd = minibatch_data'*activation_probabilities

    #Calc Neg CD
    vis_activations = hidden_states * layer.weights'
    vis_activation_probabilities = layer.activation(vis_activations)
    vis_activation_probabilities[:,1] = 1 #Fix Bias
    neg_hidden_states = vis_activation_probabilities * layer.weights
    neg_hidden_probs = layer.activation(neg_hidden_states)
    neg_cd = vis_activation_probabilities'*neg_hidden_probs

    #Weight Change
    weight_update = ((neg_cd - pos_cd) / size(minibatch_data, 1))
    return (weight_update, activation_probabilities, vis_activation_probabilities)
end

function CalculateLearningRate(epoch, training_parameters)

    if (split(string(typeof(training_parameters)), ".")[end] == "OGDTrainingParameters" || training_parameters.epoch_cycle_max < 0)
        return training_parameters.max_learning_rate
    end

    return training_parameters.min_learning_rate + 0.5*(training_parameters.max_learning_rate - training_parameters.min_learning_rate)*(1 + cos(epoch/training_parameters.epoch_cycle_max*pi))
    #return training_parameters.max_learning_rate
end

function CalculateNewWeights(current_weights, weight_update, parameters, N::Int64, epoch::Int64)
    return (current_weights #.* CalculateL2Penalization(CalculateLearningRate(epoch, parameters), parameters, N)
                                - CalculateLearningRate(epoch, parameters) .* weight_update
                                - CalculateL1Penalization(parameters, N, current_weights)
                                )
end


end
