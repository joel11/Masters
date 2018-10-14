module GradientFunctions

using Distributions
using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, CostFunctions, FFN

export ContrastiveDivergence1WeightUpdates, GradientDescentWeightUpdate, CalculateNewWeights

function ReverseVector(vector)
    return(vector[(length(vector):-1:1)])
end

function CalculateL2Penalization(parameters, N)
    l2pen =  (1 - parameters.learning_rate * parameters.l2_lambda /N)
    return (l2pen)
end

function CalculateL1Penalization(parameters, N, weights)
    l1pen = ((parameters.learning_rate * parameters.l1_lambda / N) .* sign(weights))
    return (l1pen)
end

function GradientDescentWeightUpdate(network, minibatch_input, minibatch_ouput, parameters)

    function CalculateWeightChanges(activations, lambdas)

        weight_changes = Array{Array{Float64,2},1}()

        for i in 1:length(lambdas)
            acts_withbias = hcat(fill(1.0, size(activations[i],1)), activations[i])
            change_values = acts_withbias'*lambdas[i]
            scaled_changes = (change_values ./ size(activations[i],1))
            push!(weight_changes, scaled_changes)
        end

        return (weight_changes)
    end

    function CalculateZVal(weights, previous_activation)
        bias_vals = hcat(fill(1.0, size(previous_activation,1)), previous_activation)
        return (bias_vals * weights)
    end

    function CalculateLambdaErrors(network::NeuralNetwork, activations, training_output, cost_function)
        layers = network.layers
        error = activations[end] - training_output
        z_vals = CalculateZVal(layers[end].weights, activations[(end-1)])
        #derivative_activations = FunctionDerivatives[layers[end].activation](z_vals)
        lambda_L =  cost_function.Delta(activations[end], training_output, z_vals, layers[end].activation)# derivative_activations)
        lambdas = [lambda_L]

        for i in (length(layers)-1):-1:1
            z_vals = CalculateZVal(layers[i].weights,activations[(i)])

            z_der = FunctionDerivatives[layers[i].activation](z_vals)
            output_act = lambdas[(length(layers)-i)] * layers[(i+1)].weights'[:, 2:end]
            lambda = output_act .* z_der
            push!(lambdas, lambda)
        end

        return(ReverseVector(lambdas))
    end

    activations = Feedforward(network, minibatch_input)
    lambdas = CalculateLambdaErrors(network, activations, minibatch_ouput, parameters.cost_function)
    weight_updates = CalculateWeightChanges(activations, lambdas)
    return (weight_updates)
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

function CalculateNewWeights(current_weights, weight_update, parameters::TrainingParameters, N::Int64)
    return (current_weights .* CalculateL2Penalization(parameters, N)
                                #+ momentum_factor
                                - parameters.learning_rate .* weight_update
                                - CalculateL1Penalization(parameters, N, current_weights))
end

function CalculateMomentum(momentum, parameters, weight_updates)
    #for l in 1:length(momentum)
        #momentum = momentum * parameters.momentum_rate
end

end
