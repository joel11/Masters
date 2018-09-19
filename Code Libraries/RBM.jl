module RBM

using TrainingStructures
using NeuralNetworks, TrainingStructures, ActivationFunctions, InitializationFunctions, CostFunctions, FFN
using Distributions

export CreateRBMLayer, TrainRBMNetwork, TrainRBMLayer, ReconstructVisible

function CreateRBMLayer(inputSize::Int64, outputSize::Int64, activation::Function,  initialization::Function)
    weights = [0.0; hcat(fill(0.0, inputSize), initialization(inputSize, outputSize))]
    layer = NeuralNetworks.NetworkLayer(weights, activation)
    return layer
end

function RemoveBackwardsBias(layer::NeuralNetworks.NetworkLayer)
    layer.weights = layer.weights[:,2:end]
end

#Training Methods #############################################################

function TrainRBMNetwork(training_data::Array{Float64,2}, validation_data::Array{Float64,2}, layer_sizes::Array{Int64}, activation_functions, initialization::Function, parameters::TrainingParameters)

    layer = CreateRBMLayer(layer_sizes[1], layer_sizes[2], activation_functions[1], initialization)
    epoch_records = [TrainRBMLayer(training_data, validation_data, layer, parameters)]
    RemoveBackwardsBias(layer)
    network = NeuralNetwork([layer])

    for i in 2:(length(layer_sizes)-1)
        processed_training_data = Feedforward(network, training_data)[end]
        processed_validation_data = Feedforward(network, validation_data)[end]

        next_layer = CreateRBMLayer(layer_sizes[i], layer_sizes[(i+1)], activation_functions[i], initialization)
        new_epoch_records = TrainRBMLayer(processed_training_data, processed_validation_data, next_layer, parameters)
        push!(epoch_records, new_epoch_records)
        RemoveBackwardsBias(next_layer)
        AddLayer(network, next_layer)
    end

    return (network, epoch_records)
end

function TrainRBMLayer(training_data, validation_data, layer::NeuralNetworks.NetworkLayer, parameters::TrainingParameters)

    data_b = hcat(fill(1.0, size(training_data,1)), training_data)
    number_batches = Int64.(floor(size(training_data)[1]/parameters.minibatch_size))
    momentum_old = zeros(layer.weights)
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_rbm_epochs)

        tic()
        epoch_data = data_b[(randperm(size(training_data)[1])),:]
        minibatch_errors = []
        minibatch_crosserrors = []
        weight_change_rates = Array{Array{Float64,1},1}()
        hidden_activation_likelihoods = Array{Array{Float64,2},1}()

        for m in 1:number_batches
            minibatch_data = epoch_data[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]

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
            #weight_change = parameters.learning_rate * ((pos_cd - neg_cd) / size(minibatch_data, 1))
            weight_change = ((pos_cd - neg_cd) / size(minibatch_data, 1))
            if m % 1000 == 0
                push!(weight_change_rates, [mean(weight_change[2:end,2:end] ./ layer.weights[2:end,2:end])])
                push!(hidden_activation_likelihoods, activation_probabilities)
            end

            momentum = parameters.momentum_rate .* momentum_old + (1 - parameters.momentum_rate) .* weight_change
            momentum_old = momentum

            layer.weights += parameters.learning_rate .* momentum

            push!(minibatch_errors, MeanSquaredError(minibatch_data[2:end, 2:end], vis_activation_probabilities[2:end, 2:end]))
            push!(minibatch_crosserrors, CrossEntropyError(minibatch_data[2:end, 2:end], vis_activation_probabilities[2:end, 2:end]))
        end

        validation_error = MeanSquaredError(validation_data, ReconstructVisible(layer, validation_data))

        push!(epoch_records, EpochRecord(i,
                                        mean(minibatch_errors),
                                        validation_error,
                                        mean(minibatch_crosserrors),
                                        toc(),
                                        CalculateEpochFreeEnergy(layer, training_data, validation_data),
                                        NeuralNetwork(CopyLayer(layer)),
                                        weight_change_rates,
                                        hidden_activation_likelihoods
                                        ))

        PrintEpoch(epoch_records[end])

        if parameters.stopping_function(epoch_records)
            break
        end
    end

    return (epoch_records)
end

#Generative Methods############################################################

function ProcessInput(input, weights, net, return_state)

    data_b =  hcat(fill(1.0, size(input,1)), input)
    activation_probs = net.activation(data_b * weights)
    states = Int64.(activation_probs.>= rand(Uniform(0, 1), size(activation_probs,1), size(activation_probs,2)))

    #states[:,1] = 1
    return_vals = return_state ? states : activation_probs
    return_states = return_vals[:,2:end]
    return (return_states)
end

function ProcessVisible(net::NeuralNetworks.NetworkLayer, data, return_state)
    return (ProcessInput(data, net.weights, net, return_state))
end

function ProcessHidden(net::NeuralNetworks.NetworkLayer, data, return_state)
    return (ProcessInput(data, net.weights', net, return_state))
end

function ReconstructVisible(net::NeuralNetworks.NetworkLayer, data)
    hidden_activations = ProcessVisible(net, data, false)
    reconstruction = ProcessHidden(net, hidden_activations, false)
    return (reconstruction)
end

function ReconstructVisible(net::NeuralNetwork, data)
    temp_vals = data
    for l in 1:length(net.layers)
        temp_vals = ProcessVisible(net.layers[l], temp_vals, false)
    end

    for l in length(net.layers):-1:1
        temp_vals = ProcessHidden(net.layers[l], temp_vals, false)
    end
    return (temp_vals)
end

function CalculateFreeEnergy(net::NeuralNetworks.NetworkLayer, data)

    v_bias = net.weights[2:end,1]
    vbias_term = data*v_bias

    wx_b = RBM.ProcessVisible(net, data, false)
    hidden_term = sum((log.(1.+(e.^wx_b))), 2)

    F = -hidden_term - vbias_term
    return (F)
end

function CalculateEpochFreeEnergy(net::NeuralNetworks.NetworkLayer, training_data, validation_data)
    subset_size = Int64.(floor(size(validation_data,1)/10))
    training_free_energy = CalculateFreeEnergy(net, training_data[1:subset_size, :])
    validation_free_energy = CalculateFreeEnergy(net, validation_data[1:subset_size, :])

    return (mean(validation_free_energy)/mean(training_free_energy))
end

end
