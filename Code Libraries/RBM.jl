module RBM

using TrainingStructures
using GradientFunctions
using NeuralNetworks, TrainingStructures, ActivationFunctions, InitializationFunctions, CostFunctions, FFN
using Distributions
using DatabaseOps
using DataProcessor

export CreateRBMLayer, TrainRBMNetwork, TrainRBMLayer, ReconstructVisible

function CreateRBMLayer(i, network_parameters::NetworkParameters)
    weights = [0.0; hcat(fill(0.0, network_parameters.layer_sizes[i]), network_parameters.initialization(network_parameters.layer_sizes[i], network_parameters.layer_sizes[(i+1)]))]
    layer = NeuralNetworks.NetworkLayer(weights, network_parameters.layer_activations[i])
    return layer
end

function RemoveBackwardsBias(layer::NeuralNetworks.NetworkLayer)
    layer.weights = layer.weights[:,2:end]
end

#Training Methods #############################################################

function TrainRBMNetwork(config_id, dataset::DataSet, network_parameters::NetworkParameters, training_parameters::TrainingParameters)

    layer = CreateRBMLayer(1, network_parameters)
    epoch_records = [TrainRBMLayerOld(config_id, dataset.training_input, layer, training_parameters)]
    RemoveBackwardsBias(layer)
    network = NeuralNetwork([layer])

    for i in 2:(length(network_parameters.layer_sizes)-1)
        println("i : $i")
        processed_training_data = Feedforward(network, dataset.training_input)[end]
        #processed_testing_data = Feedforward(network, dataset.testing_input)[end]

        next_layer = CreateRBMLayer(i, network_parameters)
        new_epoch_records = TrainRBMLayerOld(config_id, processed_training_data, next_layer, training_parameters)
        push!(epoch_records, new_epoch_records)
        RemoveBackwardsBias(next_layer)
        AddLayer(network, next_layer)
    end

    return (network, epoch_records)
end



function TrainRBMLayer(config_id, training_data, layer::NeuralNetworks.NetworkLayer, parameters::TrainingParameters)

    biased_training_data = hcat(fill(1.0, size(training_data,1)), Array(training_data))
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_epochs)

        tic()

        dataset = GenerateRandomisedDataset(biased_training_data, biased_training_data, parameters)
        training_input = Array{Float64,2}(dataset.training_input)
        testing_input = Array{Float64,2}(dataset.testing_input)

        #println(size(training_input))
        #println(size(testing_input))
        #println(training_input[1,:])

        weight_change_rates = Array{Array{Float64,1},1}()
        hidden_activation_likelihoods = Array{Array{Float64,2},1}()

        number_batches = Int64.(floor(size(training_input)[1]/parameters.minibatch_size))
        #minibatch_errors = []

        for m in 1:number_batches

            minibatch_data = training_input[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]
            weight_update, activation_probabilities, vis_activation_probabilities = ContrastiveDivergence1WeightUpdates(minibatch_data, layer)

            layer.weights -= parameters.max_learning_rate .* weight_update

            #if m % 1000 == 0
            #    push!(weight_change_rates, [mean(weight_update[2:end,2:end] ./ layer.weights[2:end,2:end])])
            #    push!(hidden_activation_likelihoods, activation_probabilities)
            #end

            #push!(minibatch_errors, parameters.cost_function.CalculateCost(minibatch_data[2:end, 2:end], vis_activation_probabilities[2:end, 2:end]))
        end

        training_error = parameters.cost_function.CalculateCost(training_input, ReconstructVisible(layer, training_input))
        testing_error = parameters.cost_function.CalculateCost(testing_input, ReconstructVisible(layer, testing_input))

        #EpochRecord(i, category,    IS_error,               OOS_error,     0.0, 0.0, 0.0, toq(), deepcopy(network),               nothing,             Array{Array{Float64,2},1}(),   mean_weight_changes, zero_perc, OOS_mape, CalculateLearningRate(i, parameters))
        epoch_record = EpochRecord(i, "RBM-CD",    training_error,         testing_error, 0.0, 0.0, 0.0, toq(), NeuralNetwork(CopyLayer(layer)), weight_change_rates, hidden_activation_likelihoods, Array{Float64,2}(), 0.0, 0.0, 0.0)
        CreateEpochRecord(config_id, epoch_record)
        push!(epoch_records, epoch_record)

        if parameters.verbose
            PrintEpoch(epoch_records[end])
        end

        if parameters.stopping_function(epoch_records)
            break
        end
    end

    return (epoch_records)
end

function TrainRBMLayerOld(config_id, training_data , layer::NeuralNetworks.NetworkLayer, parameters::TrainingParameters)

    biased_training_data = hcat(fill(1.0, size(training_data,1)), Array(training_data))
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_epochs)

        tic()

        dataset = GenerateRandomisedDataset(biased_training_data, biased_training_data, parameters)
        training_input = Array{Float64,2}(dataset.training_input)
        testing_input = Array{Float64,2}(dataset.testing_input)

        #println(size(training_input))
        #println(size(testing_input))
        #println(training_input[1,:])

        weight_change_rates = Array{Array{Float64,1},1}()
        hidden_activation_likelihoods = Array{Array{Float64,2},1}()

        number_batches = Int64.(floor(size(training_input)[1]/parameters.minibatch_size))
        minibatch_errors = []

        for m in 1:number_batches
            minibatch_data = training_input[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]

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
            #if m % 1000 == 0
            #    push!(weight_change_rates, [mean(weight_change[2:end,2:end] ./ layer.weights[2:end,2:end])])
            #    push!(hidden_activation_likelihoods, activation_probabilities)
            #end

            layer.weights += parameters.max_learning_rate .* weight_change

            push!(minibatch_errors, MeanSquaredError().CalculateCost(minibatch_data[2:end, 2:end], vis_activation_probabilities[2:end, 2:end]))
        end

        training_error = parameters.cost_function.CalculateCost(training_input, ReconstructVisible(layer, training_input))
        testing_error = parameters.cost_function.CalculateCost(testing_input, ReconstructVisible(layer, testing_input))

        epoch_record = EpochRecord(i, "RBM-CD",    training_error,         testing_error, 0.0, 0.0, 0.0, toq(), NeuralNetwork(CopyLayer(layer)), weight_change_rates, hidden_activation_likelihoods, Array{Float64,2}(), 0.0, 0.0, 0.0)
        CreateEpochRecord(config_id, epoch_record)
        push!(epoch_records, epoch_record)

        if parameters.verbose
            PrintEpoch(epoch_records[end])
        end

        if parameters.stopping_function(epoch_records)
            break
        end
    end

    return (epoch_records)
end

#Generative Methods############################################################

function ProcessInput(input, weights, net, return_state)

    data_b =  input#hcat(fill(1.0, size(input,1)), input)
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
    biased_hidden = hcat(fill(1.0, size(hidden_activations,1)), hidden_activations)
    reconstruction = ProcessHidden(net, biased_hidden, false)
    biased_reconstruction = hcat(fill(1.0, size(reconstruction,1)), reconstruction)
    return (biased_reconstruction)
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

function CalculateEpochFreeEnergy(net::NeuralNetworks.NetworkLayer, training_data, testing_data)
    subset_size = Int64.(floor(size(testing_data,1)/10))
    training_free_energy = CalculateFreeEnergy(net, training_data[1:subset_size, :])
    testing_free_energy = CalculateFreeEnergy(net, testing_data[1:subset_size, :])

    return (mean(testing_free_energy)/mean(training_free_energy))
end

end
