module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN

export RunSGD

function RunSGD(dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters, cost_function)

    number_batches = Int64.(floor(size(dataset.training_input)[1]/parameters.minibatch_size))
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_ffn_epochs)
        tic()
        minibatch_errors = []
        weight_change_rates = Array{Array{Float64,1},1}()

        epoch_order = randperm(size(dataset.training_input)[1])
        epoch_input = dataset.training_input[epoch_order,:]
        epoch_output = dataset.training_output[epoch_order,:]

        for m in 1:number_batches
            minibatch_input = epoch_input[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]
            minibatch_ouput = epoch_output[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]

            activations = Feedforward(network, minibatch_input)
            lambdas = CalculateLambdaErrors(network, activations, minibatch_ouput, cost_function)
            weight_changes = CalculateWeightChanges(activations, lambdas)
            N = size(dataset.training_input)[1]

            for l in 1:length(network.layers)
                network.layers[l].weights = (network.layers[l].weights .* CalculateL2Penalization(parameters, N)
                                            - parameters.learning_rate .* weight_changes[l]
                                            - CalculateL1Penalization(parameters, N, network.layers[l].weights))

            end

            new_activations = Feedforward(network, minibatch_input)[end]

            #Weight Change Rate
            if m % 100 == 0
                wc = map((x, y) -> mean(x[2:end,2:end] ./ y[2:end,2:end]), weight_changes, map(x -> x.weights, network.layers))
                push!(weight_change_rates, wc)
            end

            push!(minibatch_errors, cost_function.CalculateCost(minibatch_ouput, new_activations))
        end


        IS_error = cost_function.CalculateCost(dataset.training_output, Feedforward(network, dataset.training_input)[end])
        OOS_error = cost_function.CalculateCost(dataset.testing_output, Feedforward(network, dataset.testing_input)[end])

        IS_accuracy = parameters.is_classification ? PredictionAccuracy(network, dataset.training_input, dataset.training_output) : 0
        OOS_accuracy = parameters.is_classification ? PredictionAccuracy(network, dataset.testing_input, dataset.testing_output) : 0

        #epoch_number, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods

        push!(epoch_records, EpochRecord(i,
                                        mean(minibatch_errors),
                                        IS_error,
                                        OOS_error,
                                        IS_accuracy,
                                        OOS_accuracy,
                                        0.0,
                                        toq(),
                                        CopyNetwork(network),
                                        weight_change_rates,
                                        Array{Array{Float64,2},1}()
                                        ))

        if parameters.verbose
            PrintEpoch(epoch_records[end])
        end

        if parameters.stopping_function(epoch_records)
            break
        end
    end

    return (epoch_records)
end


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

function CalculateL2Penalization(parameters, N)
    l2pen =  (1 - parameters.learning_rate * parameters.l2_lambda /N)
    return (l2pen)
end

function CalculateL1Penalization(parameters, N, weights)
    l1pen = ((parameters.learning_rate * parameters.l1_lambda / N) .* sign(weights))
    return (l1pen)
end

function ReverseVector(vector)
    return(vector[(length(vector):-1:1)])
end

function PredictionAccuracy(network, input, output)
    validation_pred = Feedforward(network, input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == output[i,:], 1:size(output)[1])))
    return(correct_predictions/size(output)[1])
end

end
