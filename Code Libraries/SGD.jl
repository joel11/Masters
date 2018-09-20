module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  AutoEncoder, CostFunctions, FFN

export RunSGD, TrainFFNNetwork

function TrainFFNNetwork(training_data, validation_data, layer_sizes::Array{Int64}, initialization::Function, parameters::TrainingParameters, cost_function)
    activation_functions = GenerateActivationFunctions(length(layer_sizes))
    rbm_network, rbm_records = TrainRBMNetwork(training_data, validation_data, layer_sizes, activation_functions, initialization, parameters)
    sgd_records = RunSGD(training_data, validation_data, rbm_network, parameters, cost_function)

    return (rbm_network, rbm_records, sgd_records)
end


function RunSGD(training_input, training_output, validation_inout, validation_output,  network::NeuralNetwork, parameters::TrainingParameters, cost_function::Function)

    number_batches = Int64.(floor(size(training_data)[1]/parameters.minibatch_size))
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_ffn_epochs)
        tic()
        minibatch_errors = []
        weight_change_rates = Array{Array{Float64,1},1}()

        epoch_order = randperm(size(training_data)[1])
        epoch_input = training_input[epoch_order,:]
        epoch_output = training_output[epoch_order,:]

        for m in 1:number_batches
            minibatch_input = epoch_input[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]
            minibatch_ouput = epoch_output[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]

            activations = Feedforward(network, minibatch_input)
            lambdas = CalculateLambdaErrors(network, activations, minibatch_ouput)
            weight_changes = CalculateWeightChanges(activations, lambdas)

            for l in 1:length(network.layers)
                network.layers[l].weights -= parameters.learning_rate .* weight_changes[l]   #.* momentum
            end

            new_activations = Feedforward(network, minibatch_input)[end]

            #Weight Change Rate
            if m % 100 == 0
                push!(weight_change_rates, map((x, y) -> mean(x[2:end,2:end] ./ y[2:end,2:end]), weight_changes, map(x -> x.weights, network.layers)))
            end

            #momentum = parameters.momentum_rate .* momentum_old + (1 - parameters.momentum_rate) .* weight_change
            #momentum_old = momentum

            push!(minibatch_errors, cost_function(minibatch_ouput, new_activations))
        end

        validation_estimations = Feedforward(network, validation_data)[end]
        oos_error = cost_function(validation_data, validation_estimations)
        ce_error = CrossEntropyError(validation_data, validation_estimations)

        push!(epoch_records, EpochRecord(i,
                                        mean(minibatch_errors),
                                        oos_error,
                                        ce_error,
                                        toc(),
                                        0.0,
                                        CopyNetwork(network),
                                        weight_change_rates,
                                        Array{Array{Float64,2},1}()
                                        ))

        PrintEpoch(epoch_records[end])

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

function CalculateLambdaErrors(network::NeuralNetwork, activations, training_output)
    layers = network.layers
    error = activations[end] - training_output
    z_vals = CalculateZVal(layers[end].weights, activations[(end-1)])
    derivative_activations = FunctionDerivatives[layers[end].activation](z_vals)
    lambda_L = error .* derivative_activations
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

function ReverseVector(vector)
    return(vector[(length(vector):-1:1)])
end

end
