module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions, DatabaseOps

export RunSGD


function RunSGD(config_id, category, dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters)

    number_batches = Int64.(floor(size(dataset.training_input)[1]/parameters.minibatch_size))
    epoch_records = Array{EpochRecord}(0)

    for i in 1:(parameters.max_epochs)
        tic()
        minibatch_errors = []
        weight_change_rates = Array{Array{Float64,1},1}()

        epoch_order = randperm(size(dataset.training_input)[1])
        epoch_input = dataset.training_input[epoch_order,:]
        epoch_output = dataset.training_output[epoch_order,:]

        total_weight_changes = fill(0.0, (length(network.layers), 1))

        for m in 1:number_batches
            minibatch_input = epoch_input[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:]
            minibatch_ouput = Array(epoch_output[((m-1)*parameters.minibatch_size+1):m*parameters.minibatch_size,:])

            weight_updates = GradientDescentWeightUpdate(network, minibatch_input, minibatch_ouput, parameters)
            total_weight_changes[:,1] += map(mean, weight_updates)

            for l in 1:length(network.layers)
                network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, size(dataset.training_input)[1])
            end



            #Weight Change Rate
            #if m % 100 == 0
            #    push!(weight_change_rates, map((x, y) -> mean(x[2:end,:] ./ y[2:end,:]), weight_updates, map(x -> x.weights, network.layers)))
            #end

            push!(minibatch_errors, 0)
            #push!(minibatch_errors, parameters.cost_function.CalculateCost(minibatch_ouput, Feedforward(network, minibatch_input)[end]))
        end

        mean_weight_changes = total_weight_changes[:,1] ./ number_batches

        IS_error = parameters.cost_function.CalculateCost(Array(dataset.training_output), Feedforward(network, dataset.training_input)[end])
        OOS_error = parameters.cost_function.CalculateCost(Array(dataset.testing_output), Feedforward(network, dataset.testing_input)[end])

        println("IS : $IS_error ; OS : $OOS_error")

        IS_accuracy = parameters.is_classification ? PredictionAccuracy(network, dataset.training_input, dataset.training_output) : 0
        OOS_accuracy = parameters.is_classification ? PredictionAccuracy(network, dataset.testing_input, dataset.testing_output) : 0

        println("mean weight changes")
        println(mean_weight_changes)
        epoch_record = EpochRecord(i, category, mean(minibatch_errors), IS_error, OOS_error, IS_accuracy, OOS_accuracy, 0.0, toq(), CopyNetwork(network), weight_change_rates, Array{Array{Float64,2},1}(), mean_weight_changes)

        push!(epoch_records, epoch_record)

        if parameters.verbose
            PrintEpoch(epoch_records[end])
        end

        CreateEpochRecord(config_id, epoch_record)

        if parameters.stopping_function(epoch_records)
            break
        end
    end

    network = GetBestNetwork(epoch_records)

    return (epoch_records)
end

function GetBestNetwork(epoch_records)
    minindex = findmin(map(x -> x.test_cost, epoch_records))[2]
    println(string(minindex, " ", epoch_records[minindex].test_cost))
    return epoch_records[minindex].network
end

function PredictionAccuracy(network, input, output)
    validation_pred = Feedforward(network, input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == output[i,:], 1:size(output)[1])))
    return(correct_predictions/size(output)[1])
end

end
