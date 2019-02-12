module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions, DatabaseOps

export RunSGD

function RunSGD(config_id, category, dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters)

    number_batches = Int64.(floor(size(dataset.training_input)[1]/parameters.minibatch_size))
    epoch_records = Array{EpochRecord}(0)
    num_training_samples = size(dataset.training_input)[1]

    mbi_input = map(m -> ((m-1)*parameters.minibatch_size+1), 1:number_batches)
    mbi_output = map(m -> m*parameters.minibatch_size, 1:number_batches)


    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))

    for i in 1:(parameters.max_epochs)
        tic()
        #weight_change_rates = Array{Array{Float64,1},1}()
        zero_activation_history = (fill(0, (length(network.layers),1)))

        epoch_order = randperm(size(dataset.training_input)[1])
        epoch_input = (dataset.training_input[epoch_order,:])
        epoch_output = (dataset.training_output[epoch_order,:])

        total_weight_changes = Array{Float64,2}(fill(0.0, (length(network.layers), 1)))

        for m in 1:number_batches
            #weight_updates, zero_activations = GradientDescentWeightUpdate(network, epoch_input[mbi_input[m]:mbi_output[m],:], epoch_output[mbi_input[m]:mbi_output[m],:], parameters)
            GradientDescentWeightUpdate(network, epoch_input[mbi_input[m]:mbi_output[m],:], epoch_output[mbi_input[m]:mbi_output[m],:], parameters, weight_updates, zero_activations)
            zero_activation_history += zero_activations

            for l in 1:length(network.layers)
                network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, num_training_samples, i)
                total_weight_changes[l,1] += mean(weight_updates[l])
            end

        end

        if (i%1) == 0

            mean_weight_changes = total_weight_changes[:,1] ./ number_batches

            layer_sizes = map(l -> size(l.weights, 2), network.layers)
            total_activations = layer_sizes .* size(dataset.training_input)[1]
            zero_perc =  (zero_activation_history ./ total_activations)'

            IS_error = parameters.cost_function.CalculateCost(Array{Float64,2}(dataset.training_output), Feedforward(network, dataset.training_input)[end])
            OOS_error = parameters.cost_function.CalculateCost(Array{Float64,2}(dataset.testing_output), Feedforward(network, dataset.testing_input)[end])

            epoch_record = EpochRecord(i, category, IS_error, OOS_error, 0.0, 0.0, 0.0, toq(),CopyNetwork(network), nothing, Array{Array{Float64,2},1}(), mean_weight_changes, zero_perc)

            push!(epoch_records, epoch_record)

            if parameters.verbose
                PrintEpoch(epoch_records[end])
            end

            CreateEpochRecord(config_id, epoch_record)

            #if parameters.stopping_function(epoch_records)
            #    break
            #end
        else
            println(string(i, " : ", toq()))
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
