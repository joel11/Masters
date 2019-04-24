module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions, DatabaseOps, DataProcessor
using DataFrames
export RunSGD

function RunSGD(config_id, category, original_dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters)

    # Working memory objects - stops incremental memory allocations which slow things down
    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    zero_activation_history = (fill(0, (length(network.layers),1)))
    total_weight_changes = Array{Float64,2}(fill(0.0, (length(network.layers), 1)))

    epoch_records = Array{EpochRecord}(parameters.max_epochs)
    num_training_samples = Int64(floor(size(original_dataset.training_input)[1] * parameters.training_splits[1])) #size(dataset.training_input)[1]
    number_batches = Int64.(floor(num_training_samples/parameters.minibatch_size))

    mbi_input = map(m -> ((m-1)*parameters.minibatch_size+1), 1:number_batches)
    mbi_output = map(m -> m*parameters.minibatch_size, 1:number_batches)
    layer_sizes = map(l -> size(l.weights, 2), network.layers)
    total_activations = layer_sizes .* num_training_samples

    training_input = Array{Float64,2}(original_dataset.training_input)
    training_output = Array{Float64,2}(original_dataset.training_output)
    testing_input = Array{Float64,2}(original_dataset.testing_input)
    testing_output = Array{Float64,2}(original_dataset.testing_output)

    original_input = Array{Float64,2}(original_dataset.training_input)
    original_output = Array{Float64,2}(original_dataset.training_output)
    split_point = Int64(floor(size(original_input,1) * parameters.training_splits[1]))

    for i in 1:(parameters.max_epochs)
        tic()

        indice_order = 1:(size(original_input, 1)) #randperm(size(original_input, 1))
        training_indices = indice_order[1:split_point]
        testing_indices = indice_order[(split_point + 1): end]
        random_order_training_indices = randperm(length(training_indices))

        training_input = original_input[random_order_training_indices,:]
        training_output = original_output[random_order_training_indices,:]
        testing_input = original_input[testing_indices,:]
        testing_output = original_output[testing_indices,:]

        if (parameters.is_denoising)
            training_input = AddNoiseToArray(training_input, parameters.denoising_variance)
        end

        zero_activation_history = (fill(0, (length(network.layers),1)))
        total_weight_changes = Array{Float64,2}(fill(0.0, (length(network.layers), 1)))

        for m in 1:number_batches
            GradientDescentWeightUpdate(network, training_input[mbi_input[m]:mbi_output[m],:], training_output[mbi_input[m]:mbi_output[m],:], parameters, weight_updates, zero_activations, activations)
            zero_activation_history += zero_activations

            for l in 1:length(network.layers)
                network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, num_training_samples, i)
                total_weight_changes[l,1] += mean(weight_updates[l])
            end

        end

        mean_weight_changes = total_weight_changes[:,1] ./ number_batches
        zero_perc =  (zero_activation_history ./ total_activations)'

        IS_error = parameters.cost_function.CalculateCost(training_output, Feedforward(network, training_input)[end])
        test_recreation = Feedforward(network, testing_input)[end]
        OOS_error = parameters.cost_function.CalculateCost(testing_output, test_recreation)
        #OOS_error = PredictionAccuracy(network, testing_input, testing_output)

        epoch_records[i] = EpochRecord(i, category, IS_error, OOS_error, 0.0, 0.0, 0.0, toq(), deepcopy(network), nothing, Array{Array{Float64,2},1}(), mean_weight_changes, zero_perc, CalculateLearningRate(i, parameters))

        if true
            PrintEpoch(epoch_records[i])
        end

        CreateEpochRecord(config_id, epoch_records[i])

        #if parameters.stopping_function(epoch_records)
        #    break
        #end
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
