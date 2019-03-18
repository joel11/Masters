module SGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions, DatabaseOps

export RunSGD

seed = abs(Int64.(floor(randn()*100)))
ds = abs(Int64.(floor(randn()*100)))
var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
data_config = DatasetConfig(ds, "test",  5000,  [1, 7, 30],  [0.6],  [0.8, 1.0],  [2], var_pairs)
dataset = nothing

function PrepareData(data_config, dataset)
    data_raw = dataset == nothing ? GenerateDataset(data_config.data_seed, data_config.steps, data_config.variation_values) : dataset
    processed_data = ProcessData(data_raw, data_config.deltas, data_config.prediction_steps)
    standardized_data = map(x -> StandardizeData(x)[1], processed_data)
    data_splits = map(df -> SplitData(df, data_config.process_splits), standardized_data)

    saesgd_data = CreateDataset(data_splits[1][1], data_splits[2][1], [1.0])
    ogd_data = CreateDataset(data_splits[1][2], data_splits[2][2], [1.0])

    return(saesgd_data, ogd_data)
end


function RunSGD(config_id, category, original_dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters)

    # Working memory objects - stops incremental memory allocations which slow things down
    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    number_batches = Int64.(floor(size(dataset.training_input)[1]/parameters.minibatch_size))
    epoch_records = Array{EpochRecord}(parameters.max_epochs)
    num_training_samples = Int64(floor(rows * parameters.training_splits[1])) #size(dataset.training_input)[1]

    mbi_input = map(m -> ((m-1)*parameters.minibatch_size+1), 1:number_batches)
    mbi_output = map(m -> m*parameters.minibatch_size, 1:number_batches)
    layer_sizes = map(l -> size(l.weights, 2), network.layers)
    total_activations = layer_sizes .* num_training_samples

    for i in 1:(parameters.max_epochs)
        tic()

        dataset = GenerateRandomisedDataset(original_dataset)

        training_input = Array{Float64,2}(dataset.training_input)
        training_output = Array{Float64,2}(dataset.training_output)

        testing_input = Array{Float64,2}(dataset.testing_input)
        testing_output = Array{Float64,2}(dataset.testing_output)

        #epoch_order = randperm(size(training_input)[1])
        #epoch_input = (training_input[epoch_order,:])
        #epoch_output = (training_output[epoch_order,:])

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
        OOS_error = parameters.cost_function.CalculateCost(testing_output, Feedforward(network, testing_input)[end])

        epoch_records[i] = EpochRecord(i, category, IS_error, OOS_error, 0.0, 0.0, 0.0, toq(), deepcopy(network), nothing, Array{Array{Float64,2},1}(), mean_weight_changes, zero_perc)

        if parameters.verbose
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
