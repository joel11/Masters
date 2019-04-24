module OGD

using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions, DatabaseOps, DataFrames

export RunOGD

function RunOGD(config_id, category, dataset::DataSet, network::NeuralNetwork, parameters::OGDTrainingParameters)

    tic()

    training_input = Array{Float64,2}(dataset.training_input)
    training_output = Array{Float64,2}(dataset.training_output)
    predicted_values = Array{Float64,2}(size(training_output))

    # Working memory objects - stops incremental memory allocations which slow things down
    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    for i in 1:size(dataset.training_input)[1]
        sample_input = Array{Float64,2}(training_input[i, 1:end]')
        sample_output = Array{Float64,2}(training_output[i, 1:end]')

        predicted_values[i, :] = Feedforward(network, sample_input)[end]

        GradientDescentWeightUpdate(network, sample_input, sample_output, parameters, weight_updates, zero_activations, activations)

        for l in 1:length(network.layers)
            network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, 1, 1)
        end

    end

    IS_error = parameters.cost_function.CalculateCost(training_output, predicted_values)

#    epoch_records[i] = EpochRecord(i, category, IS_error, OOS_error, 0.0, 0.0, 0.0, toq(), deepcopy(network), nothing, Array{Array{Float64,2},1}(), mean_weight_changes, zero_perc, CalculateLearningRate(i, parameters))

    epoch_record     = EpochRecord(1, category, IS_error, 0.0,       0.0, 0.0, 0.0, toq(), deepcopy(network), nothing, nothing,                     nothing,              nothing, parameters.max_learning_rate)

    CreateEpochRecord(config_id, epoch_record)

    if parameters.verbose
        PrintEpoch(epoch_record)
    end

    return ([epoch_record], (training_output, predicted_values))
end

function PredictionAccuracy(network, input, output)
    validation_pred = Feedforward(network, input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == output[i,:], 1:size(output)[1])))
    return(correct_predictions/size(output)[1])
end



end
