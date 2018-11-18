module OGD


using ActivationFunctions, InitializationFunctions, NeuralNetworks, TrainingStructures, RBM,  CostFunctions, FFN, GradientFunctions

export RunOGD

function RunOGD(dataset::DataSet, network::NeuralNetwork, parameters::TrainingParameters)

    tic()
    weight_change_rates = Array{Array{Float64,1},1}()

    sizes = (0, size(dataset.training_output)[2])
    data_values = Array{Float64,2}(sizes)
    predicted_values = Array{Float64,2}(sizes)

    for i in 1:size(dataset.training_input)[1]

        data_values = vcat(data_values, dataset.training_output[i, :]')
        predicted_values = vcat(predicted_values, Feedforward(network, dataset.training_input[i, :]')[end])

        weight_updates = GradientDescentWeightUpdate(network, Array{Float64,2}(dataset.training_input[i,:]'), Array{Float64,2}(dataset.training_output[i,:]'), parameters)

        for l in 1:length(network.layers)
            #momentum_factor[l] = parameters.momentum_rate * momentum_factor[l] - weight_updates[l]
            network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, 1)
        end

        #Weight Change Rate
        if i % 100 == 0
            push!(weight_change_rates, map((x, y) -> mean(x[2:end,:] ./ y[2:end,:]), weight_updates, map(x -> x.weights, network.layers)))
        end

    end


    IS_error = parameters.cost_function.CalculateCost(dataset.training_output, Feedforward(network, dataset.training_input)[end])
    OOS_error = parameters.cost_function.CalculateCost(dataset.testing_output, Feedforward(network, dataset.testing_input)[end])

    IS_accuracy = parameters.is_classification && length(dataset.training_input) > 0 ? PredictionAccuracy(network, dataset.training_input, dataset.training_output) : 0
    OOS_accuracy = parameters.is_classification && length(dataset.training_input) > 0 ? PredictionAccuracy(network, dataset.testing_input, dataset.testing_output) : 0

    epoch_records = [EpochRecord(1, 0.0, IS_error, OOS_error, IS_accuracy, OOS_accuracy, 0.0, toq(), CopyNetwork(network), weight_change_rates, Array{Array{Float64,2},1}())]

    if parameters.verbose
        PrintEpoch(epoch_records[end])
    end

    return (epoch_records, (data_values, predicted_values))
end

function PredictionAccuracy(network, input, output)
    validation_pred = Feedforward(network, input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == output[i,:], 1:size(output)[1])))
    return(correct_predictions/size(output)[1])
end



end
