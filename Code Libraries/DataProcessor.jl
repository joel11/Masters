module DataProcessor

using DataGenerator, FFN, DataFrames, TrainingStructures

export ReconstructPrices,LimitedNormalizeData, LimitedStandardizeData, ReverseStandardization, ReverseNormalization, GenerateRandomisedDataset, SplitData, CreateDataset, ProcessData, GenerateEncodedSGDDataset, GenerateEncodedOGDDataset, StandardizeData, NormalizeData, ReverseFunctions

function ReconstructPrices(output_values, data_config, original_prices)

    output_ahead = data_config.prediction_steps[1]
    price_index = (size(original_prices,1) - size(output_values,1) - output_ahead) +1

    prices = Array{Float64}(original_prices[price_index:price_index+output_ahead-1,:])
    init_price_length = size(prices, 1)
    prices = vcat(prices, fill(0.0, (size(output_values))))

    multipliers = e.^Array(output_values)

    for i in 1:size(output_values,1)
        for c in 1:size(prices, 2)
            prices[(i+init_price_length),c] = prices[(i),c] * multipliers[i,c]
        end
    end

    prices
end

function LimitedStandardizeData(data, parameters)

    limit_point = Int64(floor(size(data,1) * parameters.process_splits[1]))

    means = map(c -> mean(data[1:limit_point,c]), 1:size(data,2))
    stds = map(c -> std(data[1:limit_point,c]), 1:size(data, 2))

    new_data = DataFrame()
    cols = names(data)

    for i in 1:size(data, 2)
        new_data[cols[i]] = (data[:,i] - means[i]) / stds[i]
    end

    return (new_data, means, stds)
end

function LimitedNormalizeData(data, parameters)
    maxes = []
    mins = []

    limit_point = Int64(floor(size(data,1) * parameters.process_splits[1]))
    new_data = DataFrame()
    cols = names(data)

    for i in 1:size(data, 2)
        min = minimum(data[1:limit_point,i])
        max = maximum(data[1:limit_point,i])
        new_data[cols[i]]  = (data[cols[i]] .- min) ./ (max .- min)
        push!(maxes, max)
        push!(mins, min)
    end

    return (new_data, maxes, mins)
end

function NullScaling(data, parameters)
    return (data, [], [])
end

function StandardizeData(data, parameters)
    means = map(c -> mean(data[:,c]), 1:size(data,2))
    stds = map(c -> std(data[:,c]), 1:size(data, 2))

    new_data = DataFrame()
    cols = names(data)

    for i in 1:size(data, 2)
        new_data[cols[i]] = (data[:,i] - means[i]) / stds[i]
    end

    return (new_data, means, stds)
end

function NormalizeData(data, parameters)
    maxes = []
    mins = []

    new_data = DataFrame()
    cols = names(data)

    for i in 1:size(data, 2)
        min = minimum(data[:,i])
        max = maximum(data[:,i])
        new_data[cols[i]]  = (data[cols[i]] .- min) ./ (max .- min)
        push!(maxes, max)
        push!(mins, min)
    end

    return (new_data, maxes, mins)
end

function ReverseNormalization(data, pv1, pv2)

    newds = DataFrame()
    #colnames = names(data)

    for c in 1:length(pv1)
        newds[parse(string(c))] = data[:,c] .* (pv1[c] - pv2[c]) .+ pv2[c]
    end

    return newds
end

function ReverseStandardization(data, pv1, pv2)

    newds = DataFrame()
    #colnames = names(data)

    for c in 1:length(pv1)
        newds[parse(string(c))] = data[:,c] .* (pv2[c]) .+ pv1[c]
    end

    return newds
end

function GenerateRandomisedDataset(input_data, output_data, parameters::TrainingParameters)

    order = randperm(size(input_data, 1))

    split_point = Int64(floor(length(order) * parameters.training_splits[1]))
    training_indices = order[1:split_point]
    testing_indices = order[(split_point + 1): end]

    return DataSet(nothing,
                    input_data[training_indices,:],
                    input_data[testing_indices,:],
                    output_data[training_indices,:],
                    output_data[testing_indices,:],
                    nothing, nothing, nothing, nothing)
end

function GenerateLogFluctuations(series, delta, start)
    function LogDiff(x1, x2)
        return log(e, x2) - log(e, x1)
    end

    fluctuations = []

    for i in start:length(series)
        push!(fluctuations, LogDiff(series[i-delta], series[i]))
    end

    return fluctuations
end

function GenerateLogDataset(data)
    ldf = DataFrame()

    for p in names(data)
        ldf[parse(string(p, "_log"))]  = GenerateLogFluctuations(data[p], 1, 2)
    end

    return ldf
end

function SplitData(data, partition_percentages)
    partition_points = map(x -> Int64.(round(size(data)[1]*x)), [0.0; partition_percentages; 1.0])
    partitions = map(i -> data[((partition_points[i]+1):partition_points[(i+1)]), :], 1:(length(partition_points)-1))
    return partitions
end

function CreateDataset(original_prices, input_data, output_data, partition_percentages, input_processvar1, input_processvar2, output_processvar1, output_processvar2)
    input_splits = SplitData(input_data, partition_percentages)
    output_splits = SplitData(output_data, partition_percentages)

    sd = DataSet(original_prices, (input_splits[1]), (input_splits[2]), (output_splits[1]), (output_splits[2]), input_processvar1, input_processvar2, output_processvar1, output_processvar2)
    return (sd)
end

function ProcessData(raw_data, deltas, prediction_steps)

    function pastsum(window, index, series)
        return index < window ? NA :  sum(series[(index + 1 - window):index])
    end

    function futuresum(window, index, series)
        return ((index + window) > length(series)) ? NA : sum(series[(1+index):(index+window)])
    end

    function lagset(pairs, log_data, sumfunction)
        ldf = DataFrame()

        for dp in pairs
            fluctuations = log_data[:,parse(string(dp[1], "_log"))]
            windowsums = map(x -> sumfunction(dp[2], x, fluctuations), 1:length(fluctuations))
            ldf[parse(string(dp[1], "_", dp[2]))] = windowsums
        end

        return ldf
    end

    log_data = GenerateLogDataset(raw_data)

    delta_pairs = mapreduce(x -> map(y -> (x, y), deltas), vcat, names(raw_data))
    pred_pairs = mapreduce(x -> map(y -> (x, y), prediction_steps), vcat, names(raw_data))

    input_data = lagset(delta_pairs, log_data, pastsum)
    output_data = lagset(pred_pairs, log_data, futuresum)

    start_point = minimum(findin(complete_cases(input_data), true))
    end_point = maximum(findin(complete_cases(output_data), true))

    return (input_data[start_point:end_point, :],output_data[start_point:end_point, :])
end

function GenerateEncodedSGDDataset(dataset, encoder_network)
    training_input = Feedforward(encoder_network, dataset.training_input)[end]
    testing_input = Feedforward(encoder_network, dataset.testing_input)[end]
    #validation_input = size(dataset.validation_input)[1] > 0 ? Feedforward(encoder_network, dataset.validation_input)[end] : Array{Any}()

    #nd = DataSet(training_input, testing_input, validation_input, dataset.training_output, dataset.testing_output, dataset.validation_output)
    nd = DataSet(dataset.original_prices, DataFrame(training_input), DataFrame(testing_input), dataset.training_output, dataset.testing_output, nothing, nothing, nothing, nothing)
end

function GenerateEncodedOGDDataset(dataset, encoder_network)
    training_input = Feedforward(encoder_network, dataset.training_input)[end]
    return DataSet(dataset.original_prices, training_input, DataFrame(), dataset.training_output, DataFrame(), nothing, nothing, nothing, nothing)
end

const ReverseFunctions = Dict{Function,Function}(StandardizeData=>ReverseStandardization,
    LimitedStandardizeData=>ReverseStandardization,
    NormalizeData=> ReverseNormalization,
    LimitedNormalizeData => ReverseNormalization)

end
