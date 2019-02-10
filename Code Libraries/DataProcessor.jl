module DataProcessor

using DataGenerator, FFN, DataFrames, TrainingStructures

export SplitData, CreateDataset, ProcessData, GenerateEncodedSGDDataset, GenerateEncodedOGDDataset, StandardizeData

function StandardizeData(data)
    means = map(c -> mean(data[:,c]), 1:size(data,2))
    stds = map(c -> std(data[:,c]), 1:size(data, 2))

    new_data = DataFrame()
    cols = names(data)

    for i in 1:size(data, 2)
        new_data[cols[i]] = (data[:,i] - means[i]) / stds[i]
    end

    return (new_data, means, stds)
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

function CreateDataset(input_data, output_data, partition_percentages)
    input_splits = SplitData(input_data, partition_percentages)
    output_splits = SplitData(output_data, partition_percentages)

    #sd = DataSet(Array(input_splits[1]), Array(input_splits[2]), Array(input_splits[3]), (output_splits[1]), (output_splits[2]), (output_splits[3]))
    sd = DataSet(Array(input_splits[1]), Array(input_splits[2]), (output_splits[1]), (output_splits[2]), 0, 0)
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
    nd = DataSet(training_input, testing_input, dataset.training_output, dataset.testing_output)
end

function GenerateEncodedOGDDataset(dataset, encoder_network)
    training_input = Feedforward(encoder_network, dataset.training_input)[end]
    #testing_input = Feedforward(encoder_network, dataset.testing_input)[end]
    #validation_input = size(dataset.validation_input)[1] > 0 ? Feedforward(encoder_network, dataset.validation_input)[end] : Array{Any}()

    #nd = DataSet(training_input, testing_input, validation_input, dataset.training_output, dataset.testing_output, dataset.validation_output)
    nd = DataSet(training_input, nothing, dataset.training_output, nothing)
end

end
