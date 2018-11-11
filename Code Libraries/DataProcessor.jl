push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using DataGenerator, FFN

function LogDiff(x1, x2)
    return log(e, x2) - log(e, x1)
end

function GenerateLogFluctuations(series, delta, start)
    fluctuations = []

    for i in start:length(series)
        push!(fluctuations, LogDiff(series[i-delta], series[i]))
    end

    return fluctuations
end

function ProcessSeriesDataset(price_series::Dict, deltas)
    new_series = Dict()
    start_point = maximum(deltas) + 1

    for r in price_series
        for d in deltas
            new_series[string(r[1], "_", d)] = GenerateLogFluctuations(r[2], d, start_point)
        end
        #new_series[string(r[1], "_0")] = r[2][start_point:end]
    end

    return new_series
end

function FormatDataset(seed, time_steps, deltas, prediction_steps, partition_one, partition_two)

    seed  = 1
    time_steps = 3650
    deltas = [1, 7, 30]
    prediction_steps = [1, 7]
    partition_one = 0.9
    partition_two = 0.95



    raw_data = GenerateDataset(seed, time_steps)
    delta_data = ProcessSeriesDataset(raw_data,deltas)

    n_stocks = 2#length(raw_data)
    #prediction_steps = [1,3]

    record_names = []
    for s in 1:n_stocks
        for d in deltas
            push!(record_names, string("stock", s, "_", d))
        end
    end

    data = reduce(hcat, map(x -> delta_data[x], record_names))'

    groups = map(x -> data[(length(deltas)*(x-1)+1):(length(deltas)*(x-1)+length(prediction_steps)), :], 1:n_stocks)
    output = reduce(vcat, map(data -> reduce(hcat,map(x -> data[x, (1+prediction_steps[x]):end][1:(size(data)[2]-maximum(prediction_steps))] ,1:length(prediction_steps)))', groups))
    input = data[:, (1:size(output)[2])]


    sp1 = Int64.(round(size(input)[2]*partition_one))
    sp2 = Int64.(round(size(input)[2]*partition_two))
    train_input = (input[:, 1:sp1])
    test_input = (input[:, (sp1+1):sp2])
    validation_input = (input[:, (sp2+1):end])

    train_output = (output[:, 1:sp1])
    test_output = (output[:, (sp1+1):sp2])
    validation_output = (output[:, (sp2+1):end])

    sd = DataSet(train_input', test_input', validation_input', train_output', test_output', validation_output')

    return sd
end

function GenerateEncodedSGDDataset(dataset, encoder_network)
    training_input = Feedforward(encoder_network, dataset.training_input)[end]
    testing_input = Feedforward(encoder_network, dataset.testing_input)[end]
    validation_input = Feedforward(encoder_network, dataset. validation_input)[end]

    nd = DataSet(training_input, testing_input, validation_input, dataset.training_output, dataset.testing_output, dataset.validation_output)
end

function ValidateGeneration(price_series, deltas, num_stocks)
    for sn in 1:num_stocks
        for d in deltas

            s = price_series[string("stock", sn)]
            sd = new_series[string("stock", sn, "_", d)]

            correct = true

            for i in (1+d):length(sd)
                correct = correct && LogDiff(s[(i-d)], s[i]) == sd[i]
            end

            println(sn, " ", d, " ", correct)

        end
    end
end

#price_series = GenerateDataset(1)
#deltas = [1, 30, 180]
#new_series = ProcessSeriesDataset(price_series, deltas)
#ValidateGeneration(price_series, deltas, 9)
