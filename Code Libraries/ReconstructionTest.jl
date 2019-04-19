workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using DataFrames
using TrainingStructures
#using BSON
#using PlotlyJS
#using DataProcessor
using DataJSETop40

function GenerateLogFluctuations(series, delta, start)
    function LogDiff(x1, x2)
        return log(10, x2) - log(10, x1)
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



function ReconstructPrices2(output_values, data_config, original_prices)

    output_ahead = data_config.prediction_steps[1]
    price_index = (size(original_prices,1) - size(output_values,1) - output_ahead)

    prices = Array{Float64}(original_prices[price_index:price_index+output_ahead,:])
    init_price_length = size(prices, 1)
    prices = vcat(prices, fill(0.0, (size(output_values))))

    multipliers = (10).^Array(output_values)

    for i in 1:size(output_values,1)
        for c in 1:size(prices, 2)
            #prices[(i+init_price_length),c] = prices[(i),c] * multipliers[i,c]
            prices[(i+init_price_length),c] = original_prices[(price_index + i),c] * multipliers[i,c]
        end
    end

    prices
end

jsedata = ReadJSETop40Data()
datasets = jsedata[1:10, [:AGL]] #nothing


#using DataGenerator
data_config = DatasetConfig(1, "test",  5000,  [2],  [0.6],  [0.8, 1.0],  [2], (), nothing)
#datasets = GenerateDataset(data_config.data_seed, data_config.steps, data_config.variation_values)

input,output = ProcessData(datasets, data_config.deltas, data_config.prediction_steps)
#std_output, f1, f2 = StandardizeData(output, nothing)
#norm_output = ReverseStandardization(std_output, f1, f2)
recon_output = ReconstructPrices2(output, data_config, datasets)




#End of ConfigTestsFFNRelu
using DatabaseOps

jsedata = ReadJSETop40Data()
dataset = jsedata[:, [:ACL, :AGL]] #nothing
predictions = RunQuery("select * from prediction_results where configuration_id = 892")

rows = Vector(Array(predictions[:configuration_id] .== 892)) & Vector(Array(predictions[:stock] .== "ACL_2"))
actuals = Array(predictions[rows, :][:actual])
datasetactuals = Array(dataset[(size(dataset, 1) - size(actuals, 1)+1):(end), :ACL])
combarr = Array{Float64,2}(hcat(actuals, datasetactuals))

price_plot = plot(combarr)
savefig(price_plot, "/users/joeldacosta/desktop/PriceCheck.html")
