workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using DataFrames
using TrainingStructures
using BSON
using PlotlyJS
using DataProcessor

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

function ReverseStandardization(data, pv1, pv2)

    newds = DataFrame()
    #colnames = names(data)

    for c in 1:length(pv1)
        newds[parse(string(c))] = data[:,c] .* (pv2[c]) .+ pv1[c]
    end

    return newds
end

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

    input = input_data[start_point:end_point, :]
    output = output_data[start_point:end_point, :]

    return (input,output)
end


function ReconstructPrices(output_values, data_config, original_prices, output_ahead)

    #output_ahead = data_config.prediction_steps[1]
    price_index = (size(original_prices,1) - size(output_values,1) - output_ahead + 1)

    prices = Array{Float64}(original_prices[price_index:price_index+output_ahead-1,:])
    init_price_length = size(prices, 1)
    prices = vcat(prices, fill(0.0, (size(output_values))))

    multipliers = (10).^Array(output_values)

    for i in 1:size(output_values,1)
        for c in 1:size(prices, 2)
            prices[(i+init_price_length),c] = prices[(i),c] * multipliers[i,c]
        end
    end

    prices
end

actuals = BSON.load(string("MidOutputActuals.bson"))[:actualvals]
predicteds = BSON.load(string("MidOutputPredicted.bson"))[:predictedvals]
prices = BSON.load(string("MidOutputOriginal.bson"))[:prices]
config = BSON.load(string("MidOutputConfig.bson"))[:configvals]


actualoneprices = ReconstructPrices(actuals, config, prices, 1)
predictedoneprices = ReconstructPrices(predicteds, config, prices, 1)
actualtwoprices = ReconstructPrices(actuals, config, prices, 2)
predictedtwoprices = ReconstructPrices(predicteds, config, prices, 2)

df = DataFrame()
df[:actualone] = actualoneprices[:,1]
df[:actualtwo] = actualtwoprices[2:end,1]
df[:predictedone] = predictedoneprices[:,1]
df[:predictedtwo] = predictedtwoprices[2:end,1]

price_plot = plot(Array(df), name = ["stock 1" "stock 2" "stock 3" "stock 4" "stock 5" "stock 6"])
savefig(price_plot, "/users/joeldacosta/desktop/Recreategraphs.html")

###################################################################################################
original_prices = prices
output_ahead = 2
output_values = predicteds
data_config = config

price_index = (size(original_prices,1) - size(output_values,1) - output_ahead + 1)

prices = Array{Float64}(original_prices[price_index:price_index+output_ahead-1,:])
init_price_length = size(prices, 1)
prices = vcat(prices, fill(0.0, (size(output_values))))

multipliers = (10).^Array(output_values)

i = 1
c = 1
for i in 1:size(output_values,1)
    for c in 1:size(prices, 2)
        #prices[(i+init_price_length),c] = prices[(i),c] * multipliers[i,c]
        prices[(i+init_price_length),c] = original_prices[(price_index + i),c] * multipliers[i,c]
    end
end

prices








#using DataGenerator
#seed = abs(Int64.(floor(randn()*100)))
#variation_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
#data_config = DatasetConfig(seed, "test",  5000,  [1],  [0.6],  [0.8, 1.0],  [2], variation_pairs, StandardizeData)
#datasets = GenerateDataset(data_config.data_seed, data_config.steps, data_config.variation_values)

#input,output = ProcessData(datasets, data_config.deltas, data_config.prediction_steps)
#std_output, f1, f2 = StandardizeData(output)
#norm_output = ReverseStandardization(std_output, f1, f2)
#recon_output = ReconstructPrices(norm_output, data_config, datasets)

#bson("MidOutputActuals.bson", Dict(:actualvals => Array(deprocessed_actual)))
#bson("MidOutputPredicted.bson", Dict(:predictedvals => Array(deprocessed_predicted)))
#bson("MidOutputOriginal.bson", Dict(:prices => Array(ogd_data.original_prices)))
#bson("MidOutputConfig.bson", Dict(:configvals => ep.data_config))
