workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")


using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess

export RunFFNConfigurationTest, RunSAEConfigurationTest, PrepareData

function NullScaling(data, parameters)
    return (data, [], [])
end

function PrepareData(data_config, dataset)
    data_raw = dataset == nothing ? GenerateDataset(data_config.data_seed, data_config.steps, data_config.variation_values) : dataset
    processed_data = ProcessData(data_raw, data_config.deltas, data_config.prediction_steps)
    standardized_data = map(x -> data_config.scaling_function(x, data_config), processed_data)
    data_splits = map(df -> SplitData(df[1], data_config.process_splits), standardized_data)

    saesgd_data = CreateDataset(data_raw, data_splits[1][1], data_splits[2][1], [1.0], standardized_data[1][2], standardized_data[1][3], standardized_data[2][2], standardized_data[2][3])
    ogd_data = CreateDataset(data_raw, data_splits[1][2], data_splits[2][2], [1.0], standardized_data[1][2], standardized_data[1][3], standardized_data[2][2], standardized_data[2][3])

    return(saesgd_data, ogd_data)
end

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

srand(1234567891234567)
seed = abs(Int64.(floor(randn()*100)))
ds = abs(Int64.(floor(randn()*100)))
var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
data_config = DatasetConfig(ds, "test",  5000,  [1,7],  [0.6],  [0.8, 1.0],  [2], var_pairs, LimitedStandardizeData)

processed_data = PrepareData(data_config, nothing)
#recon_output = ReverseStandardization(processed_data[2].training_output, processed_data[2].output_processingvar1, processed_data[2].output_processingvar2)
#output_values = recon_output
original_prices = processed_data[2].original_prices

#ReconstructPrices(output_values, data_config, original_prices)

#size(output_values)
#size(original_prices)

#deprocessed is (599, 12),
#prices are 5001, 6: stock1, stock2, stock3 ...

#Recon Testing##################################################################

#config 109

function GenerateStockReturns(step_predictions, start_t, finish_t, timestep, original_prices)

    groups = by(step_predictions, [:stock], df -> [df[:,4:5]])

    for i in 1:size(groups,1)

        df = groups[i,2]
        names!(df,  [:observed_t2, :expected_t2])
        df[:time] = start_t:(finish_t-timestep)
        stock_name_step = get(groups[i,1])
        stock_name = split(stock_name_step, "_")[1]
        df[:observed_t] = Array(original_prices[start_t:(finish_t-timestep),parse(stock_name)])

        df[:trade] = vcat(Int64.(Array(df[:expected_t2]) .> Array(df[:observed_t])))
        df[:perfect_trade] = vcat(Int64.(Array(df[:observed_t2]) .> Array(df[:observed_t])))

        df[:trade_cost] = df[:trade] .* df[:observed_t]
        df[:perfect_trade_cost] = df[:perfect_trade] .* df[:observed_t]

        df[:return_observed_amount] = Array(df[:observed_t2]).*df[:trade]
        df[:return_expected_amount] = Array(df[:expected_t2]).*df[:trade]
        df[:return_perfect_amount] =  Array(df[:observed_t2]).*df[:perfect_trade]

        #df[:return_observed_rate] = (Array(df[:observed_t2])./Array(df[:observed_t])).*df[:trade]
        #df[:return_observed_amount] = (Array(df[:observed_t2]) .- Array(df[:observed_t])).*df[:trade]

        #df[:return_expected_rate] = (Array(df[:expected_t2])./Array(df[:observed_t])).*df[:trade]
        #df[:return_expected_amount] = (Array(df[:expected_t2]) .- Array(df[:observed_t])).*df[:trade]

        #df[:return_perfect_amount] = (Array(df[:observed_t2]) .- Array(df[:observed_t])).*df[:perfect_trade]
    end

    return groups
end


function GenerateStrategyReturns(stockreturns)

    strat_df = DataFrame()

    observed_returns = mapreduce(i -> Array(stockreturns[i,2][:return_observed_amount]), hcat, 1:size(stockreturns,1))
    expected_returns = mapreduce(i -> Array(stockreturns[i,2][:return_expected_amount]), hcat, 1:size(stockreturns,1))
    perfect_returns = mapreduce(i -> Array(stockreturns[i,2][:return_perfect_amount]), hcat, 1:size(stockreturns,1))

    trade_costs = mapreduce(i -> stockreturns[i,2][:trade_cost], hcat, 1:size(stockreturns,1))
    perfect_trade_costs = mapreduce(i -> stockreturns[i,2][:perfect_trade_cost], hcat, 1:size(stockreturns,1))

    strat_df[:total_observed_returns] = mapreduce(i -> sum(observed_returns[i, :]), vcat, 1:size(observed_returns, 1))
    strat_df[:total_expected_returns] = mapreduce(i -> sum(expected_returns[i, :]), vcat, 1:size(expected_returns, 1))
    strat_df[:total_perfect_returns] = mapreduce(i -> sum(perfect_returns[i, :]), vcat, 1:size(perfect_returns, 1))
    strat_df[:total_trade_costs] = mapreduce(i -> sum(trade_costs[i, :]), vcat, 1:size(trade_costs, 1))
    strat_df[:total_perfect_trade_costs] = mapreduce(i -> sum(perfect_trade_costs[i, :]), vcat, 1:size(perfect_trade_costs, 1))

    strat_df[:return_expected_rate] = strat_df[:total_expected_returns] ./ strat_df[:total_trade_costs]
    strat_df[:return_observed_rate] = strat_df[:total_observed_returns] ./ strat_df[:total_trade_costs]

    strat_df[:return_expected_rate][isnan.(strat_df[:return_expected_rate])] = 0
    strat_df[:return_observed_rate][isnan.(strat_df[:return_observed_rate])] = 0

    strat_df[:cumulative_return_expected] = cumsum(strat_df[:total_expected_returns] - strat_df[:total_trade_costs])
    strat_df[:cumulative_return_observed] = cumsum(strat_df[:total_observed_returns] - strat_df[:total_trade_costs])

    strat_df[:cumulative_return_perfect] = cumsum(strat_df[:total_perfect_returns] - strat_df[:total_perfect_trade_costs])

    return strat_df
end




using DatabaseOps
using PlotlyJS

timestep = 2
start_t = 3001
finish_t = 5001
original_prices
db = SQLite.DB("database_test.db")
results = SQLite.query(db, "select * from prediction_results where configuration_id = 109")

stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
stratreturns = GenerateStrategyReturns(stockreturns)
stockreturns[1,2]
#rets = hcat(stratreturns[:,8].*100,stratreturns[:,9].*100,stratreturns[:,10].*100)
rets = hcat(stratreturns[:,8], stratreturns[:,9],stratreturns[:,10])

price_plot = plot(rets)
savefig(price_plot, "/users/joeldacosta/desktop/PriceRecon.html")

sum(stratreturns[:total_perfect_trade_costs])

sum(stratreturns[:total_perfect_returns])/sum(stratreturns[:total_perfect_trade_costs])*100
