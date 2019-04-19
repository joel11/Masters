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
using PlotlyJS
#=
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
end=#

#=
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
        df[:trade_perfect] = vcat(Int64.(Array(df[:observed_t2]) .> Array(df[:observed_t])))

        df[:return_observed] = Array(df[:observed_t2]).*df[:trade]
        df[:return_expected] = Array(df[:expected_t2]).*df[:trade]
        df[:return_perfect] =  Array(df[:observed_t2]).*df[:trade_perfect]


        df[:cost] = df[:trade] .* df[:observed_t]
        df[:cost_perfect] = df[:trade_perfect] .* df[:observed_t]

        df[:trading_cost] = df[:observed_t] .* (0.1/365*2) + df[:observed_t] .* (0.45/100)

        df[:fullcost] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade]
        df[:fullcost_perfect] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade_perfect]

    end

    return groups
end

function GenerateStrategyReturns(stockreturns, timestep)

    strat_df = DataFrame()

    returns_observed = mapreduce(i -> Array(stockreturns[i,2][:return_observed]), hcat, 1:size(stockreturns,1))
    returns_expected = mapreduce(i -> Array(stockreturns[i,2][:return_expected]), hcat, 1:size(stockreturns,1))
    returns_perfect = mapreduce(i -> Array(stockreturns[i,2][:return_perfect]), hcat, 1:size(stockreturns,1))

    trade_costs = mapreduce(i -> stockreturns[i,2][:cost], hcat, 1:size(stockreturns,1))
    trade_costs_perfect = mapreduce(i -> stockreturns[i,2][:cost_perfect], hcat, 1:size(stockreturns,1))

    full_costs = mapreduce(i -> stockreturns[i,2][:fullcost], hcat, 1:size(stockreturns,1))
    full_costs_perfect = mapreduce(i -> stockreturns[i,2][:fullcost_perfect], hcat, 1:size(stockreturns,1))

    strat_df[:total_returns_observed] = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    strat_df[:total_returns_expected] = mapreduce(i -> sum(returns_expected[i, :]), vcat, 1:size(returns_expected, 1))
    strat_df[:total_returns_perfect] = mapreduce(i -> sum(returns_perfect[i, :]), vcat, 1:size(returns_perfect, 1))
    strat_df[:total_trade_costs] = mapreduce(i -> sum(trade_costs[i, :]), vcat, 1:size(trade_costs, 1))
    strat_df[:total_trade_costs_perfect] = mapreduce(i -> sum(trade_costs_perfect[i, :]), vcat, 1:size(trade_costs_perfect, 1))

    strat_df[:total_full_costs] = mapreduce(i -> sum(full_costs[i, :]), vcat, 1:size(full_costs, 1))
    strat_df[:total_full_costs_perfect] = mapreduce(i -> sum(full_costs_perfect[i, :]), vcat, 1:size(full_costs_perfect, 1))

    strat_df[:daily_rates_observed] = vcat(-strat_df[1:timestep, :total_trade_costs],
        -strat_df[(timestep+1):end, :total_trade_costs] + strat_df[1:(end-2), :total_returns_observed])

    strat_df[:daily_rates_observed_fullcosts] = vcat(-strat_df[1:timestep, :total_full_costs],
        -strat_df[(timestep+1):end, :total_full_costs] + strat_df[1:(end-2), :total_returns_observed])

    strat_df[:cumulative_profit_observed] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:cumulative_profit_observed_fullcosts] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_full_costs])
    strat_df[:cumulative_profit_observed_perfect] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_trade_costs_perfect])
    strat_df[:cumulative_profit_observed_perfect_fullcosts] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_full_costs_perfect])

    strat_df[:cumulative_observed_rate] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_expected_rate] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_perfect_rate] = cumsum(strat_df[:total_returns_perfect]) ./ cumsum(strat_df[:total_trade_costs_perfect])

    strat_df[:cumulative_observed_rate_fullcost] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_expected_rate_fullcost] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_perfect_rate_fullcost] = cumsum(strat_df[:total_returns_perfect]) ./ cumsum(strat_df[:total_full_costs_perfect])

    return strat_df
end
=#



config_id  = 626
original_prices = nothing
ConfigStrategyOutput(626, nothing)
GenerateTotalProfit(626, nothing)
#using DatabaseOps
#
#original_prices
#original_prices2 = DataFrame()
#original_prices2[:stock1] = [5; 6; 7; 6; 8; 10; 9; 9.5; 10; 11; 11; 10]
#original_prices2[:stock2] = [6;5;4;4;4.5;6;7;6.5;8;8.5;10;11]
