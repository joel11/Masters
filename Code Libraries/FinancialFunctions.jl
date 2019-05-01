module FinancialFunctions

using DatabaseOps
using ExperimentProcess
using DataFrames
export GenerateTotalProfit, GenerateStockReturns, GenerateStrategyReturns, SharpeRatio, CalculateProfit, CalculateReturns, CalculateReturnsOneD

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function GenerateTotalProfit(config_id, original_prices)
    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id  and predicted is not null")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end


    results = RunQuery("select * from prediction_results where configuration_id = $config_id and predicted is not null")

    if (size(results, 1) == 0 || get(maximum(results[:time_step])) == timestep)
        return 0
    end


    num_predictions = get(maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions - 1

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)


    return strategyreturns[end, :cumulative_profit_observed]
end

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
        df[:trade_benchmark] = vcat(Int64.(Array(df[:observed_t2]) .> Array(df[:observed_t])))

        df[:return_observed] = Array(df[:observed_t2]).*df[:trade]
        df[:return_expected] = Array(df[:expected_t2]).*df[:trade]
        df[:return_benchmark] =  Array(df[:observed_t2]).*df[:trade_benchmark]


        df[:cost] = df[:trade] .* df[:observed_t]
        df[:cost_benchmark] = df[:trade_benchmark] .* df[:observed_t]

        df[:trading_cost] = df[:observed_t] .* (0.1/365*2) + df[:observed_t] .* (0.45/100)

        df[:fullcost] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade]
        df[:fullcost_benchmark] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade_benchmark]

    end

    return groups
end

function GenerateStrategyReturns(stockreturns, timestep)

    strat_df = DataFrame()

    returns_observed = mapreduce(i -> Array(stockreturns[i,2][:return_observed]), hcat, 1:size(stockreturns,1))
    returns_expected = mapreduce(i -> Array(stockreturns[i,2][:return_expected]), hcat, 1:size(stockreturns,1))
    returns_benchmark = mapreduce(i -> Array(stockreturns[i,2][:return_benchmark]), hcat, 1:size(stockreturns,1))

    trade_costs = mapreduce(i -> stockreturns[i,2][:cost], hcat, 1:size(stockreturns,1))
    trade_costs_benchmark = mapreduce(i -> stockreturns[i,2][:cost_benchmark], hcat, 1:size(stockreturns,1))

    full_costs = mapreduce(i -> stockreturns[i,2][:fullcost], hcat, 1:size(stockreturns,1))
    full_costs_benchmark = mapreduce(i -> stockreturns[i,2][:fullcost_benchmark], hcat, 1:size(stockreturns,1))

    strat_df[:total_returns_observed] = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    strat_df[:total_returns_expected] = mapreduce(i -> sum(returns_expected[i, :]), vcat, 1:size(returns_expected, 1))
    strat_df[:total_returns_benchmark] = mapreduce(i -> sum(returns_benchmark[i, :]), vcat, 1:size(returns_benchmark, 1))
    strat_df[:total_trade_costs] = mapreduce(i -> sum(trade_costs[i, :]), vcat, 1:size(trade_costs, 1))
    strat_df[:total_trade_costs_benchmark] = mapreduce(i -> sum(trade_costs_benchmark[i, :]), vcat, 1:size(trade_costs_benchmark, 1))

    strat_df[:total_full_costs] = mapreduce(i -> sum(full_costs[i, :]), vcat, 1:size(full_costs, 1))
    strat_df[:total_full_costs_benchmark] = mapreduce(i -> sum(full_costs_benchmark[i, :]), vcat, 1:size(full_costs_benchmark, 1))

    strat_df[:daily_rates_observed] = vcat(-strat_df[1:timestep, :total_trade_costs],
        -strat_df[(timestep+1):end, :total_trade_costs] + strat_df[1:(end-2), :total_returns_observed])

    strat_df[:daily_rates_observed_fullcosts] = vcat(-strat_df[1:timestep, :total_full_costs],
        -strat_df[(timestep+1):end, :total_full_costs] + strat_df[1:(end-2), :total_returns_observed])

    strat_df[:cumulative_profit_observed] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:cumulative_profit_observed_fullcosts] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_full_costs])
    strat_df[:cumulative_profit_observed_benchmark] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_trade_costs_benchmark])
    strat_df[:cumulative_profit_observed_benchmark_fullcosts] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_full_costs_benchmark])

    strat_df[:cumulative_observed_rate] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_expected_rate] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_benchmark_rate] = cumsum(strat_df[:total_returns_benchmark]) ./ cumsum(strat_df[:total_trade_costs_benchmark])

    strat_df[:cumulative_observed_rate_fullcost] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_expected_rate_fullcost] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_benchmark_rate_fullcost] = cumsum(strat_df[:total_returns_benchmark]) ./ cumsum(strat_df[:total_full_costs_benchmark])

    return strat_df
end

end
