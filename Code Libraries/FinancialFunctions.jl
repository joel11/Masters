module FinancialFunctions

using DatabaseOps
using DataFrames
using DataProcessor

export GenerateCSCVReturnsCost, GenerateStockReturns, GenerateCSCVReturns, GenerateTotalProfit, GenerateTotalProfitOld, GenerateCSCVStockReturns, GenerateStrategyReturns, SharpeRatio, CalculateProfit, CalculateReturns, CalculateReturnsOneD

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function GenerateCSCVReturnsCost(config_id, original_prices, results)

    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, original_prices)
        original_prices = processed_data[2].original_prices
    end

    if (size(results, 1) == 0 || (maximum(results[:time_step])) == timestep)
        return 0
    end

    num_predictions = (maximum(results[:time_step]))
    finish_t =  size(original_prices, 1) - timestep
    start_t = maximum(data_config.deltas) + 1
    length = finish_t - start_t

    stockreturns = GenerateCSCVStockReturns(results, start_t, finish_t, timestep, original_prices)

    strategyreturns = GenerateCSCVStrategyReturnsCost(stockreturns, timestep)

    return strategyreturns[:, [:time_step, :total_profit_observed, :total_profit_rate]]
end

function GenerateCSCVReturns(config_id, original_prices, results)

    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, original_prices)
        original_prices = processed_data[2].original_prices
    end

    if (size(results, 1) == 0 || (maximum(results[:time_step])) == timestep)
        return 0
    end

    num_predictions = (maximum(results[:time_step]))
    finish_t =  size(original_prices, 1) - timestep
    start_t = maximum(data_config.deltas) + 1
    length = finish_t - start_t

    stockreturns = GenerateCSCVStockReturns(results, start_t, finish_t, timestep, original_prices)

    strategyreturns = GenerateCSCVStrategyReturns(stockreturns, timestep)

    return strategyreturns[:, [:time_step, :total_profit_observed, :total_profit_rate]]
end

function GenerateCSCVStockReturns(step_predictions, start_t, finish_t, timestep, original_prices)

    groups = by(step_predictions, [:stock], df -> [df[:,4:5]])

    for i in 1:size(groups,1)

        df = groups[i,2]
        #names!(df,  [:observed_t2, :expected_t2])
        #df[:time] = start_t:(finish_t-timestep)
        #stock_name_step = get(groups[i,1])
        #stock_name = split(stock_name_step, "_")[1]
        #df[:observed_t] = Array(original_prices[start_t:(finish_t-timestep),parse(stock_name)])

        names!(df,  [:observed_t2, :expected_t2])
        df[:time] = start_t:(size(df,1) + start_t -1)

        stock_name_step = (typeof(groups[i,1]) == String) ? groups[i,1] : get(groups[i,1])

        stock_name = split(stock_name_step, "_")[1]

        #TODO changed this
        df[:observed_t] = Array(original_prices[(start_t):(finish_t),parse(stock_name)])
        #Old: df[:observed_t] = Array(original_prices[(start_t-timestep):(finish_t-timestep),parse(stock_name)])

        df[:trade] = vcat(Int64.(Array(df[:expected_t2]) .> Array(df[:observed_t])))
        df[:trade_benchmark] = vcat(Int64.(Array(df[:observed_t2]) .> Array(df[:observed_t])))

        df[:return_observed] = Array(df[:observed_t2]).*df[:trade]
        df[:return_expected] = Array(df[:expected_t2]).*df[:trade]
        df[:return_benchmark] =  Array(df[:observed_t2]).*df[:trade_benchmark]


        df[:cost] = df[:trade] .* df[:observed_t]
        df[:cost_benchmark] = df[:trade_benchmark] .* df[:observed_t]

        df[:trading_cost] = df[:observed_t] .* (0.1/365*timestep) + df[:observed_t] .* (0.45/100)

        df[:fullcost] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade]
        df[:fullcost_benchmark] = (df[:observed_t] .+ df[:trading_cost]) .* df[:trade_benchmark]

    end

    return groups
end

function GenerateCSCVStrategyReturns(stockreturns, timestep)

    #[:time_step, :total_profit_observed, :total_profit_rate]

    strat_df = DataFrame()

    returns_observed = mapreduce(i -> Array(stockreturns[i,2][:return_observed]), hcat, 1:size(stockreturns,1))
    trade_costs = mapreduce(i -> stockreturns[i,2][:cost], hcat, 1:size(stockreturns,1))

    total_trade_costs = fill(0.0, (size(trade_costs, 1)))
    total_returns_observed = fill(0.0, (size(returns_observed, 1)))

    for i in 1:size(total_returns_observed, 1)
        total_returns_observed[i] = sum(returns_observed[i, :])
        total_trade_costs[i] = sum(trade_costs[i, :])
    end

    strat_df[:total_returns_observed] = total_returns_observed
    strat_df[:total_trade_costs] = total_trade_costs

    #strat_df[:total_returns_observed] = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    #strat_df[:total_trade_costs] = mapreduce(i -> sum(trade_costs[i, :]), vcat, 1:size(trade_costs, 1))
    #test = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    #println(all(test .== total_returns_observed))


    strat_df[:time_step] = collect(1:size(strat_df,1))
    strat_df[:total_profit_observed] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:total_profit_rate] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs]) ./ strat_df[:total_trade_costs]

    return strat_df
end

function GenerateCSCVStrategyReturnsCost(stockreturns, timestep)

    #[:time_step, :total_profit_observed, :total_profit_rate]

    strat_df = DataFrame()

    returns_observed = mapreduce(i -> Array(stockreturns[i,2][:return_observed]), hcat, 1:size(stockreturns,1))
    trade_costs = mapreduce(i -> stockreturns[i,2][:fullcost], hcat, 1:size(stockreturns,1))

    total_trade_costs = fill(0.0, (size(trade_costs, 1)))
    total_returns_observed = fill(0.0, (size(returns_observed, 1)))

    for i in 1:size(total_returns_observed, 1)
        total_returns_observed[i] = sum(returns_observed[i, :])
        total_trade_costs[i] = sum(trade_costs[i, :])
    end

    strat_df[:total_returns_observed] = total_returns_observed
    strat_df[:total_trade_costs] = total_trade_costs

    #strat_df[:total_returns_observed] = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    #strat_df[:total_trade_costs] = mapreduce(i -> sum(trade_costs[i, :]), vcat, 1:size(trade_costs, 1))
    #test = mapreduce(i -> sum(returns_observed[i, :]), vcat, 1:size(returns_observed, 1))
    #println(all(test .== total_returns_observed))


    strat_df[:time_step] = collect(1:size(strat_df,1))
    strat_df[:total_profit_observed] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:total_profit_rate] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs]) ./ strat_df[:total_trade_costs]

    return strat_df
end

function GenerateStockReturns(step_predictions, start_t, finish_t, timestep, original_prices)

    groups = by(step_predictions, [:stock], df -> [df[:,4:5]])

    for i in 1:size(groups,1)

        df = groups[i,2]

        names!(df,  [:observed_t2, :expected_t2])
        df[:time] = start_t:(size(df,1) + start_t -1)

        stock_name_step = (typeof(groups[i,1]) == String) ? groups[i,1] : get(groups[i,1])

        stock_name = split(stock_name_step, "_")[1]
        df[:observed_t] = Array(original_prices[(start_t-timestep):(finish_t-timestep),parse(stock_name)])

        df[:trade] = vcat(Int64.(Array(df[:expected_t2]) .> Array(df[:observed_t])))
        df[:trade_benchmark] = vcat(Int64.(Array(df[:observed_t2]) .> Array(df[:observed_t])))

        df[:return_observed] = Array(df[:observed_t2]).*df[:trade]
        df[:return_expected] = Array(df[:expected_t2]).*df[:trade]
        df[:return_benchmark] =  Array(df[:observed_t2]).*df[:trade_benchmark]


        df[:cost] = df[:trade] .* df[:observed_t]
        df[:cost_benchmark] = df[:trade_benchmark] .* df[:observed_t]

        df[:trading_cost] = df[:observed_t] .* (0.1/365*timestep) + df[:observed_t] .* (0.45/100)

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

    #strat_df[:daily_rates_observed] = vcat(-strat_df[1:timestep, :total_trade_costs],
    #    -strat_df[(timestep+1):end, :total_trade_costs] + strat_df[1:(end-2), :total_returns_observed])

    #strat_df[:daily_rates_observed_fullcosts] = vcat(-strat_df[1:timestep, :total_full_costs],
    #    -strat_df[(timestep+1):end, :total_full_costs] + strat_df[1:(end-2), :total_returns_observed])

    strat_df[:cumulative_profit_observed] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:cumulative_profit_observed_fullcosts] = cumsum(strat_df[:total_returns_observed] .- strat_df[:total_full_costs])
    strat_df[:cumulative_profit_observed_benchmark] = cumsum(strat_df[:total_returns_benchmark] .- strat_df[:total_trade_costs_benchmark])
    strat_df[:cumulative_profit_observed_benchmark_fullcosts] = cumsum(strat_df[:total_returns_benchmark] .- strat_df[:total_full_costs_benchmark])

    strat_df[:cumulative_observed_rate] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_expected_rate] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_trade_costs])
    strat_df[:cumulative_benchmark_rate] = cumsum(strat_df[:total_returns_benchmark]) ./ cumsum(strat_df[:total_trade_costs_benchmark])

    strat_df[:cumulative_observed_rate_fullcost] = cumsum(strat_df[:total_returns_observed]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_expected_rate_fullcost] = cumsum(strat_df[:total_returns_expected]) ./ cumsum(strat_df[:total_full_costs])
    strat_df[:cumulative_benchmark_rate_fullcost] = cumsum(strat_df[:total_returns_benchmark]) ./ cumsum(strat_df[:total_full_costs_benchmark])

    strat_df[:time_step] = collect(1:size(strat_df,1))

    strat_df[:total_profit_observed] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs])
    strat_df[:total_profit_observed_fullcosts] = (strat_df[:total_returns_observed] .- strat_df[:total_full_costs])

    strat_df[:total_profit_rate] = (strat_df[:total_returns_observed] .- strat_df[:total_trade_costs]) ./ strat_df[:total_trade_costs]
    strat_df[:total_profit_rate_fullcost] = (strat_df[:total_returns_observed] .- strat_df[:total_full_costs]) ./ strat_df[:total_full_costs]

    return strat_df
end

end
