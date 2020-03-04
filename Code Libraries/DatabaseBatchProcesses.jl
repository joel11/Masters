module DatabaseBatchProcesses

using MLBase
using Combinatorics
using StatsBase
using FinancialFunctions
using DatabaseOps
using DataFrames
using DataJSETop40
using CSV

export RunBatchTradeProcess, RunBatchAnalyticsProcess, WriteCSVFileForDSR

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end

function RunBatchTradeProcess(config_ids, is_oos_split_timestep, dataset)

    min = minimum(config_ids)
    max = maximum(config_ids)
	config_starts = min:100:max

    for c in config_starts
        range = c:(c+99)
        println(range)

        WriteOOSTradeCostReturns(range, dataset)
        WriteOOSTradeReturns(range, dataset)
        WriteISTradeReturns(range, dataset)
        WriteISTradeCostReturns(range, dataset)
    end
end

function RunBatchAnalyticsProcess(config_ids, is_oos_split_timestep, dataset)

    min = minimum(config_ids)
    max = maximum(config_ids)
	config_starts = min:100:max

    for c in config_starts
        range = c:(c+99)
        println(range)

        UpdateISProfits(range)
        UpdateOOSProfits(range)
        UpdateOOSCostProfits(range)
        WriteOOSSharpeRatios(range)
        WriteOOSSharpeRatiosCost(range)
        GenerateConfusionDistributions(range, dataset)
    end
end

function UpdateOOSProfits(config_ids)

    ids = TransformConfigIDs(config_ids)

    time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_oos_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    start = get(maximum(time_steps[:min_ts]))
    finish = get(minimum(time_steps[:max_ts]))

    RunQuery("delete from config_oos_pl where configuration_id in ($ids)")

    RunQuery("insert into config_oos_pl (configuration_id, total_pl)
            select configuration_id, sum(total_profit_observed)
            from config_oos_trade_returns
            where configuration_id in ($ids)
            and time_step between $start and $finish
            group by configuration_id")
end

function UpdateOOSCostProfits(config_ids)

    ids = TransformConfigIDs(config_ids)

    time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_oos_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    start = get(maximum(time_steps[:min_ts]))
    finish = get(minimum(time_steps[:max_ts]))

    RunQuery("delete from config_oos_pl_cost where configuration_id in ($ids)")

    RunQuery("insert into config_oos_pl_cost (configuration_id, total_pl)
            select configuration_id, sum(total_profit_observed)
            from config_oos_trade_returns_cost
            where configuration_id in ($ids)
            and time_step between $start and $finish
            group by configuration_id")
end

function UpdateISProfits(config_ids)

    ids = TransformConfigIDs(config_ids)

    time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_is_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    start = get(maximum(time_steps[:min_ts]))
    finish = get(minimum(time_steps[:max_ts]))

    RunQuery("delete from config_is_pl where configuration_id in ($ids)")

    RunQuery("insert into config_is_pl (configuration_id, total_pl)
            select configuration_id, sum(total_profit_observed)
            from config_is_trade_returns
            where configuration_id in ($ids)
            and time_step between $start and $finish
            group by configuration_id")
end

function WriteOOSSharpeRatios(config_ids)

    ids = TransformConfigIDs(config_ids)

    time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_oos_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    start = get(maximum(time_steps[:min_ts]))
    finish = get(minimum(time_steps[:max_ts]))

    RunQuery("delete from config_oos_sharpe_ratio where configuration_id in ($ids)")

    returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                 from config_oos_trade_returns
                                 where configuration_id in ($ids)
                                and time_step between $start and $finish
                                 ")

    returns[:,1] = Array{Int64,1}(returns[:,1])
    returns[:,2] = Array{Int64,1}(returns[:,2])
    returns[:,3] = Array{Float64,1}(returns[:,3])

    ratios = by(returns, :configuration_id, df-> SharpeRatio(df[:total_profit_rate_observed], 0.0))

    CreateSRRecords(ratios)
end

function WriteOOSSharpeRatiosCost(config_ids)

    ids = TransformConfigIDs(config_ids)

    time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_oos_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    start = get(maximum(time_steps[:min_ts]))
    finish = get(minimum(time_steps[:max_ts]))

    RunQuery("delete from config_oos_sharpe_ratio_cost where configuration_id in ($ids)")

    returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                 from config_oos_trade_returns_cost
                                 where configuration_id in ($ids)
                                and time_step between $start and $finish"
                                 )

    returns[:,1] = Array{Int64,1}(returns[:,1])
    returns[:,2] = Array{Int64,1}(returns[:,2])
    returns[:,3] = Array{Float64,1}(returns[:,3])

    ratios = by(returns, :configuration_id, df-> SharpeRatio(df[:total_profit_rate_observed], 0.0))

    CreateSRRecordsCost(ratios)
end

function WriteISTradeCostReturns(config_ids, dataset)
    ids = TransformConfigIDs(config_ids)
    RunQuery("delete from config_is_trade_returns_cost where configuration_id in ($ids)")

    WriteTradeReturnsBase(config_ids, dataset, GenerateTradeReturnsCost, CreateISTradeCostRecords, "backtest_results", true)
end

function WriteISTradeReturns(config_ids, dataset)

    ids = TransformConfigIDs(config_ids)
    RunQuery("delete from config_is_trade_returns where configuration_id in ($ids)")

    WriteTradeReturnsBase(config_ids, dataset, GenerateTradeReturns, CreateISTradeRecords, "backtest_results", true)
end

function WriteOOSTradeCostReturns(config_ids, dataset)
    ids = TransformConfigIDs(config_ids)
    RunQuery("delete from config_oos_trade_returns_cost where configuration_id in ($ids)")

    WriteTradeReturnsBase(config_ids, dataset, GenerateTradeReturnsCost, CreateOOSTradeCostRecords, "prediction_results", false)
end

function WriteOOSTradeReturns(config_ids, dataset)

    ids = TransformConfigIDs(config_ids)
    RunQuery("delete from config_oos_trade_returns where configuration_id in ($ids)")

    WriteTradeReturnsBase(config_ids, dataset, GenerateTradeReturns, CreateOOSTradeRecords, "prediction_results", false)
end

function WriteTradeReturnsBase(config_ids, dataset, financial_returns_function, db_write_function, predictions_table, in_sample)

    ids = TransformConfigIDs(config_ids)

    println("A")
    tic()
    all_vals = RunQuery("select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted
                                from $predictions_table
                                where configuration_id in ($ids)")

    toc()
    println("B")
    tic()

    for i in 1:size(all_vals,2)
        all_vals[:, i] = Array(all_vals[:, i])
    end

    aggregated_returns = DataFrame()
    aggregated_returns[:configuration_id] = Array{Int64}(0)
    aggregated_returns[:time_step] = Array{Int64}(0)
    aggregated_returns[:total_profit_observed] = Array{Float64}(0)
    aggregated_returns[:total_profit_rate] = Array{Float64}(0)

    error_count = 0
    configs = unique(Array(all_vals[:configuration_id]))

    toc()
    println("C")
    tic()

    for c in configs
        try
            config_predictionvals = all_vals[Array(Array(all_vals[:configuration_id]) .== c),:]
            model_returns = financial_returns_function(c, dataset, config_predictionvals, in_sample)
            model_returns[:configuration_id] = c
            model_returns = model_returns[:,[:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]

            append!(aggregated_returns, model_returns)
        catch y
            error_count = error_count + 1
            println(string("Error on: ", c))
            println(y)
            continue
        end
    end

    toc()

    println("Error Count: $error_count")
    db_write_function(aggregated_returns)
end

function GenerateConfusionDistributions(config_ids, original_prices)

    ids = TransformConfigIDs(config_ids)
    RunQuery("delete from config_confusion where configuration_id in ($ids)")
    all_results = RunQuery("select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from prediction_results where configuration_id in ($ids)")

    confusion_df = DataFrame(configuration_id = [], no_trade_perc = [], trade_perc = [], all_perc = [])
    confusion_df[:,1] = Array{Int64}(confusion_df[:,1])
    confusion_df[:,2] = Array{Float64}(confusion_df[:,2])
    confusion_df[:,3] = Array{Float64}(confusion_df[:,3])
    confusion_df[:,4] = Array{Float64}(confusion_df[:,4])

    config_id = config_ids[1]

    for config_id in config_ids
        results = all_results[Array(all_results[:, :configuration_id]) .== config_id, :]

        if size(results, 1) > 0

            num_predictions = get(maximum(results[:time_step]))
            finish_t = size(original_prices, 1)
            start_t = finish_t - num_predictions + 1

            config_results = RunQuery("select * from configuration_run where configuration_id = $config_id")
            sae_id = get(config_results[1, :sae_config_id])
            data_config = ReadSAE(sae_id)[2]
            timestep = data_config.prediction_steps[1]

            results = all_results[Array(all_results[:configuration_id]) .== config_id,:]

            stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)

            trades = mapreduce(i -> stockreturns[i,2][:,[:trade, :trade_benchmark]], vcat, 1:size(stockreturns,1))

            actual =   trades[:,2]
            model =    trades[:,1]

            df = DataFrame(confusmat(2, actual .+ 1, model .+ 1))

            no_trade_perc = df[1,1] / (df[1,1] + df[1,2])
            trade_perc = df[2,2] / (df[2,1] + df[2,2])
            all_perc = (df[1,1] + df[2,2]) / (sum(df[:,1]) + sum(df[:,2]))

            appendDF = DataFrame(configuration_id = [config_id], no_trade_perc = [no_trade_perc], trade_perc = [trade_perc], all_perc = [all_perc])
            appendDF[:,1] = Array{Int64}(appendDF[:,1])
            appendDF[:,2] = Array{Float64}(appendDF[:,2])
            appendDF[:,3] = Array{Float64}(appendDF[:,3])
            appendDF[:,4] = Array{Float64}(appendDF[:,4])

            append!(confusion_df, appendDF)
        end
    end

    CreateConfusionRecords(confusion_df)
end

function WriteCSVFileForDSR(config_ids, filename)
    #This process does assume that all trials were run with the same IS & OOS dataset points (i.e. same split_percentage)
    #config_ids = 28880:53000
    #filename = "/users/joeldacosta/Desktop/dsr_returns_new.csv"

    ids = TransformConfigIDs(config_ids)

    deltas = RunQuery("select distinct(deltas) from dataset_config where configuration_id in ($ids)")
    split_percentage = parse(Float64, get(RunQuery("select distinct(process_splits) from dataset_config where configuration_id in ($ids)")[1,1]))

    minimum_delta = minimum(map(a -> minimum(a), map(x -> map(i -> parse(Int64, i), x), map(d -> ascii.(split(d, ",")), Array(deltas[:,1])))))
    maximum_deltas = map(a -> maximum(a), map(x -> map(i -> parse(Int64, i), x), map(d -> ascii.(split(d, ",")), Array(deltas[:,1]))))

    total_timestep_diff = maximum(maximum_deltas) - minimum(maximum_deltas)
    oos_timestep_diff = Int64.(diff * (1 - split_percentage))

    max_ts = get(RunQuery("select max(time_step) - 1 from config_oos_trade_returns_cost p
                inner join dataset_config d on p.configuration_id = d.configuration_id and deltas not like '%$minimum_delta,%'")[1,1])

    min_ts = get(RunQuery("select min(time_step) + 1 from config_oos_trade_returns_cost p
                inner join dataset_config d on p.configuration_id = d.configuration_id and deltas like '%$minimum_delta,%'")[1,1]) + oos_timestep_diff

    query = "  select configuration_id, total_profit_rate_observed from
                (select p.configuration_id configuration_id, time_step,
                        ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                from config_oos_trade_returns_cost p
                inner join dataset_config d on p.configuration_id = d.configuration_id and deltas not like '%$minimum_delta,%'
                where p.configuration_id in ($ids)
                and time_step < $max_ts
                union all
                select p.configuration_id configuration_id, time_step,
                        ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                from config_oos_trade_returns_cost p
                inner join dataset_config d on p.configuration_id = d.configuration_id and deltas like '%$minimum_delta,%'
                where p.configuration_id in ($ids)
                and time_step > $min_ts
                order by configuration_id, time_step)"

    new_results = RunQuery(query)

    CSV.write(filename, new_results)
end

end
