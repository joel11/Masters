#=
function WriteMMSTradeReturnsBaseOld(config_ids, dataset, financial_returns_function, db_write_function)

    ids = TransformConfigIDs(config_ids)

    backtest_vals = RunQuery("select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from backtest_results where configuration_id in ($ids)")
    prediction_vals = RunQuery("select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from prediction_results where configuration_id in ($ids)")

    backtest_length = maximum(Array(backtest_vals[:time_step]))

    for i in 1:size(prediction_vals,2)
        prediction_vals[:, i] = Array(prediction_vals[:, i])
        backtest_vals[:, i] = Array(backtest_vals[:, i])
    end

    prediction_vals[:time_step] = Array(prediction_vals[:time_step]) .+ backtest_length

    all_vals = backtest_vals
    append!(all_vals, prediction_vals)

    aggregated_returns = DataFrame()
    aggregated_returns[:configuration_id] = Array{Int64}(0)
    aggregated_returns[:time_step] = Array{Int64}(0)
    aggregated_returns[:total_profit_observed] = Array{Float64}(0)
    aggregated_returns[:total_profit_rate] = Array{Float64}(0)

    prediction_length = maximum(Array(all_vals[:time_step]))
    error_count = 0
    configs = unique(Array(all_vals[:configuration_id]))

    for c in configs
        try
            config_predictionvals = all_vals[Array(Array(all_vals[:configuration_id]) .== c),:]
            model_returns = financial_returns_function(c, dataset, config_predictionvals)
            model_returns[:configuration_id] = c
            model_returns = model_returns[:,[:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]

            if size(model_returns, 1) != prediction_length
               difference = prediction_length - size(model_returns, 1)
               max_timestep = maximum(model_returns[:time_step])
               zero_df = DataFrame()
               zero_df[:time_step] = (max_timestep + 1):(max_timestep + difference)
               zero_df[:configuration_id] = c
               zero_df[:total_profit_observed] = 0
               zero_df[:total_profit_rate] = 0
               zero_df = zero_df[:, [:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]
               model_returns = vcat(zero_df, model_returns)
            end

            append!(aggregated_returns, model_returns)
        catch y
            error_count = error_count + 1
            println(string("Error on: ", c))
            println(y)
            continue
        end
    end

    println("Error Count: $error_count")
    db_write_function(aggregated_returns)
end
=#

