#module CSCV

workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using Combinatorics
using StatsBase
using FinancialFunctions
using DatabaseOps
using DataFrames
using DataJSETop40

export GenerateCSCVReturns, RunCSCV, CalculatePBO

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end
#=
function CalculateProfit(actual, predicted, current_actual)

    prof = nothing

    if predicted > current_actual
        prof = actual - current_actual
    else
        prof = 0
    end

    return prof
end

function CalculateReturns(actuals, predicted, timestep)
    ret =  mapreduce(r -> map(c -> CalculateProfit(actuals[r, c], predicted[r, c], actuals[r - timestep, c]), 1:size(actuals)[2]), hcat, (1+timestep):size(actuals)[1])'
    return (ret)
    #return map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])
end
=#

function ProcessCombination(data, training_indices, testing_indices)
    #training_indices = set_pairs[46][1]
    #testing_indices = set_pairs[46][2]

    ## 4
    ## a & b
    #println("4ab")
    #tic()


    train_ind = sort(mapreduce(x -> map(y -> y, x),vcat, training_indices))
    test_ind =  sort(mapreduce(x -> map(y -> y, x),vcat, testing_indices))

    #train_data = data[train_ind, :]
    #test_data = data[test_ind, :]
    #println(toc())

    ## c & d
    #println("4c")
    #tic()
    ##all_ind = sort(vcat(train_ind, test_ind))
    #N = size(data[all_ind, :])[2]
    #all_pvals = fill(0.0, N)
    #for i in 1:N
    #    all_pvals[i] = SharpeRatio(data[all_ind ,i], 0.00)
    #end
    #println(all_pvals)
    #sum(isnan.(all_pvals))

    N = size(data[train_ind, :])[2]
    train_pvals = fill(0.0, N)
    test_pvals = fill(0.0, N)

    for i in 1:N
        train_pvals[i] = SharpeRatio(data[train_ind,i], 0.00)
        test_pvals[i]  = SharpeRatio(data[test_ind ,i], 0.00)
    end

    test_pvals[isnan.(test_pvals)] = 0
    train_pvals[isnan.(train_pvals)] = 0

    #train_pvals = map(x -> SharpeRatio(train_data[:,x], 0.00), 1:N)
    #test_pvals = map(x -> SharpeRatio(test_data[:,x], 0.00), 1:N)
    #println(toc())

    #println("4d")
    #tic()
    r = ordinalrank(train_pvals)
    rbar = ordinalrank(test_pvals)
    #println(toc())

    ## e & f & g
    #println("4efg")
    #tic()
    #N = size(r,1)
    is_index = findin(r, maximum(r))
    oos_val = rbar[is_index][1]
    wc = oos_val / (1 + N)
    lval = wc / (1 - wc)
    lambda_c = log(e, lval)
    #println(toc())
    return lambda_c
end


function RunCSCV(data, splits)

    # data = arr_data
    # 16:    12 870
    # 24: 2 704 156
    # 32:
    # splits = 32
    # nrows = 1280
    # length = Int64.(floor(nrows/splits))
    # ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
    # all = Set(ranges)
    # m = Int64.(splits/2)
    # training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
    # set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)

    ##1 & 2
    println("1 & 2")
    tic()
    nrows = size(data)[1]
    length = Int64.(floor(nrows/splits))

    ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
    all = Set(ranges)
    m = Int64.(splits/2)
    println(toc())

    ##3
    println("3")
    tic()
    training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
    set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)
    println(toc())

    ##4
    println("4")
    tic()
    logitvals = map(sp -> ProcessCombination(data, sp[1], sp[2]), set_pairs)

    println(toc())

    ##5
    println("5")
    tic()
    overfit_dist = map(x -> (x, size(findin(logitvals, x))[1]/(size(logitvals)[1])), sort(unique(logitvals)))

    println(toc())
    return overfit_dist
end

function CalculatePBO(overfit_distribution)
    return mapreduce(x -> (x[1] <= 0 ? x[2] : 0), +, overfit_distribution)
end

function RunGenerations()
    jsedata = ReadJSETop40Data()
    dataset = jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

    range = 28136:29415



    for c in config_starts
        range = c:(c+399)
        println(range)
        tic()
        WriteCSCVReturns(range, dataset)
        println(toc())
    end
end

function WriteCSCVReturns(config_ids, dataset)

    #config_ids = (27992, 27993)

    println("A1")
    tic()

    ids = TransformConfigIDs(config_ids)
    #backtest_query = "select * from backtest_results_index where configuration_id in ($ids) and predicted is not null"
    backtest_query = "select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from backtest_results where configuration_id in ($ids)"
    backtest_vals = RunQuery(backtest_query)

    toc()
    println("A2")
    tic()

    #prediction_query = "select * from prediction_results_index where configuration_id in ($ids) and predicted is not null"
    prediction_query = "select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from prediction_results where configuration_id in ($ids)"
    prediction_vals = RunQuery(prediction_query)

    toc()
    println("B1")
    tic()

    backtest_length = maximum(Array(backtest_vals[:time_step]))

    for i in 1:size(prediction_vals,2)
        prediction_vals[:, i] = Array(prediction_vals[:, i])
        backtest_vals[:, i] = Array(backtest_vals[:, i])
    end


    prediction_vals[:time_step] = Array(prediction_vals[:time_step]) .+ backtest_length

    all_vals = backtest_vals
    append!(all_vals, prediction_vals)

    # for i in 1:2300
    #     ss = all_vals[Array(all_vals[:time_step] .== i), :]
    #     actual = ss[1, :actual]
    #     p1 = ss[1, :predicted]
    #     p2 = ss[2, :predicted]
    #     if (p1 < actual && actual < p2) || (p2 < actual && actual < p1)
    #         println(i)
    #     end
    # end
    # all_vals[Array(all_vals[:time_step] .== 200), :]



    aggregated_returns = DataFrame()
    aggregated_returns[:configuration_id] = Array{Int64}(0)
    aggregated_returns[:time_step] = Array{Int64}(0)
    aggregated_returns[:total_profit_observed] = Array{Float64}(0)
    aggregated_returns[:total_profit_rate] = Array{Float64}(0)

    prediction_length = maximum(Array(all_vals[:time_step]))
    error_count = 0
    configs = unique(Array(all_vals[:configuration_id]))

    toc()
    println("C")
    tic()

    for c in configs
        try

            #c = config_ids[1]

            config_predictionvals = all_vals[Array(Array(all_vals[:configuration_id]) .== c),:]
            model_returns = GenerateCSCVReturns(c, dataset, config_predictionvals)
            model_returns[:configuration_id] = c
            model_returns = model_returns[:,[:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]


            #results = config_predictionvals
            #original_prices = dataset
            #config_id = c
            #GenerateCSCVReturns(config_id, original_prices, results)

            if size(model_returns, 1) != prediction_length
                println("Zeros")
                difference = prediction_length - size(model_returns, 1)
                max_timestep = maximum(model_returns[:time_step])
                zero_df = DataFrame()
                zero_df[:time_step] = (max_timestep + 1):(max_timestep + difference)
                zero_df[:configuration_id] = c
                zero_df[:total_profit_observed] = 0
                zero_df[:total_profit_rate] = 0
                zero_df = zero_df[:, [:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]
                model_returns = vcat(model_returns ,zero_df)
            end

            #aggregated_returns = vcat(aggregated_returns, model_returns)
            append!(aggregated_returns, model_returns)

            #return_data[:,parse(string("iteration_", c))] = Array(model_returns)[:,1]
        catch y
            error_count = error_count + 1
            println(string("Error on: ", c))
            println(y)
            continue
        end
    end

    #aggregated_returns[Array(aggregated_returns[:time_step] .== 100), :]

    toc()

    println("Error Count: $error_count")
    CreateCSCVRecords(aggregated_returns)
end

#=

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

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    #df[:observed_t] =
    #    original_prices[200, :AGL]
    #    Array(original_prices[(start_t-timestep):(finish_t-timestep),parse("AGL")])[100]

    strategyreturns = GenerateCSCVStrategyReturns(stockreturns, timestep)

    return strategyreturns[:, [:time_step, :total_profit_observed, :total_profit_rate]]
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

=#

function ExperimentCSCVProcess(config_ids)

    #ogd_lrs = (0.1,0.05,0.01,0)
    #ogd_dict = Dict()

    #sgd_lrs = (0.5,	0.1,	0.01,	0.001)
    #sgd_dict = Dict()

    #for sgdlr in sgd_lrs

        # cscv_returns = RunQuery("select cv.configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
        #                             from cscv_returns cv
        #                             where configuration_id >= 28136
        #                             and  configuration_id in (
        #                                 select cr.configuration_id
        #                                 from configuration_run cr
        #                                 inner join training_parameters tp on tp.configuration_id = cr.configuration_id and category = 'FFN'
        #                                 where experiment_set_name like 'Iteration10 FFN AGL PBO%'
        #                                 and learning_rate = $sgdlr
        #                                 )
        #                             order by cv.configuration_id
        #                             ")

        cscv_returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                     from cscv_returns
                                     where configuration_id >= 28136
                                     order by configuration_id, time_step")

        cscv_groups = by(cscv_returns, [:configuration_id], df -> [df[:,[:time_step, :total_profit_rate_observed]]])

        maxes = fill(0, (size(cscv_groups,1), 1))

        #i = 1280

        for i in 1:size(cscv_groups,1)
            one_data = cscv_groups[i,2]
            #nullvals = map(i -> isnull(one_data[i, :total_profit_rate_observed]), 1:size(one_data, 1))
            #sum(nullvals)
            #one_data[nullvals, :total_profit_rate_observed] = 0
            zp_ind = Array(one_data[:total_profit_rate_observed]) .!= 0
            if sum(zp_ind) > 0
                maxnon = maximum(Array(one_data[zp_ind, :time_step]))
            else
                maxnon = 0
            end
            maxes[i] = maxnon
        end
        #sum(maxes .< 20)

        arr_data = fill(0.0, (size(cscv_groups[1,2],1), size(cscv_groups,1)))

        for r in 1:size(cscv_groups,1)
            arr_data[:,r] = Array(cscv_groups[r,2][:total_profit_rate_observed])
        end

        inds = Array{Bool}((maxes .> 20)[:,1])

        #sum(inds)
        #############################################################################

        #ltd_arr = arr_data #[1:(end-16), inds]
        ltd_arr = arr_data[:, inds]

        #28200; 29367
        #configids = 28136:29415
        #nulls = configids[!inds]
        #configstring = string("(" , mapreduce(i -> "$i,", string,  nulls)[1:(end-1)], ")")
        #println(configstring)

        #println("new")
        #pbo_dict = Dict()
        #for s in (2, 4, 8, 12, 16)
            #distribution = RunCSCV(ltd_arr, s)
            distribution = RunCSCV(ltd_arr, 16)
            pbo = CalculatePBO(distribution)
        #    pbo_dict[s] = pbo
        #end

        #for s in (2, 4, 8, 12, 16)
        #    println("$s : ", pbo_dict[s])
        #end

        #sgd_dict[sgdlr] = pbo_dict
    #end

    for s in (2, 4, 8, 12, 16)
        println("$s : ", sgd_dict[0.001][s])
    end


    # 2 : 0.5
    # 4 : 0.16666666666666666
    # 8 : 0.014285714285714285
    # 12 : 0.008658008658008658
    # 16 : 0.002564102564102564

    using PlotlyJS
    x0 = map(i -> round(distribution[i][1],1), 1:size(distribution,1))
    y0 = map(i -> distribution[i][2], 1:size(distribution,1))
    trace = bar(;x=x0,y=y0, name="Logit Distribution", bargap=0.1)
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/CSCV AGL.html"))

    return (pbo, mses)
end

function OGDPL()

    cscv_returns = RunQuery("select cv.configuration_id, time_step, ifnull(total_profit_rate_observed, 0) total_profit_rate_observed
                                from cscv_returns cv
                                where configuration_id >= 27523
                                and  configuration_id in (
                                    select cr.configuration_id
                                    from configuration_run cr
                                    inner join training_parameters tp on tp.configuration_id = cr.configuration_id and category = 'FFN'
                                    where experiment_set_name like 'Iteration3 FFN AGL PBO%'
                                    and min_learning_rate = 0.001
                                    and learning_rate = 0.1
                                    )
                                order by cv.configuration_id
                                ")

    cscv_groups = by(cscv_returns, [:configuration_id], df -> [df[:,[:time_step, :total_profit_rate_observed]]])


    # backtest 2339
    # prediction 1561
    # total 3900


end


#vals = ExperimentCSCVProcess(26363:27658)
#config_ids = 26363:27658

#end
