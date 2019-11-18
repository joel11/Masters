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
    println("4ab")
    tic()


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
    println(toc())
    return lambda_c
end


function RunCSCV(data, splits)

    #data = arr_data
    #splits = 4

    #splits = 12
    #nrows = 1561
    #length = Int64.(floor(nrows/splits))
    #ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
    #all = Set(ranges)
    #m = Int64.(splits/2)
    #training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
    #set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)
    #sp = set_pairs[1]
    #train_ind = sort(mapreduce(x -> map(y -> y, x),vcat, sp[1]))
    #test_ind =  sort(mapreduce(x -> map(y -> y, x),vcat, sp[2]))
    #train_data = data[train_ind, :]
    #test_data = data[test_ind, :]

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

    #config_starts = 1713:400:(26863+200)
    #config_starts = 26864:400:(28879+200)

    config_starts = 1713:500:28879

    for c in config_starts
        range = c:(c+499)
        println(range)
        tic()
        WriteCSCVReturns(range, dataset)
        println(toc())
    end
end

function WriteCSCVReturns(config_ids, dataset)

    ids = TransformConfigIDs(config_ids)
    query = "select * from prediction_results where configuration_id in ($ids) and predicted is not null"
    all_predictionvals = RunQuery(query)

    aggregated_returns = DataFrame()
    prediction_length = get(maximum(all_predictionvals[:time_step]))
    error_count = 0
    configs = unique(Array(all_predictionvals[:configuration_id]))

    for c in configs
        println(c)
        try
            config_predictionvals = all_predictionvals[Array(Array(all_predictionvals[:configuration_id]) .== c),:]

            model_returns = GenerateCSCVReturns(c, dataset, config_predictionvals)
            model_returns[:configuration_id] = c
            model_returns = model_returns[:,[:configuration_id, :time_step, :total_profit_observed, :total_profit_rate]]

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

            aggregated_returns = vcat(aggregated_returns, model_returns)

            #return_data[:,parse(string("iteration_", c))] = Array(model_returns)[:,1]
        catch y
            error_count = error_count + 1
            println(string("Error on: ", c))
            println(y)
            continue
        end
    end

    println("Error Count: $error_count")
    CreateCSCVRecords(aggregated_returns)
end

function ExperimentCSCVProcess(config_ids)

    #cscv_returns = RunQuery("select * from cscv_returns")# where configuration_id < 26864")
    cscv_returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0) total_profit_rate_observed
                                from cscv_returns where configuration_id >= 26264")

    #return_data = DataFrame()
    cscv_groups = by(cscv_returns, [:configuration_id], df -> [df[:,[:time_step, :total_profit_rate_observed]]])

    #maxes = fill(0, (2410, 1))
    #i = 2
    #for i in 1:2410
    #    one_data = deepcopy(cscv_groups[i,2])
    #    nullvals = map(i -> isnull(one_data[i, :total_profit_rate_observed]), 1:size(one_data, 1))
    #    sum(nullvals)
    #    one_data[nullvals, :total_profit_rate_observed] = 0
    #    zp_ind = Array(one_data[:total_profit_rate_observed]) .> 0
    #    maxnon = maximum(Array(one_data[zp_ind, :time_step]))
    #    maxes[i] = maxnon
    #end
    #sum(maxes .< 100)

    #all_groups = deepcopy(cscv_groups)

    #28879 - 26864
    #cscv_groups[end,:]
    #cscv_groups = all_groups[(end-2000):end,:]
    arr_data = fill(0.0, (1561, size(cscv_groups,1)))

    #function nullfunc(x)
    #    if isnull(x)
    #        return 0.0
    #    end
    #    return get(x)
    #end

    for r in 1:size(cscv_groups,1)
        #println(map(nullfunc, vals))
        #println(cscv_groups[1,2][:total_profit_rate_observed])
        #name = string("configuration_", get(cscv_groups[r,1]))

        #vals = (cscv_groups[r,2][:total_profit_rate_observed])

        #floatvals = Array{Float64}(map(nullfunc, vals))

        #config_returns = map(i -> nullfunc(vals[i]), 1:1561)
        #arr_data[:,r] = config_returns

        arr_data[:,r] = Array(cscv_groups[r,2][:total_profit_rate_observed])

        #arr_data = hcat(arr_data, config_returns)
        #return_data[parse(name)] = config_returns
    end

    #return_data

    #arr_data = Array(return_data)
    #splits = 12
    #nrows = 1561
    #length = Int64.(floor(nrows/splits))
    #ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
    #all = Set(ranges)
    #m = Int64.(splits/2)
    #training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
    #set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)
    #sp = set_pairs[1]
    #train_ind = sort(mapreduce(x -> map(y -> y, x),vcat, sp[1]))
    #test_ind =  sort(mapreduce(x -> map(y -> y, x),vcat, sp[2]))
    #train_data = arr_data[train_ind, :]
    #test_data = arr_data[test_ind, :]

    #############################################################################

    #ltd_arr = arr_data[1:1550, :]


    println("new")
    distribution = RunCSCV(arr_data, 10)
    pbo = CalculatePBO(distribution)

    using PlotlyJS
    x0 = map(i -> round(distribution[i][1],1), 1:size(distribution,1))
    y0 = map(i -> distribution[i][2], 1:size(distribution,1))
    trace = bar(;x=x0,y=y0, name="Logit Distribution", bargap=0.1)
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/CSCV Distribution Smaller Networks.html"))

    return (pbo, mses)
end

#vals = ExperimentCSCVProcess(26363:27658)
#config_ids = 26363:27658

#end
#=

function GenerateCSCVReturnsOld(config_id, original_prices, results)

    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end

    if (size(results, 1) == 0 || (maximum(results[:time_step])) == timestep)
        return 0
    end

    num_predictions = (maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions + 1

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateCSCVStrategyReturns(stockreturns, timestep)

    return strategyreturns[:, [:time_step, :total_profit_observed, :total_profit_rate]]
end

=#


#=

function GenerateTotalProfitOld(config_id, original_prices)

    println("G1")

    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    println("G2")

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end

    println("G3")
    results = RunQuery("select * from prediction_results where configuration_id = $config_id and predicted is not null")
    println("G4")
    if (size(results, 1) == 0 || get(maximum(results[:time_step])) == timestep)
        return 0
    end

    println("G5")

    num_predictions = get(maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions + 1

    println("G6")

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)

    println("G7")

    #all(round.(stockreturns[1,2][:observed_t],4) .== round.(Array(original_prices[601:996,1]),4))
    #all(round.(Array(stockreturns[1,2][:observed_t2]),4) .== round.(Array(original_prices[606:1001,1]),4))

    return strategyreturns[end, :cumulative_profit_observed]
end
=#

#=
function GenerateTotalProfit(config_id, original_prices, results)

    configresults = RunQuery("select * from configuration_run where configuration_id = $config_id")
    sae_id = get(configresults[1, :sae_config_id])
    data_config = ReadSAE(sae_id)[2]
    timestep = data_config.prediction_steps[1]

    if original_prices == nothing
        processed_data = PrepareData(data_config, nothing)
        original_prices = processed_data[2].original_prices
    end

    if (size(results, 1) == 0 || get(maximum(results[:time_step])) == timestep)
        return 0
    end

    num_predictions = get(maximum(results[:time_step]))
    finish_t = size(original_prices, 1)
    start_t = finish_t - num_predictions + 1

    stockreturns = GenerateStockReturns(results, start_t, finish_t, timestep, original_prices)
    strategyreturns = GenerateStrategyReturns(stockreturns, timestep)

    #all(round.(stockreturns[1,2][:observed_t],4) .== round.(Array(original_prices[601:996,1]),4))
    #all(round.(Array(stockreturns[1,2][:observed_t2]),4) .== round.(Array(original_prices[606:1001,1]),4))

    return strategyreturns[end, :cumulative_profit_observed]
end

=#
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
