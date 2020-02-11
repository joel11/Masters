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

	config_starts = 28880:1000:51904

    for c in config_starts
        range = c:(c+999)
        println(range)
        tic()
        #WriteCSCVReturns(range, dataset)
        WriteCSCVCostReturns(range, dataset)
        println(toc())
    end
end

function WriteCSCVCostReturns(config_ids, dataset)

    #config_ids = (35000, 50231, 50232) #49617:(49617+399)

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

            model_returns = GenerateCSCVReturnsCost(c, dataset, config_predictionvals)
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
    CreateCSCVCostRecords(aggregated_returns)
end

function WriteCSCVReturns(config_ids, dataset)

    #config_ids = (35000, 50231, 50232) #49617:(49617+399)

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



function ExperimentCSCVProcess(config_ids)

        cscv_returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                     from cscv_returns")
                                     #where configuration_id >= 50393") #Large
                                     #where configuration_id >= 51689") #PBO test

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
        #sum(maxes .< 100)

        row_limit = minimum(map(r -> size(cscv_groups[r,2],1), 1:size(cscv_groups,1)))
        arr_data = fill(0.0, (row_limit, size(cscv_groups,1)))

        for r in 1:size(cscv_groups,1)
            arr_data[:,r] = Array(cscv_groups[r,2][1:row_limit, :total_profit_rate_observed])
        end

        inds = Array{Bool}((maxes .> 3000)[:,1])

        sum(inds)
        #############################################################################

        #ltd_arr = arr_data #[1:(end-16), inds]
        ltd_arr = arr_data[:, inds]

        distribution = RunCSCV(ltd_arr, 16)
        pbo = CalculatePBO(distribution)

        #println("new")
        pbo_dict = Dict()
        for s in (2, 4, 8, 12) #, 16)
            println("s: $s")
            distribution = RunCSCV(ltd_arr, s)
            #distribution = RunCSCV(ltd_arr, 16)
            pbo = CalculatePBO(distribution)
            pbo_dict[s] = pbo
        end

        for s in (2, 4, 8, 12, 16)
            println("$s : ", pbo_dict[s])
        end

        #sgd_dict[sgdlr] = pbo_dict
    #end




    using PlotlyJS

    l = Layout(width = 1600, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Frequency </br> </b>"))
        , xaxis = Dict(:title => string("<b> Logits </b>")))

    xvals = unique(map(i -> round(distribution[i][1],1), 1:size(distribution,1)))
    yvals = Array{Float64,1}()

    for x in xvals
        inds = filter(i -> round(distribution[i][1],1) == x, 1:size(distribution,1))
        push!(yvals, sum(map(i -> i[2], distribution[inds])))
    end

    trace = bar(;x=xvals,y=yvals, name="Logit Distribution", bargap=0.1, marker = Dict(:line => Dict(:width => 3, :color => "orange"), :color=>"orange"))
    data = [trace]
    savefig(plot(data, l), string("/users/joeldacosta/desktop/All CSCV PBO Test.html"))




    return (pbo, mses)
end

function WriteSharpeRatios()

    returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                 from cscv_returns
                                 where configuration_id < 51905
                                 and configuration_id not between 50249 and 50392
                                 and time_step > 2333")

    returns[:,1] = Array{Int64,1}(returns[:,1])
    returns[:,2] = Array{Int64,1}(returns[:,2])
    returns[:,3] = Array{Float64,1}(returns[:,3])

    ratios = by(returns, :configuration_id, df-> SharpeRatio(df[:total_profit_rate_observed], 0.0))

    CreateSRRecords(ratios)
end

function WriteSharpeRatiosCost()

    returns = RunQuery("select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                 from cscv_cost_returns
                                 where configuration_id < 51905
                                 and configuration_id not between 50249 and 50392
                                 and time_step > 2333")

    returns[:,1] = Array{Int64,1}(returns[:,1])
    returns[:,2] = Array{Int64,1}(returns[:,2])
    returns[:,3] = Array{Float64,1}(returns[:,3])

    ratios = by(returns, :configuration_id, df-> SharpeRatio(df[:total_profit_rate_observed], 0.0))

    CreateSRRecordsCost(ratios)
end

function RunConfusionGenerations()
    jsedata = ReadJSETop40Data()
    dataset = jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

    config_starts = 48388:400:52336

    for c in config_starts
        range = c:(c+399)
        println(range)
        tic()
        GenerateConfusionDistributions(range, dataset)
        println(toc())
    end
end

function GenerateConfusionDistributions(config_range, original_prices)

    config_min = minimum(config_range)
    config_max = maximum(config_range)

    all_results = RunQuery("select configuration_id, time_step, stock, actual, ifnull(predicted, 0) predicted from prediction_results where configuration_id between $config_min and $config_max")

    confusion_df = DataFrame(configuration_id = [], no_trade_perc = [], trade_perc = [], all_perc = [])
    confusion_df[:,1] = Array{Int64}(confusion_df[:,1])
    confusion_df[:,2] = Array{Float64}(confusion_df[:,2])
    confusion_df[:,3] = Array{Float64}(confusion_df[:,3])
    confusion_df[:,4] = Array{Float64}(confusion_df[:,4])

    for config_id in config_range
        #tic()
        results = all_results[Array(all_results[:, :configuration_id]) .== config_id, :]

        if size(results, 1) > 0

            num_predictions = get(maximum(results[:time_step]))
            finish_t = size(original_prices, 1)
            start_t = finish_t - num_predictions + 1

            #results = RunQuery("select * from configuration_run where configuration_id = $config_id")
            #sae_id = get(results[1, :sae_config_id])
            #data_config = ReadSAE(sae_id)[2]
            #timestep = data_config.prediction_steps[1]
            timestep = 5

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

            #df_vals = DataFrame(configuration_id = [config_id], no_trade_perc = [no_trade_perc], trade_perc = [trade_perc], all_perc = [all_perc])

            append!(confusion_df, appendDF)
            #println(appendDF)
            #println(toc())
        end
    end

    CreateConfusionRecords(confusion_df)
end

#end
