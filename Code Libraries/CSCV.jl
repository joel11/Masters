module CSCV

using Combinatorics
using StatsBase
using FinancialFunctions
using DatabaseOps
using DataFrames
using DataJSETop40
using PlotlyJS

export GenerateCSCVReturns, RunCSCV, CalculatePBO, ExperimentCSCVProcess

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end

function CalculatePBO(overfit_distribution)
    return mapreduce(x -> (x[1] <= 0 ? x[2] : 0), +, overfit_distribution)
end

function ProcessCombination(data, training_indices, testing_indices)

    ## 4
    ## a & b
    train_ind = sort(mapreduce(x -> map(y -> y, x),vcat, training_indices))
    test_ind =  sort(mapreduce(x -> map(y -> y, x),vcat, testing_indices))

    ## c & d
    N = size(data[train_ind, :])[2]
    train_pvals = fill(0.0, N)
    test_pvals = fill(0.0, N)

    for i in 1:N
        train_pvals[i] = SharpeRatio(data[train_ind,i], 0.00)
        test_pvals[i]  = SharpeRatio(data[test_ind ,i], 0.00)
    end

    test_pvals[isnan.(test_pvals)] = 0
    train_pvals[isnan.(train_pvals)] = 0

    #println("4d")
    r = ordinalrank(train_pvals)
    rbar = ordinalrank(test_pvals)

    ## e & f & g
    is_index = findin(r, maximum(r))
    oos_val = rbar[is_index][1]
    wc = oos_val / (1 + N)
    lval = wc / (1 - wc)
    lambda_c = log(e, lval)

    return lambda_c
end

function RunCSCV(data, splits)

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

function ExperimentCSCVProcess(config_ids, splits)

    ids = TransformConfigIDs(config_ids)

    oos_time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_oos_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    oos_start = get(maximum(oos_time_steps[:min_ts]))
    oos_finish = get(minimum(oos_time_steps[:max_ts]))

    is_time_steps = RunQuery("select min(time_step) min_ts, max(time_step) max_ts
                            from config_is_trade_returns
                            where configuration_id in ($ids) group by configuration_id")

    is_start = get(maximum(is_time_steps[:min_ts]))
    is_finish = get(minimum(is_time_steps[:max_ts]))

    query = "select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                from config_is_trade_returns
                                where time_step between $is_start and $is_finish
                                and configuration_id in ($ids)
            union
            select configuration_id, time_step, ifnull(total_profit_rate_observed, 0.0) total_profit_rate_observed
                                from config_oos_trade_returns
                                where time_step between $oos_start and $oos_finish
                                and configuration_id in ($ids)
            order by configuration_id, time_step"

    cscv_returns = RunQuery(query)

    cscv_groups = by(cscv_returns, [:configuration_id], df -> [df[:,[:time_step, :total_profit_rate_observed]]])


    #Assess maximum prediction point in order to filter out 'NULL' networks, which would bias the result unnecessarily
    maxes = fill(0, (size(cscv_groups,1), 1))
    for i in 1:size(cscv_groups,1)
        one_data = cscv_groups[i,2]
        zp_ind = Array(one_data[:total_profit_rate_observed]) .!= 0
        if sum(zp_ind) > 0
            maxnon = maximum(Array(one_data[zp_ind, :time_step]))
        else
            maxnon = 0
        end
        maxes[i] = maxnon
    end

    row_limit = minimum(map(r -> size(cscv_groups[r,2],1), 1:size(cscv_groups,1)))
    arr_data = fill(0.0, (row_limit, size(cscv_groups,1)))

    for r in 1:size(cscv_groups,1)
        arr_data[:,r] = Array(cscv_groups[r,2][1:row_limit, :total_profit_rate_observed])
    end

    inds = Array{Bool}((maxes .> is_finish + 1000)[:,1])

    ltd_arr = arr_data[:, inds]

    pbo_dict = Dict()
    for s in splits
        println("s: $s")
        distribution = RunCSCV(ltd_arr, s)
        pbo = CalculatePBO(distribution)
        pbo_dict[s] = pbo
        PlotLogitDistribution(distribution, string("Logit Distribution ", s))
    end

    for s in splits
        println("$s : ", pbo_dict[s])
    end

    return (pbo_dict)
end

function PlotLogitDistribution(distribution, file_name)

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Frequency </br> </b>"))
        , xaxis = Dict(:title => string("<b> Logits </b>"))
        , font = Dict(:size => 18))

    xvals = unique(map(i -> round(distribution[i][1],1), 1:size(distribution,1)))
    yvals = Array{Float64,1}()

    for x in xvals
        inds = filter(i -> round(distribution[i][1],1) == x, 1:size(distribution,1))
        push!(yvals, sum(map(i -> i[2], distribution[inds])))
    end

    trace = bar(;x=xvals,y=yvals, name="Logit Distribution", bargap=0.1, marker = Dict(:line => Dict(:width => 3, :color => "orange"), :color=>"orange"))
    data = [trace]
    savefig(plot(data, l), string("./GraphOutputDir/", file_name, ".html"))
end

end
