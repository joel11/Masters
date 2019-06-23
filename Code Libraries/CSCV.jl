module CSCV

#workspace()
#push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using Combinatorics
using StatsBase
using FinancialFunctions
using DatabaseOps
using DataFrames

export RunCSCV, CalculatePBO

function TransformConfigIDs(config_ids)
    return (mapreduce(c -> string(c, ","), (x, y) -> string(x, y), config_ids)[1:(end-1)])
end

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


function ProcessCombination(data, training_indices, testing_indices)
    ## 4
    ## a & b

    train_ind = sort(mapreduce(x -> map(y -> y, x),vcat, training_indices))
    test_ind =  sort(mapreduce(x -> map(y -> y, x),vcat, testing_indices))

    train_data = data[train_ind, :]
    test_data = data[test_ind, :]

    ## c & d
    N = size(train_data)[2]
    train_pvals = map(x -> SharpeRatio(train_data[:,x], 0.01), 1:N)
    test_pvals = map(x -> SharpeRatio(test_data[:,x], 0.01), 1:N)

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
    nrows = size(data)[1]
    length = Int64.(floor(nrows/splits))

    ranges = map(i -> (i*length+1):((i+1)*(length)), 0:(splits-1))
    all = Set(ranges)
    m = Int64.(splits/2)

    ##3
    training_sets = map(i -> Set(ranges[i]), combinations(1:(size(ranges)[1]),m))
    set_pairs = map(ts -> (ts, setdiff(all, ts)), training_sets)

    ##4
    logitvals = map(sp -> ProcessCombination(data, sp[1], sp[2]), set_pairs)

    ##5
    overfit_dist = map(x -> (x, size(findin(logitvals, x))[1]/(size(logitvals)[1])), sort(unique(logitvals)))

    return overfit_dist
end

function CalculatePBO(overfit_distribution)
    return mapreduce(x -> (x[1] <= 0 ? x[2] : 0), +, overfit_distribution)
end

#config_ids = Array(RunQuery("select distinct(configuration_id) from prediction_results where configuration_id between 3704 and 4759")[:,1])



function ExperimentCSCVProcess(config_ids)
    return_data = DataFrame()
    mses = []

    ids = TransformConfigIDs(config_ids)
    query = "select configuration_id, actual, predicted from prediction_results where configuration_id in ($ids)"
    all_predictionvals = RunQuery(query)

    for c in config_ids
        println(c)

        predictionvals = all_predictionvals[Array(all_predictionvals[:,:configuration_id]) .== c,:]
        pa = Array{Float64}(fill(NaN, size(predictionvals,1),2))
        pa[:,1] = Array(predictionvals[:predicted])
        pa[:,2] = Array(predictionvals[:actual])

        actual = pa[:,2:2]
        predicted = pa[:,1:1]

        mse = sum((actual - predicted).^2)/length(actual)
        model_returns = CalculateReturns(actual, predicted, 5)
        push!(mses, mse)

        if size(model_returns, 1) != size(return_data, 1)
            zeros = map(i -> 0, 1:(size(return_data, 1)-size(model_returns, 1)))
            model_returns = vcat(model_returns ,zeros)
        end
        return_data[:,parse(string("iteration_", c))] = Array(model_returns)[:,1]
    end

    distribution = RunCSCV(return_data, 16)
    pbo = CalculatePBO(distribution)

    using PlotlyJS
    x0 = map(i -> round(distribution[i][1],1), 1:size(distribution,1))
    y0 = map(i -> distribution[i][2], 1:size(distribution,1))
    trace = bar(;x=x0,y=y0, name="Logit Distribution", bargap=0.1)
    data = [trace]
    savefig(plot(data), string("/users/joeldacosta/desktop/CSCV Distribution.html"))

    return (pbo, mses)
end

#vals = ExperimentCSCVProcess(26363:27658)
#config_ids = 26363:27658

end
