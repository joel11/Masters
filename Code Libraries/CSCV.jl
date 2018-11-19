module CSCV

using Combinatorics
using StatsBase
using FinancialFunctions

export RunCSCV, CalculatePBO

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


#splits = 8
#data = randn(1000, 10)
end
