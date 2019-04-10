module FinancialFunctions

export SharpeRatio, CalculateProfit, CalculateReturns, CalculateReturnsOneD

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function CalculateExpectedProfit(actual, predicted)

    prof = nothing
    #accuracy = predicted / actual

    if sign(actual) == sign(predicted)
        prof = abs(actual) #* accuracy
    else
        prof = - abs(actual - predicted)
    end
    return prof
end

function CalculateRealisedProfit(actual, predicted)

    prof = nothing
    #accuracy = predicted / actual

    if sign(actual) == sign(predicted)
        prof = abs(actual) #* accuracy
    else
        prof = - abs(actual - predicted)
    end
    return prof
end

function CalculateReturnsOneD(actuals, predicted)
    total_return = 0
    for i in 1:size(actuals,1)
        total_return += CalculateProfit(actuals[i], predicted[i])
    end
    return total_return
end

function CalculateReturns(actuals, predicted)
    ret =  mapreduce(r -> map(c -> CalculateProfit(actuals[r, c], predicted[r, c]), 1:size(actuals)[2]), hcat, 1:size(actuals)[1])'
    return (ret)
    #return map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])
end

end
