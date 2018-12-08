module FinancialFunctions

export SharpeRatio, CalculateProfit, CalculateReturns

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function CalculateProfit(actual, predicted)
    #if sign(actual) == sign(predicted)
#        return abs(actual)
#    else
#        return - abs(actual - predicted)
#    end
    return abs(actual - abs(actual - predicted))
end

function CalculateReturns(actuals, predicted)
    return mapreduce(r -> map(c -> CalculateProfit(actuals[r, c], predicted[r, c]), 1:size(actuals)[2]), hcat, 1:size(actuals)[1])'
    #return map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])
end

end
