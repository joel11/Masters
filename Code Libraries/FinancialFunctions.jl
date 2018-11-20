module FinancialFunctions

export SharpeRatio, CalculateProfit, CalculateReturns

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function CalculateProfit(actual, predicted)
    rev = (abs(actual) > abs(predicted)) ? abs(predicted) : sign(predicted) * (actual - predicted)
    return rev
end

function CalculateReturns(actual, predicted)
    return map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])
end

end
