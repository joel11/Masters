module FinancialFunctions

export SharpeRatio, CalculateProfit, CalculateReturns

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

function CalculateProfit(actual, predicted)

    prof = nothing

    if predicted > 0 && actual > 0 && (predicted / actual < 1.1)
        prof = actual
    else#if predicted >= 0 && actual < 0
        println(string("A",- actual))
        prof = - actual
    #else
    #    prof = 0
    end
    println("$actual , $predicted : $prof")
    return prof
end

function CalculateReturns(actuals, predicted)
    ret =  mapreduce(r -> map(c -> CalculateProfit(actuals[r, c], predicted[r, c]), 1:size(actuals)[2]), hcat, 1:size(actuals)[1])'
    return (ret)
    #return map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])
end

end
