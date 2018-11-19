module FinancialFunctions

export SharpeRatio

function SharpeRatio(returns, rfr)
    return (mean(returns) - rfr)/std(returns)
end

end
