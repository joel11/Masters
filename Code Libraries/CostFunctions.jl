
module CostFunctions

using ActivationFunctions

export MeanSquaredError, CrossEntropyError, CategoricalCrossEntropyError, LoglikelihoodError, CostFunction, CalculateMAPE

function CalculateMAPE(y, y_hat)
    vals = abs.((Array(y) - Array(y_hat)) ./ Array(y))
    mape_vals = vals[map(x ->  (isa(x, Number) && !isnan(x) && !isinf(x)), vals)]
    if length(mape_vals) > 0
        return median(mape_vals)*100
    end
    return 100
end

abstract type CostFunction end

type MeanSquaredError <: CostFunction

    CalculateCost::Function
    Delta::Function

    function MeanSquaredError()
        function cost_function(y::Array{Float64,2}, y_hat::Array{Float64,2})
            return sum((y .- y_hat) .^2) /length(y)#, 1)
            #println(string( sum((y - y_hat).^2), " , ", length(y), " , ", cost))
            #return cost
        end

        function delta_function(a::Array{Float64,2}, y::Array{Float64,2}, z_vals::Array{Float64,2}, activation::Function)
            der = FunctionDerivatives[activation](z_vals)
            return ((a-y).*der)
        end

        return new(cost_function, delta_function)
    end
end

type CrossEntropyError <: CostFunction

    CalculateCost::Function
    Delta::Function

    function CrossEntropyError()
        function cost_function(y, y_hat)
            one = y.*log.(e, y_hat)
            two = (1.-y).*log.(e, 1.-y_hat)
            n = size(y, 1)
            return(-sum(one + two)/n)
        end

        function delta_function(a, y, z_vals, activation)
            return (a-y)
        end

        return new(cost_function, delta_function)
    end
end

type LoglikelihoodError <: CostFunction

    CalculateCost::Function
    Delta::Function

    function LoglikelihoodError()
        function cost_function(y, y_hat)
            n = size(y, 1)
            return(sum(map( i-> (-log.(e, y_hat[i,:])'[Bool.(y[i,:])]), 1:size(y)[1]))[1]/n)
        end

        function delta_function(a, y, z_vals, activation)
            return (a-y)
        end

        return new(cost_function, delta_function)
    end
end

end
