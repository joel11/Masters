module ActivationFunctions

export SigmoidActivation, SigmoidPrime, TanhActivation, TanhPrime, SoftmaxActivation, ReluActivation, NoisyReluActivation, LinearActivation, FunctionDerivatives



function SigmoidActivation(x::Array{Float64,2})
  return (1.0 ./ (1.0 .+ exp.(-x)))
end

function SigmoidPrime(x::Array{Float64,2})
    act = SigmoidActivation(x)
    return (act.*(1-act))
end

function TanhActivation(x::Array{Float64,2})
    return tanh.(x)
end

function TanhPrime(x::Array{Float64,2})
    v = 1 .- (tanh.(x)).^2
    return v
end

function LinearActivation(x::Array{Float64,2})
    return (x)
end

function LinearPrime(x::Array{Float64,2})
    return 1
end

function ReluActivation(x::Array{Float64,2})
    return (max.(0, x))
end

function ReluPrime(x::Array{Float64,2})
    return (max.(0,x)./x)
end

function SoftmaxActivation(vals)
    function Softmax(x)
        num_x = (x .- maximum(x))
        soft_den = sum(exp.(num_x))
        return (exp.(num_x) ./ soft_den)
    end

    reduce(hcat, map(m -> Softmax(vals[m,:]), 1:size(vals,1)))'
end


const FunctionDerivatives = Dict{Function,Function}(SigmoidActivation=>SigmoidPrime, TanhActivation=> TanhPrime, ReluActivation=>ReluPrime, LinearActivation=>LinearPrime)

end

module InitializationFunctions

using Distributions

export  HintonUniformInit, XavierGlorotUniformInit, HeUniformInit

function NormalRandomInit(input, output)
    weights = rand(Normal(0, 1), input, output)
    return (weights)
end

function HintonUniformInit(input, output)
    weights = rand(Normal(0, 0.01), input, output)
    return (weights)
end

function XavierGlorotUniformInit(input, output)
    init_range = sqrt(6.0/(input + output))
    weights = rand(Uniform(-init_range, init_range), input, output)
    return (weights)
end

function XavierGlorotNormalInit(input, output)
    weights = rand(Normal(0, sqrt(2.0/(input+output))), input, output)
    return(weights)
end

function HeUniformInit(input, output)
    init_range = sqrt(6.0/input)
    weights = rand(Uniform(-init_range, init_range), input, output)
    return (weights)
end

function HeNormalInit(input, output)
    weights = rand(Normal(0, sqrt(2.0/input)), input, output)
    return(weights)
end

end

module CostFunctions

using ActivationFunctions

export MeanSquaredError, CrossEntropyError, CategoricalCrossEntropyError, LoglikelihoodError, CostFunction, CalculateMAPE

function CalculateMAPE(y, y_hat)
    vals = abs.((y - y_hat) ./ y)
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

#y = [[1 0];[0 1]; [0 0]]'
#y_hat = [[0.4 0.3]; [0.6 0.7]; [0.1 0.1]]'
#sum(map( i-> (-log.(e, y_hat[i,:])'[Bool.(y[i,:])]), 1:size(y)[1]))[1]
#-log(e, 0.4) + -log(e, 0.7)
