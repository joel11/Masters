module ActivationFunctions

export SigmoidActivation, SigmoidPrime, SoftmaxActivation, ReluActivation, NoisyReluActivation, LinearActivation, FunctionDerivatives



function SigmoidActivation(x)
  return (1.0 ./ (1.0 .+ exp.(-x)))
end

function SigmoidPrime(x)
    act = SigmoidActivation(x)
    return (act.*(1-act))
end

function LinearActivation(x)
    return (x)
end

function LinearPrime(x)
    return 1
end

function ReluActivation(x)
    return (max.(0, x))
end

function ReluPrime(x)
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


FunctionDerivatives = Dict{Function,Function}(SigmoidActivation=>SigmoidPrime, ReluActivation=>ReluPrime, LinearActivation=>LinearPrime)

end

module InitializationFunctions

using Distributions

export  HintonUniformInit, XavierGlorotUniformInit

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

#input = 100
#output = 40
#v = HintonUniformInit(5, 4)
#var(v)
#xgn = XavierGlorotNormalInit(input, output)
#println(var(xgn), " ", 2/(input + output))
#xgn = XavierGlorotUniformInit(input, output)
#println(var(xgn), " ", 2/(input + output))
#xgn = HeNormalInit(input, output)
#println(var(xgn), " ", 2/(input))
#xgn = HeUniformInit(input, output)
#println(var(xgn), " ", 2/(input))
end

module CostFunctions

using ActivationFunctions

export MeanSquaredError, CrossEntropyError, CategoricalCrossEntropyError, LoglikelihoodError, CostFunction

abstract type CostFunction end

type MeanSquaredError <: CostFunction

    CalculateCost::Function
    Delta::Function

    function MeanSquaredError()
        function cost_function(y, y_hat)
            cost =  sum((y - y_hat).^2)/size(y, 1)
            println(string( sum((y - y_hat).^2), " , ", size(y, 1), " , ", cost))
            return cost
        end

        function delta_function(a, y, z_vals, activation)
            derivative_activations = FunctionDerivatives[activation](z_vals)
            return ((a-y).*derivative_activations)
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
