module ActivationFunctions

export SigmoidActivation, SigmoidPrime, SoftmaxActivation, ReluActivation, NoisyReluActivation, LinearActivation, FunctionDerivatives



function SigmoidActivation(x)
  return (1.0 ./ (1.0 .+ exp.(-x)))
end

function SigmoidPrime(x)
    act = SigmoidActivation(x)
    return (act.*(1-act))
end

function SoftmaxActivation(x)
    soft_den = sum(exp.(x))
    return (exp.(x) ./ soft_den)
end

function ReluActivation(x)
    return (max.(0, x))
end

function NoisyReluActivation(x)
    noisy_values = x + randn(size(x))
    return(max.(0, noisy_values))
end

FunctionDerivatives = Dict{Function,Function}(SigmoidActivation=>SigmoidPrime)

end

module InitializationFunctions

using Distributions

export  HintonUniformInit, XavierGlorotUniformInit

function HintonUniformInit(input, output)
    weights = rand(Normal(0, 0.01), input, output)
    return (weights)
end

function XavierGlorotUniformInit(input, output)
    init_range = sqrt(6.0/(input + output))
    weights = rand(Uniform(-init_range, init_range), input, output)
    return (weights)
end


end

module CostFunctions

export MeanSquaredError, CrossEntropyError

function MeanSquaredError(y, y_hat)
    return(sum((y - y_hat).^2)/size(y, 1))
end

function CrossEntropyError(y, y_hat)
    one = y.*log.(e, y_hat)
    two = (1.-y).*log.(e, 1.-y_hat)
    n = size(y, 1)
    return(-sum(one + two)/n)
end

end
