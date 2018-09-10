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

export MeanSquaredError

function MeanSquaredError(y, y_hat)
    return(sum((y - y_hat).^2)/size(y, 1))
end


end


module StoppingFunctions

using TrainingStructures

export GenValidationChangeReached, NonStopping

function NonStopping(records::Array{EpochRecord})
    return false
end

function GenValidationChangeReached(target_change_rate)
    function ValidationChangeReached(records::Array{EpochRecord})
        if length(records) <= 1
            return false
        end

        second_last_change = records[(end-1)].validation_cost_error
        last_change = records[(end)].validation_cost_error
        return((second_last_change - last_change)/second_last_change <= target_change_rate)
    end

    return ValidationChangeReached
end

end
