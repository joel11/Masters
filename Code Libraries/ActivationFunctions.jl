module ActivationFunctions

export SigmoidActivation, SoftmaxActivation, ReluActivation, NoisyReluActivation, LinearActivation, Function_dictionary



function SigmoidActivation(x)
  return (1.0 ./ (1.0 .+ exp.(-x)))
end

function SigmoidPrime(x)
    act = SigmoidActivation(x)
    return (act*(1-act))
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

Function_dictionary = Dict(SigmoidActivation=>SigmoidPrime)

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
