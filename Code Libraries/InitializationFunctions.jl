module InitializationFunctions

using Distributions

export  DCNormalInit, DCUniformInit, HintonUniformInit, XavierGlorotUniformInit, HeUniformInit, NormalRandomInit, HeNormalInit

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

function DCUniformInit(input, output)
    init_range = sqrt(6.0/mean([input output]))
    weights = rand(Uniform(-init_range, init_range), input, output)
    return (weights)
end

function DCNormalInit(input, output)
    weights = rand(Normal(0, sqrt(2.0/(mean[input output]))), input, output)
    return(weights)
end

end
