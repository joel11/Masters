module ActivationFunctions

export LeakyReluActivation, SigmoidActivation, SigmoidPrime, TanhActivation, TanhPrime, SoftmaxActivation, ReluActivation, NoisyReluActivation, LinearActivation, FunctionDerivatives

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

####

function LeakyReluActivation(x::Array{Float64,2})
    return max.(0, x) .+ min.(0, x * 0.01)
end

function LeakyReluPrime(x::Array{Float64,2})
    return max.(0, x)./x .+ min.(0, x * 0.01)./x
end

function SoftmaxActivation(vals)
    function Softmax(x)
        num_x = (x .- maximum(x))
        soft_den = sum(exp.(num_x))
        return (exp.(num_x) ./ soft_den)
    end

    reduce(hcat, map(m -> Softmax(vals[m,:]), 1:size(vals,1)))'
end


const FunctionDerivatives = Dict{Function,Function}(SigmoidActivation=>SigmoidPrime, TanhActivation=> TanhPrime, ReluActivation=>ReluPrime, LeakyReluActivation=>LeakyReluPrime, LinearActivation=>LinearPrime)

end
