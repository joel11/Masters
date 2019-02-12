module FunctionsStopping

using TrainingStructures

export GenValidationChangeReached, NonStopping

function NonStopping(parameter_tuple)
    function NonStoppingFunc(parameter_tuple)
        return false
    end
    return NonStoppingFunc
end

function GenValidationChangeReached(parameter_tuple)

    function ValidationChangeReached(records::Array{EpochRecord})
        if length(records) <= over_epochs || length(records) < 2*over_epochs
            return false
        end

        second_last_change = maximum(map(x -> x.training_cost, records[(end - over_epochs):(end-1)]))
        last_change = records[(end)].training_cost
        println(round(second_last_change - last_change, 10))
        println(round(second_last_change, 10))
        println((second_last_change - last_change)/second_last_change)
        return(abs((second_last_change - last_change)/second_last_change) <= target_change_rate)
    end
    target_change_rate = parameter_tuple[1]
    over_epochs = parameter_tuple[2]

    return ValidationChangeReached
end

end
