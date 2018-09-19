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
