module TrainingStructures

export TrainingParameters, EpochRecord, PrintEpoch

type TrainingParameters

    learning_rate::Float64
    minibatch_size::Int64
    momentum_rate::Float64
    number_epochs::Int64


end

type EpochRecord

    epoch_number::Int64
    reconstruction_error::Float64
    cross_entropy_error::Float64
    run_time::Float64
    energy_ratio::Float64
    weights::Array{Float64, 2}
    weight_change_rates::Array{Float64}

    function EpochRecord(epoch_number, reconstruction_error,cross_entropy_error,run_time,energy_ratio,weights, weight_change_rates)
        return new(epoch_number, reconstruction_error,cross_entropy_error,run_time,energy_ratio,weights, weight_change_rates)
    end
end

function PrintEpoch(epoch_record::EpochRecord)
    println("Epoch $(epoch_record.epoch_number) Error, Time, Energy Ratio, Cross Error:
                $(epoch_record.reconstruction_error),
                $(epoch_record.run_time),
                $(epoch_record.energy_ratio),
                $(epoch_record.cross_entropy_error)")
end

end
