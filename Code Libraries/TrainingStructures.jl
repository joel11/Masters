module TrainingStructures

using NeuralNetworks

export TrainingParameters, EpochRecord, PrintEpoch

type TrainingParameters

    learning_rate::Float64
    minibatch_size::Int64
    momentum_rate::Float64
    max_rbm_epochs::Int64
    max_ffn_epochs::Int64
    stopping_function

end

type EpochRecord

    epoch_number::Int64
    mean_cost_error::Float64
    validation_cost_error::Float64
    run_time::Float64
    energy_ratio::Float64

    network::NeuralNetworks.NeuralNetwork
    weight_change_rates::Array{Array{Float64,1},1}
    hidden_activation_likelihoods::Array{Array{Float64,2},1}

    function EpochRecord(epoch_number, mean_cost_error, validation_cost_error,run_time,energy_ratio, network, weight_change_rates, hidden_activation_likelihoods)
        return new(epoch_number, mean_cost_error, validation_cost_error,run_time,energy_ratio, network, weight_change_rates, hidden_activation_likelihoods)
    end
end

function PrintEpoch(epoch_record::EpochRecord)
    println("Epoch $(epoch_record.epoch_number) Mean Minibatch Error, Validation Error, Time, Energy Ratio:
                $(epoch_record.mean_cost_error),
                $(epoch_record.validation_cost_error),
                $(epoch_record.run_time),
                $(epoch_record.energy_ratio)")
end

end
