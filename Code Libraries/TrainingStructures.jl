module TrainingStructures

using NeuralNetworks

export TrainingParameters, EpochRecord, PrintEpoch, DataSet

type DataSet
    training_input::Array{Float64,2}
    testing_input::Array{Float64,2}
    validation_input::Array{Float64,2}

    training_output::Array{Float64,2}
    testing_output::Array{Float64, 2}
    validation_output::Array{Float64,2}
end

type TrainingParameters

    learning_rate::Float64
    minibatch_size::Int64
    momentum_rate::Float64
    max_rbm_epochs::Int64
    max_ffn_epochs::Int64
    stopping_function
    verbose::Bool
    is_classification::Bool

    l1_lambda::Float64
    l2_lambda::Float64
end

type EpochRecord

    #epoch_number, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods

    epoch_number::Int64

    mean_minibatch_cost::Float64
    training_cost::Float64
    test_cost::Float64

    training_accuracy::Float64
    test_accuracy::Float64

    energy_ratio::Float64

    run_time::Float64
    network::NeuralNetworks.NeuralNetwork
    weight_change_rates::Array{Array{Float64,1},1}
    hidden_activation_likelihoods::Array{Array{Float64,2},1}

    function EpochRecord(epoch_number, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods)
        return new(epoch_number, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods)
    end
end

function PrintEpoch(epoch_record::EpochRecord)
    println("Epoch $(epoch_record.epoch_number) Training Cost, Test Cost, Time,:
                $(epoch_record.training_cost),
                $(epoch_record.test_cost),
                $(epoch_record.run_time)")
end

end
