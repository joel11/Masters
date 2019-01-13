module TrainingStructures

using NeuralNetworks, CostFunctions

export TrainingParameters, EpochRecord, PrintEpoch, DataSet, NetworkParameters, DatasetConfig, ExperimentConfig

type DataSet
    training_input#::Array{Float64,2}
    testing_input#::Array{Float64,2}
    #validation_input#::Array{Float64,2}

    training_output#::Array{Float64,2}
    testing_output#::Array{Float64, 2}
    #validation_output#::Array{Float64,2}

    scaling_min
    scaling_max

end

type TrainingParameters
    category::String
    learning_rate::Float64
    minibatch_size::Int64
    momentum_rate::Float64
    max_epochs::Int64
    stopping_parameters::Tuple
    stopping_function
    verbose::Bool
    is_classification::Bool

    l1_lambda::Float64
    l2_lambda::Float64
    cost_function::CostFunction


    function TrainingParameters(category, learning_rate, minibatch_size, momentum_rate, max_epochs,
        stopping_parameters, stopping_function, verbose, is_classification, l1_lambda, l2_lambda, cost_function)
        return new(category, learning_rate, minibatch_size, momentum_rate, max_epochs,
            stopping_parameters, stopping_function(stopping_parameters), verbose, is_classification, l1_lambda, l2_lambda, cost_function)
    end
end

type NetworkParameters
    category::String
    layer_sizes::Array{Int64}
    layer_activations::Array{Function}
    initialization::Function
end

type DatasetConfig

    data_seed::Int64
    category::String
    steps::Int64
    deltas::Array{Int64}
    process_splits::Array{Float64}
    training_splits::Array{Float64}
    prediction_steps::Array{Int64}
    variation_values

    function DatasetConfig(data_seed, category, steps, deltas, process_splits, training_splits, prediction_steps, variation_values)
        return new(data_seed, category, steps, deltas, process_splits, training_splits, prediction_steps, variation_values)
    end
end

type ExperimentConfig

    seed::Int64
    experiment_set_name::String
    rbm_pretraining::Bool
    data_config::DatasetConfig

    sae_network
    ffn_network
    sae_sgd
    ffn_sgd
    rbm_cd
    ogd
    #ogd_ho
    sae_only
end

type EpochRecord

    epoch_number::Int64
    category::String
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

    mean_weight_changes

    function EpochRecord(epoch_number, category, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods, mean_weight_changes)
        return new(epoch_number, category, mean_minibatch_cost, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods, mean_weight_changes)
    end
end

function PrintEpoch(epoch_record::EpochRecord)
    println("Epoch $(epoch_record.epoch_number) Training Cost, Test Cost, Time:$(epoch_record.training_cost), $(epoch_record.test_cost), $(epoch_record.run_time)")
end

end
