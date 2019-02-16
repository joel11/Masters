module TrainingStructures

using DataFrames, NeuralNetworks, CostFunctions

export TrainingParameters, EpochRecord, PrintEpoch, DataSet, NetworkParameters, DatasetConfig, SAEExperimentConfig, FFNExperimentConfig, OGDTrainingParameters

type DataSet
    training_input::DataFrame #::Array{Float64,2}
    testing_input::DataFrame #Array{Float64,2}

    training_output::DataFrame# ::Array{Float64,2}
    testing_output::DataFrame #Array{Float64,2}

    standardizing_means
    standardizing_deviations

    input_column_names
    output_column_names

end

type TrainingParameters
    category::String
    max_learning_rate::Float64
    min_learning_rate::Float64
    epoch_cycle_max::Float64
    minibatch_size::Int64
    momentum_rate::Float64
    max_epochs::Int64
    stopping_parameters::Tuple
    stopping_function
    verbose::Bool
    is_classification::Bool

    l1_lambda::Float64
    l2_lambda::Float64
    cost_function

    function TrainingParameters(category, max_learning_rate, min_learning_rate, epoch_cycle_max, minibatch_size, momentum_rate, max_epochs,
        stopping_parameters, stopping_function, verbose, is_classification, l1_lambda, l2_lambda, cost_function)
        return new(category, max_learning_rate, min_learning_rate, epoch_cycle_max, minibatch_size, momentum_rate, max_epochs,
            stopping_parameters, stopping_function(stopping_parameters), verbose, is_classification, l1_lambda, l2_lambda, cost_function)
    end
end

type OGDTrainingParameters
    category::String
    max_learning_rate::Float64
    verbose::Bool
    cost_function

    #momentum_rate::Float64
    #l1_lambda::Float64
    #l2_lambda::Float64

    function OGDTrainingParameters(category, max_learning_rate, verbose, cost_function)
        return new(category, max_learning_rate, verbose, cost_function)
    end
end

type NetworkParameters
    category::String
    layer_sizes::Array{Int64}
    layer_activations::Array{Function}
    initialization::Function
    output_activation
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

type SAEExperimentConfig

    seed::Int64
    experiment_set_name::String
    rbm_pretraining::Bool
    data_config::DatasetConfig

    sae_network
    sae_sgd
    rbm_cd

end

type FFNExperimentConfig

    seed::Int64
    experiment_set_name::String
    rbm_pretraining::Bool
    data_config::DatasetConfig

    sae_config_id
    auto_encoder::NeuralNetwork
    ffn_network
    ffn_sgd
    ogd::OGDTrainingParameters
    rbm_cd
end

type EpochRecord

    epoch_number::Int64
    category::String
    training_cost::Float64
    test_cost::Float64

    training_accuracy::Float64
    test_accuracy::Float64

    energy_ratio::Float64

    run_time::Float64
    network::NeuralNetworks.NeuralNetwork
    weight_change_rates
    hidden_activation_likelihoods

    mean_weight_changes
    zero_activation_perc

    function EpochRecord(epoch_number, category, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods, mean_weight_changes, zero_activation_perc)
        return new(epoch_number, category, training_cost, test_cost, training_accuracy, test_accuracy, energy_ratio, run_time, network, weight_change_rates, hidden_activation_likelihoods, mean_weight_changes, zero_activation_perc)
    end
end

function PrintEpoch(epoch_record::EpochRecord)
    println("Epoch $(epoch_record.epoch_number) Training Cost, Test Cost, Time:$(epoch_record.training_cost), $(epoch_record.test_cost), $(epoch_record.run_time)")
end

end
