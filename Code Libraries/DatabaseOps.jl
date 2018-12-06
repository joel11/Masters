module DatabaseOps

using SQLite
using TrainingStructures

export RecordExperimentConfig, CreateEpochRecord, CreatePredictionRecords

db = SQLite.DB("database_test")

function CreateConfigurationRecord(seed, set_name)
    ts = Dates.now()
    cmd_record = string("INSERT INTO configuration_run (seed_used, experiment_set_name, start_time) values($(seed), '$(set_name)', '$(ts)')")
    SQLite.execute!(db, cmd_record)
    max_id = get(SQLite.query(db, "select max(configuration_id) from configuration_run")[1,1])
    return max_id
end

function CreateNetworkRecord(config_id, parameters)
    ls = mapreduce(x -> string(x, ","), string, parameters.layer_sizes)[1:(end-1)]
    la = mapreduce(x -> string(split(string(x), ".")[end], ","), string, parameters.layer_activations)[1:(end-1)]
    init = split(string(parameters.initialization), ".")[end]
    network_cmd = "insert into network_parameters
            (configuration_id, category, layer_sizes, layer_activations, initialization)
            values ($(config_id), '$(parameters.category)', '$(ls)', '$(la)', '$(init)')"

    SQLite.execute!(db, network_cmd)
end

function CreateTrainingRecord(config_id, parameters)
    cf = split(string(typeof(parameters.cost_function)), ".")[end]
    sf = "unset"

    training_cmd = "insert into training_parameters
            (configuration_id, category, learning_rate, minibatch_size, max_epochs, l1_lambda, l2_lambda, cost_function, stopping_function)
            values ($(config_id), '$(parameters.category)', $(parameters.learning_rate), $(parameters.minibatch_size), $(parameters.max_epochs),
            $(parameters.l1_lambda), $(parameters.l2_lambda), '$(cf)', '$(sf)')"

    SQLite.execute!(db, training_cmd)
end

function CreateEpochRecord(config_id, epoch_record)
    ts = Dates.now()

    cminibatch = isnan(epoch_record.mean_minibatch_cost) ? "null" : epoch_record.mean_minibatch_cost
    ctraining = isnan(epoch_record.training_cost)? "null" : epoch_record.training_cost
    ctest = isnan(epoch_record.test_cost)? "null" : epoch_record.test_cost

    training_cmd = "insert into epoch_records
            (configuration_id, category, record_time, epoch_number, mean_minibatch_cost, training_cost, testing_cost, run_time)
            values ($(config_id), '$(epoch_record.category)', '$(ts)',  $(epoch_record.epoch_number),
            $(cminibatch), $(ctraining), $(ctest), $(epoch_record.run_time))"

    SQLite.execute!(db, training_cmd)
end

function CreateDatasetConfigRecord(config_id, dataset_config)

    deltas_val = mapreduce(x -> string(x, ","), string, dataset_config.deltas)[1:(end-1)]
    processsplit_val = mapreduce(x -> string(x, ","), string, dataset_config.process_splits)[1:(end-1)]
    trainingsplit_val = mapreduce(x -> string(x, ","), string, dataset_config.training_splits)[1:(end-1)]
    predictions_val = mapreduce(x -> string(x, ","), string, dataset_config.prediction_steps)[1:(end-1)]
    variation_vals = mapreduce(x -> string(x, ","), string, dataset_config.variation_values)[1:(end-1)]

    training_cmd = "insert into dataset_config
            (configuration_id, data_seed, category, steps, deltas, process_splits, training_splits, prediction_steps, variation_values)
            values ($(config_id), $(dataset_config.data_seed),  '$(dataset_config.category)', $(dataset_config.steps), '$(deltas_val)', '$(processsplit_val)',
             '$(trainingsplit_val)', '$(predictions_val)', '$(variation_vals)')"

    SQLite.execute!(db, training_cmd)

end

function CreatePredictionRecords(config_id, actual, predictions)



end

function RecordExperimentConfig(exp_config)
    config_id = CreateConfigurationRecord(exp_config.seed, exp_config.experiment_set_name)
    CreateDatasetConfigRecord(config_id, exp_config.data_config)
    CreateNetworkRecord(config_id, exp_config.sae_network)
    CreateNetworkRecord(config_id, exp_config.ffn_network)
    CreateTrainingRecord(config_id, exp_config.sae_sgd)
    CreateTrainingRecord(config_id, exp_config.ffn_sgd)
    CreateTrainingRecord(config_id, exp_config.ogd)
    CreateTrainingRecord(config_id, exp_config.ogd_ho)
    return config_id
end

end
