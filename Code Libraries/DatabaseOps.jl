module DatabaseOps

using SQLite
using TrainingStructures
using BSON
using DataProcessor
export CreateOOSTradeCostRecords,CreateOOSTradeRecords, CreateISTradeCostRecords, CreateISTradeRecords, CreateSRRecordsCost, CreateConfusionRecords, CreateSRRecords, CreateMMSTradeCostRecords, WriteFFN, CreateBacktestRecords, CreateMMSTradeRecords, WriteSAE, ReadSAE, RecordSAEExperimentConfig, RecordFFNExperimentConfig, CreateEpochRecord, CreatePredictionRecords, RunQuery

#db = SQLite.DB("database_actual.db")
db = SQLite.DB("database_actual_agg.db")
#db = SQLite.DB("database_new.db")
#db_test  = SQLite.DB("database_test.db")

function WriteSAE(config_id, experiment_config, net)
    file_name = string("./SAERepo/SAE_", config_id, ".bson")
    values = Dict(:config_id => config_id, :data_configuration => experiment_config.data_config, :sae => net)
    bson(file_name, values)
end

function WriteFFN(config_id, experiment_config, net)
    file_name = string("./FFNRepo/FFN_", config_id, ".bson")
    values = Dict(:config_id => config_id, :data_configuration => experiment_config.data_config, :ffn => net)
    bson(file_name, values)
end

function ReadSAE(config_id)
    ln = BSON.load(string("./SAERepo/SAE_", config_id, ".bson"))
    return (ln[:sae], ln[:data_configuration])
end

function ReadFFN(config_id)
    ln = BSON.load(string("./FFNRepo/FFN_", config_id, ".bson"))
    return (ln[:ffn], ln[:data_configuration])
end

function CreateConfigurationRecord(seed, set_name, rbm_pretraining, sae_config_id)
    ts = Dates.now()
    cmd_record = string("INSERT INTO configuration_run (seed_used, experiment_set_name, rbm_pretraining, sae_config_id, start_time) values($(seed), '$(set_name)', $(rbm_pretraining), $(sae_config_id), '$(ts)')")
    SQLite.execute!(db, cmd_record)
    max_id = get(SQLite.query(db, "select max(configuration_id) from configuration_run")[1,1])
    return max_id
end

function CreateNetworkRecord(config_id, parameters)
    ls = mapreduce(x -> string(x, ","), string, parameters.layer_sizes)[1:(end-1)]
    la = mapreduce(x -> string(split(string(x), ".")[end], ","), string, parameters.layer_activations)[1:(end-1)]
    init = split(string(parameters.initialization), ".")[end]

    outputfunction = split(string(parameters.output_activation), ".")[end]
    if parameters.encoding_activation == nothing
        encodingfunction = "None"
    else
        encodingfunction = split(string(parameters.encoding_activation), ".")[end]
    end

    network_cmd = "insert into network_parameters
            (configuration_id, category, layer_sizes, layer_activations, initialization, output_activation, encoding_activation)
            values ($(config_id), '$(parameters.category)', '$(ls)', '$(la)', '$(init)', '$(outputfunction)', '$(encodingfunction)')"

    SQLite.execute!(db, network_cmd)
end

function CreateTrainingRecord(config_id, parameters)
    cf = split(string(typeof(parameters.cost_function)), ".")[end]
    sf = string(split(string(parameters.stopping_function), ".")[end], string(parameters.stopping_parameters))

    training_cmd = "insert into training_parameters
            (configuration_id, category, learning_rate, minibatch_size, max_epochs, l1_lambda, cost_function, stopping_function, min_learning_rate, epoch_cycle_max, is_denoising, denoising_variance)
            values ($(config_id), '$(parameters.category)', $(parameters.max_learning_rate), $(parameters.minibatch_size), $(parameters.max_epochs),
            $(parameters.l1_lambda),'$(cf)', '$(sf)', $(parameters.min_learning_rate), $(parameters.epoch_cycle_max), $(parameters.is_denoising), $(parameters.denoising_variance))"

    SQLite.execute!(db, training_cmd)
end

function CreateOGDTrainingRecord(config_id, parameters)
    cf = split(string(typeof(parameters.cost_function)), ".")[end]

    training_cmd = "insert into training_parameters
            (configuration_id, category, learning_rate, minibatch_size, max_epochs, l1_lambda, cost_function, stopping_function)
            values ($(config_id), '$(parameters.category)', $(parameters.max_learning_rate), 1, 1, null, '$(cf)',  null)"

    SQLite.execute!(db, training_cmd)
end

function CreateEpochRecord(config_id, epoch_record)
    ts = Dates.now()

    ctraining = isnan(epoch_record.training_cost)? "null" : epoch_record.training_cost
    ctest = isnan(epoch_record.test_cost)? "null" : epoch_record.test_cost

    training_cmd = "insert into epoch_records
            (configuration_id, category, record_time, epoch_number, training_cost, testing_cost, run_time, learning_rate)
            values ($(config_id), '$(epoch_record.category)', '$(ts)',  $(epoch_record.epoch_number),
            $(ctraining), $(ctest), $(epoch_record.run_time), $(epoch_record.learning_rate))"

    SQLite.execute!(db, training_cmd)
end

function CreateDatasetConfigRecord(config_id, dataset_config)

    deltas_val = mapreduce(x -> string(x, ","), string, dataset_config.deltas)[1:(end-1)]
    processsplit_val = mapreduce(x -> string(x, ","), string, dataset_config.process_splits)[1:(end-1)]
    trainingsplit_val = mapreduce(x -> string(x, ","), string, dataset_config.training_splits)[1:(end-1)]
    predictions_val = mapreduce(x -> string(x, ","), string, dataset_config.prediction_steps)[1:(end-1)]
    variation_vals = mapreduce(x -> string(x, ","), string, dataset_config.variation_values)[1:(end-1)]
    scaling = split(string(dataset_config.scaling_function), ".")[2]
    training_cmd = "insert into dataset_config
            (configuration_id, data_seed, category, steps, deltas, process_splits, training_splits, prediction_steps, variation_values, scaling_function)
            values ($(config_id), $(dataset_config.data_seed),  '$(dataset_config.category)', $(dataset_config.steps), '$(deltas_val)', '$(processsplit_val)',
             '$(trainingsplit_val)', '$(predictions_val)', '$(variation_vals)', '$scaling')"

    SQLite.execute!(db, training_cmd)
end

function CreatePredictionRecords(config_id, actual, predictions)
    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end


    for c in 1:size(actual)[2]
        for r in 1:size(actual)[1]
            push!(records, (string("(", config_id,",", r, ",'", names(actual)[c],"',", NanRemover(actual[r, c]),",", NanRemover(predictions[r,c]), ")")))
        end
    end

    prediction_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    prediction_cmd = "insert into prediction_results (configuration_id, time_step, stock, actual, predicted) values $(prediction_values)"
    SQLite.execute!(db, prediction_cmd)
end

function CreateBacktestRecords(config_id, actual, predictions)
    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end


    for c in 1:size(actual)[2]
        for r in 1:size(actual)[1]
            push!(records, (string("(", config_id,",", r, ",'", names(actual)[c],"',", NanRemover(actual[r, c]),",", NanRemover(predictions[r,c]), ")")))
        end
    end

    prediction_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    prediction_cmd = "insert into backtest_results (configuration_id, time_step, stock, actual, predicted) values $(prediction_values)"
    SQLite.execute!(db, prediction_cmd)
end

function CreateISTradeRecords(mms_returns)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(mms_returns)[1]
        push!(records, (string("(", mms_returns[r,1],",", mms_returns[r,2], ",", mms_returns[r,3], ",", NanRemover(mms_returns[r,4]), ")")))
    end

    mms_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    mms_cmd = "insert into config_is_trade_returns (configuration_id, time_step, total_profit_observed, total_profit_rate_observed) values $(mms_values)"
    SQLite.execute!(db, mms_cmd)
end

function CreateISTradeCostRecords(mms_returns)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(mms_returns)[1]
        push!(records, (string("(", mms_returns[r,1],",", mms_returns[r,2], ",", mms_returns[r,3], ",", NanRemover(mms_returns[r,4]), ")")))
    end

    mms_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    mms_cmd = "insert into config_is_trade_returns_cost (configuration_id, time_step, total_profit_observed, total_profit_rate_observed) values $(mms_values)"
    SQLite.execute!(db, mms_cmd)
end

function CreateOOSTradeRecords(mms_returns)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(mms_returns)[1]
        push!(records, (string("(", mms_returns[r,1],",", mms_returns[r,2], ",", mms_returns[r,3], ",", NanRemover(mms_returns[r,4]), ")")))
    end

    mms_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    mms_cmd = "insert into config_oos_trade_returns (configuration_id, time_step, total_profit_observed, total_profit_rate_observed) values $(mms_values)"
    SQLite.execute!(db, mms_cmd)
end

function CreateOOSTradeCostRecords(mms_returns)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(mms_returns)[1]
        push!(records, (string("(", mms_returns[r,1],",", mms_returns[r,2], ",", mms_returns[r,3], ",", NanRemover(mms_returns[r,4]), ")")))
    end

    mms_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    mms_cmd = "insert into config_oos_trade_returns_cost (configuration_id, time_step, total_profit_observed, total_profit_rate_observed) values $(mms_values)"
    SQLite.execute!(db, mms_cmd)
end

function RunQuery(query, testDB = false)

    if testDB == false
        return(SQLite.query(db, query))
    else
        return(SQLite.query(db_test, query))
    end
end

function RecordSAEExperimentConfig(exp_config)
    config_id = CreateConfigurationRecord(exp_config.seed, exp_config.experiment_set_name, exp_config.rbm_pretraining, 0)
    CreateDatasetConfigRecord(config_id, exp_config.data_config)
    CreateNetworkRecord(config_id, exp_config.sae_network)
    CreateTrainingRecord(config_id, exp_config.sae_sgd)

    if exp_config.rbm_pretraining == true
        CreateTrainingRecord(config_id, exp_config.rbm_cd)
    end

    return config_id
end

function RecordFFNExperimentConfig(exp_config)
    config_id = CreateConfigurationRecord(exp_config.seed, exp_config.experiment_set_name, exp_config.rbm_pretraining, exp_config.sae_config_id)
    CreateDatasetConfigRecord(config_id, exp_config.data_config)

    CreateNetworkRecord(config_id, exp_config.ffn_network)
    CreateTrainingRecord(config_id, exp_config.ffn_sgd)
    CreateOGDTrainingRecord(config_id, exp_config.ogd)
    #CreateTrainingRecord(config_id, exp_config.ogd_ho)

    return config_id
end

function CreateSRRecords(sharpe_ratios)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(sharpe_ratios)[1]
        push!(records, (string("(", sharpe_ratios[r,1],",", NanRemover(sharpe_ratios[r,2]), ")")))
    end

    sr_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    sr_cmd = "insert into config_oos_sharpe_ratio (configuration_id, sharpe_ratio) values $(sr_values)"
    SQLite.execute!(db, sr_cmd)
end

function CreateSRRecordsCost(sharpe_ratios)

    records = []

    function NanRemover(x)
        if (isnan(x) || isinf(x))
            return "null"
        end
        return x
    end

    for r in 1:size(sharpe_ratios)[1]
        push!(records, (string("(", sharpe_ratios[r,1],",", NanRemover(sharpe_ratios[r,2]), ")")))
    end

    sr_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    sr_cmd = "insert into config_oos_sharpe_ratio_cost (configuration_id, sharpe_ratio) values $(sr_values)"
    SQLite.execute!(db, sr_cmd)
end

function CreateConfusionRecords(confusion_df)

    records = []

    for r in 1:size(confusion_df)[1]
        push!(records, (string("(", confusion_df[r,1],",", confusion_df[r,2], ",", confusion_df[r,3], ",", confusion_df[r,4], ")")))
    end

    confusion_values = (mapreduce(x->string(x, ","), string, records)[1:(end-1)])
    confusion_cmd = "insert into config_confusion (configuration_id, trades_percentage, no_trades_percentage, all_trades_percentage) values $(confusion_values)"
    SQLite.execute!(db, confusion_cmd)
end

end
