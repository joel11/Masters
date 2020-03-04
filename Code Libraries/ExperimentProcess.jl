module ExperimentProcess

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator

export RunFFNConfigurationTest, RunSAEConfigurationTest

function RunSAEConfigurationTest(ep::SAEExperimentConfig, dataset)

    srand(ep.seed)

    ################################################################################
    #a. Record all config
    config_id = RecordSAEExperimentConfig(ep)

    ################################################################################
    #b. Prepare data accordingly
    full_dataset = PrepareData(ep.data_config, dataset, ep.sae_sgd)
    saesgd_data = full_dataset[1]

    ################################################################################
    #c. Run training, and record all epochs

    ## SAE Training & Encoding
    training_objects = (ep.rbm_pretraining == true  ? (TrainRBMSAE(config_id, "SAE-SGD-RBM", saesgd_data, ep.sae_network, ep.rbm_cd, ep.sae_sgd))
                                                    : (TrainInitSAE(config_id, "SAE-SGD-Init", saesgd_data, ep.sae_network, ep.sae_sgd)))

    full_network = training_objects[end]
    sgd_records = training_objects[(end-1)]

    actual_data = saesgd_data.training_input[1:Int64(floor(size(saesgd_data.training_input,1)*0.2)),:]
    ffdata = Feedforward(full_network, actual_data)
    reconstructed_data = ffdata[end]

    reverse_function = ReverseFunctions[ep.data_config.scaling_function]
    deprocessed_actual = reverse_function(actual_data, saesgd_data.input_processingvar1, saesgd_data.input_processingvar2)
    deprocessed_recon = reverse_function(reconstructed_data, saesgd_data.input_processingvar1, saesgd_data.input_processingvar2)
    #reconstructed_actual = ReconstructPrices(deprocessed_actual, ep.data_config, actual_data)
    #reconstructed_recon = ReconstructPrices(deprocessed_recon, ep.data_config, actual_data)

    data_pair = (deprocessed_actual, deprocessed_recon)

    return (config_id, ep.experiment_set_name, data_pair, sgd_records, ffdata, full_network)
end

function RunFFNConfigurationTest(ep::FFNExperimentConfig, dataset)

    srand(ep.seed)
    config_id = RecordFFNExperimentConfig(ep)

    ##Data Processing
    saesgd_data, ogd_data = PrepareData(ep.data_config, dataset, ep.ffn_sgd)
    #encoded_dataset =  saesgd_data #GenerateEncodedSGDDataset(saesgd_data, ep.auto_encoder, true)
    #encoded_ogd_dataset = ogd_data #GenerateEncodedOGDDataset(ogd_data, ep.auto_encoder, true)

    #for backtest predictions
    ffn_conf = deepcopy(ep.ffn_sgd)
    ffn_conf.training_splits = [1.0]
    backtest_saesgd_data, backtest_ogd_data = PrepareData(ep.data_config, dataset, ffn_conf)

    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, ep.auto_encoder, true)
    encoded_ogd_dataset = GenerateEncodedOGDDataset(ogd_data, ep.auto_encoder, true)
    encoded_backtests_dataset = GenerateEncodedOGDDataset(backtest_saesgd_data, ep.auto_encoder, true)

    ## FFN-SGD Training
    #ffn_network = (ep.rbm_pretraining == true ? (TrainRBMNetwork(config_id, encoded_dataset, ep.ffn_network, ep.rbm_cd)[1])
    #                                          : NeuralNetwork(ep.ffn_network.layer_sizes, ep.ffn_network.layer_activations, ep.ffn_network.initialization))
    ffn_sgd_records, ffn_network = TrainInitFFN(config_id, "FFN-SGD", encoded_dataset, ep.ffn_network, ep.ffn_sgd)

    println("SGD Done")

    actual = encoded_backtests_dataset.training_output
    predicted =  Feedforward(ffn_network, encoded_backtests_dataset.training_input)[end]
    sgd_reconstructed_actual, sgd_reconstructed_predicted = ReconstructSGDPredictions(ep.data_config, actual, predicted, backtest_saesgd_data, encoded_backtests_dataset)
    CreateBacktestRecords(config_id, sgd_reconstructed_actual, sgd_reconstructed_predicted)

    WriteFFN(config_id, ep, ffn_network)

    ## OGD Training
    ogd_records, comparisons = RunOGD(config_id, "OGD", encoded_ogd_dataset, ffn_network, ep.ogd)

    println("OGD Done")

    ## Record Predictions vs Actual
    actual = DataFrame(comparisons[1])
    predicted = DataFrame(comparisons[2])
    ogd_reconstructed_actual, ogd_reconstructed_predicted = ReconstructPredictions(ep.data_config, actual, predicted, ogd_data, encoded_ogd_dataset)
    CreatePredictionRecords(config_id, ogd_reconstructed_actual, ogd_reconstructed_predicted)


    return (config_id, actual, predicted, ffn_sgd_records, ffn_network)
end

function ReconstructSGDPredictions(data_config, actual, predicted, prepared_dataset, encoded_dataset)

    reverse_function = ReverseFunctions[data_config.scaling_function]

    deprocessed_actual = reverse_function(actual, prepared_dataset.output_processingvar1, prepared_dataset.output_processingvar2)
    deprocessed_predicted = reverse_function(predicted, prepared_dataset.output_processingvar1, prepared_dataset.output_processingvar2)

    reconstructed_actual = ReconstructSGDPrices(deprocessed_actual, data_config, prepared_dataset.original_prices)
    reconstructed_predicted = ReconstructSGDPrices(deprocessed_predicted, data_config, prepared_dataset.original_prices)

    reconstructed_actual = DataFrame(reconstructed_actual)
    reconstructed_predicted = DataFrame(reconstructed_predicted)

    names!(reconstructed_actual, names(encoded_dataset.training_output))
    names!(reconstructed_predicted, names(encoded_dataset.training_output))

    return (reconstructed_actual, reconstructed_predicted)
end

function ReconstructPredictions(data_config, actual, predicted, prepared_dataset, encoded_dataset)

    reverse_function = ReverseFunctions[data_config.scaling_function]

    deprocessed_actual = reverse_function(actual, prepared_dataset.output_processingvar1, prepared_dataset.output_processingvar2)
    deprocessed_predicted = reverse_function(predicted, prepared_dataset.output_processingvar1, prepared_dataset.output_processingvar2)

    reconstructed_actual = ReconstructPrices(deprocessed_actual, data_config, prepared_dataset.original_prices)
    reconstructed_predicted = ReconstructPrices(deprocessed_predicted, data_config, prepared_dataset.original_prices)

    reconstructed_actual = DataFrame(reconstructed_actual)
    reconstructed_predicted = DataFrame(reconstructed_predicted)

    names!(reconstructed_actual, names(encoded_dataset.training_output))
    names!(reconstructed_predicted, names(encoded_dataset.training_output))

    return (reconstructed_actual, reconstructed_predicted)
end

end
