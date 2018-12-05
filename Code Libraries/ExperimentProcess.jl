module ExperimentProcess

using DatabaseOps

export RunConfigurationTest

function RunConfigurationTest(ep)

    srand(ep.seed)

    ################################################################################
    #a. Record all config
    config_id = RecordExperimentConfig(ep)

    ################################################################################
    #b. Prepare data accordingly
    data_raw = GenerateDataset(ep.data_config.data_seed, ep.data_config.steps, ep.data_config.variation_values)
    data_splits = SplitData(data_raw, ep.data_config.process_splits)
    processed_data = map(x -> ProcessData(x, ep.data_config.deltas, ep.data_config.prediction_steps), data_splits)
    saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], ep.data_config.training_splits), processed_data)

    ################################################################################
    #c. Run training, and record all epochs

    ## SAE Training & Encoding
    sae_network, sgd_records = TrainInitSAE(config_id, "SAE-SGD", saesgd_data, ep.sae_network, ep.sae_sgd, LinearActivation)
    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, sae_network)

    ## FFN-SGD Training
    ffn_network = NeuralNetwork(ep.ffn_network.layer_sizes, ep.ffn_network.layer_activations, ep.ffn_network.initialization)
    ffn_sgd_records = RunSGD(config_id, "FFN-SGD", encoded_dataset, ep.ffn_network, ep.ffn_sgd)

    ## OGD Training
    encoded_ogd_dataset = GenerateEncodedSGDDataset(ogd_dataset, sae_network)
    ogd_records = RunOGD(config_id, "OGD", encoded_ogd_dataset, ffn_network, ep.ogd)

    ## Holdout
    ## Use validation data from OGD dataset
    encoded_holdout_dataset = GenerateEncodedSGDDataset(holdout_dataset, sae_network)
    holdout_records, comparisons = RunOGD(config_id, "OGD-HO", encoded_holdout_dataset, ffn_network,  ep.ogd_ho)


    ## Record Predictions vs Actual
    actual = comparisons[1]
    predicted = comparisons[2]
    CreatePredictionRecords(config_id, actual, predicted)

end

end
