module ExperimentProcess

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
using FinancialFunctions
using DatabaseOps
using HyperparameterOptimization
using ExperimentProcess

export RunConfigurationTest, RunSAEConfigurationTest

function RunSAEConfigurationTest(ep, dataset)

    srand(ep.seed)

    ################################################################################
    #a. Record all config
    config_id = RecordExperimentConfig(ep)

    ################################################################################
    #b. Prepare data accordingly
    data_raw = dataset == nothing ? GenerateDataset(ep.data_config.data_seed, ep.data_config.steps, ep.data_config.variation_values) : dataset
    data_splits = SplitData(data_raw, ep.data_config.process_splits)
    processed_data = map(x -> ProcessData(x, ep.data_config.deltas, ep.data_config.prediction_steps), data_splits)
    #saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], ep.data_config.training_splits), processed_data)
    #saesgd_data = NormalizeDatasetForTanh(CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits))
    saesgd_data = NormalizeDatasetForSigmoid(CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits))
    #saesgd_data = CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits)
    ################################################################################
    #c. Run training, and record all epochs

    ## SAE Training & Encoding
    training_objects = (ep.rbm_pretraining == true ? (TrainRBMSAE(config_id, "SAE-SGD-RBM", saesgd_data, ep.sae_network, ep.rbm_cd, ep.sae_sgd))
                                              : (TrainInitSAE(config_id, "SAE-SGD-Init", saesgd_data, ep.sae_network, ep.sae_sgd)))


    full_network = training_objects[end]
    sgd_records = training_objects[(end-1)]
    actual_data = saesgd_data.testing_input

    ffdata = Feedforward(full_network, actual_data)
    reconstructed_data = ffdata[end]
    #data_pair = (actual_data, reconstructed_data)
    data_pair = (DenormalizatData(actual_data, saesgd_data.scaling_min, saesgd_data.scaling_max)
                , DenormalizatData(reconstructed_data, saesgd_data.scaling_min, saesgd_data.scaling_max))

    return (config_id, ep.experiment_set_name, data_pair, sgd_records, ffdata)
end

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
    #saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], ep.data_config.training_splits), processed_data)
    saesgd_data = CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits)
    ogd_data = CreateDataset(processed_data[2][1], processed_data[2][2], [1.0])

    ################################################################################
    #c. Run training, and record all epochs

    ## SAE Training & Encoding
    sae_network = (ep.rbm_pretraining == true ? (TrainRBMSAE(config_id, "SAE-SGD-RBM", saesgd_data, ep.sae_network, ep.rbm_cd, ep.sae_sgd)[1])
                                              : (TrainInitSAE(config_id, "SAE-SGD-Init", saesgd_data, ep.sae_network, ep.sae_sgd, LinearActivation)[1]))
    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, sae_network)

    ## FFN-SGD Training
    ffn_network = (ep.rbm_pretraining == true ? (TrainRBMNetwork(config_id, encoded_dataset, ep.ffn_network, ep.rbm_cd)[1])
                                              : NeuralNetwork(ep.ffn_network.layer_sizes, ep.ffn_network.layer_activations, ep.ffn_network.initialization))
    ffn_sgd_records = RunSGD(config_id, "FFN-SGD", encoded_dataset, ffn_network, ep.ffn_sgd)

    ## OGD Training
    encoded_ogd_dataset = GenerateEncodedOGDDataset(ogd_data, sae_network)
    ogd_records, comparisons = RunOGD(config_id, "OGD", encoded_ogd_dataset, ffn_network, ep.ogd)

    ## Holdout
    ## Use validation data from OGD dataset
    #encoded_holdout_dataset = GenerateEncodedSGDDataset(holdout_data, sae_network)
    #holdout_records, comparisons = RunOGD(config_id, "OGD-HO", encoded_holdout_dataset, ffn_network,  ep.ogd_ho)

    ## Record Predictions vs Actual
    actual = comparisons[1]
    predicted = comparisons[2]
    CreatePredictionRecords(config_id, actual, predicted)
    return (config_id)
end

end
