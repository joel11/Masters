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

export RunFFNConfigurationTest, RunSAEConfigurationTest

function PrepareData(data_config, dataset)
    data_raw = dataset == nothing ? GenerateDataset(data_config.data_seed, data_config.steps, data_config.variation_values) : dataset
    processed_data = ProcessData(data_raw, data_config.deltas, data_config.prediction_steps)
    standardized_data = map(x -> StandardizeData(x)[1], processed_data)
    data_splits = map(df -> SplitData(df, data_config.process_splits), standardized_data)

    saesgd_data = CreateDataset(data_splits[1][1], data_splits[2][1], data_config.training_splits)
    ogd_data = CreateDataset(data_splits[1][2], data_splits[2][2], [1.0])

    return(saesgd_data, ogd_data)
end

function RunSAEConfigurationTest(ep::SAEExperimentConfig, dataset)

    srand(ep.seed)

    ################################################################################
    #a. Record all config
    config_id = RecordSAEExperimentConfig(ep)

    ################################################################################
    #b. Prepare data accordingly
    saesgd_data = PrepareData(ep.data_config, dataset)[1]

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
    data_pair = (actual_data, reconstructed_data)

    return (config_id, ep.experiment_set_name, data_pair, sgd_records, ffdata, full_network)
end


function RunFFNConfigurationTest(ep::FFNExperimentConfig, dataset)

    srand(ep.seed)

    ##Data Processing
    sae_network, data_config = ReadSAE(ep.sae_config_id)
    encoder = GetAutoencoder(network)
    saesgd_data, ogd_data = PrepareData(data_config, dataset)
    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, encoder)
    encoded_ogd_dataset = GenerateEncodedOGDDataset(ogd_data, encoder)

    ## FFN-SGD Training
    ffn_network = (ep.rbm_pretraining == true ? (TrainRBMNetwork(config_id, encoded_dataset, ep.ffn_network, ep.rbm_cd)[1])
                                              : NeuralNetwork(ep.ffn_network.layer_sizes, ep.ffn_network.layer_activations, ep.ffn_network.initialization))
    ffn_sgd_records = RunSGD(config_id, "FFN-SGD", encoded_dataset, ffn_network, ep.ffn_sgd)

    ## OGD Training
    ogd_records, comparisons = RunOGD(config_id, "OGD", encoded_ogd_dataset, ffn_network, ep.ogd)

    ## Record Predictions vs Actual
    actual = comparisons[1]
    predicted = comparisons[2]
    CreatePredictionRecords(config_id, actual, predicted)
    return (config_id)
end

end
