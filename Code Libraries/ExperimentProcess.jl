module ExperimentProcess

#workspace()
#push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
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
    full_dataset = PrepareData(ep.data_config, dataset)
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

    mape = round(CalculateMAPE(deprocessed_actual, deprocessed_recon), 2)
    CreateMapeRecord(config_id, mape)

    return (config_id, ep.experiment_set_name, data_pair, sgd_records, ffdata, full_network)
end

function RunFFNConfigurationTest(ep::FFNExperimentConfig, dataset)

    srand(ep.seed)
    config_id = RecordFFNExperimentConfig(ep)

    ##Data Processing
    saesgd_data, ogd_data = PrepareData(ep.data_config, dataset)
    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, ep.auto_encoder, true)
    encoded_ogd_dataset = GenerateEncodedOGDDataset(ogd_data, ep.auto_encoder, true)

    ## FFN-SGD Training
    #ffn_network = (ep.rbm_pretraining == true ? (TrainRBMNetwork(config_id, encoded_dataset, ep.ffn_network, ep.rbm_cd)[1])
    #                                          : NeuralNetwork(ep.ffn_network.layer_sizes, ep.ffn_network.layer_activations, ep.ffn_network.initialization))

    ffn_sgd_records, ffn_network = TrainInitFFN(config_id, "FFN-SGD", encoded_dataset, ep.ffn_network, ep.ffn_sgd)

    ## OGD Training
    ogd_records, comparisons = RunOGD(config_id, "OGD", encoded_ogd_dataset, ffn_network, ep.ogd)

    ## Record Predictions vs Actual
    actual = DataFrame(comparisons[1])
    predicted = DataFrame(comparisons[2])

    reverse_function = ReverseFunctions[ep.data_config.scaling_function]
    deprocessed_actual = reverse_function(actual, ogd_data.output_processingvar1, ogd_data.output_processingvar2)
    deprocessed_predicted = reverse_function(predicted, ogd_data.output_processingvar1, ogd_data.output_processingvar2)
    reconstructed_actual = ReconstructPrices(deprocessed_actual, ep.data_config, ogd_data.original_prices)
    reconstructed_predicted = ReconstructPrices(deprocessed_predicted, ep.data_config, ogd_data.original_prices)

    reconstructed_actual = DataFrame(reconstructed_actual)
    reconstructed_predicted = DataFrame(reconstructed_predicted)

    #reconstructed_actual = DataFrame(deprocessed_actual)
    #reconstructed_predicted = DataFrame(deprocessed_predicted)

    names!(reconstructed_actual, names(encoded_ogd_dataset.training_output))
    names!(reconstructed_predicted, names(encoded_ogd_dataset.training_output))

    CreatePredictionRecords(config_id, reconstructed_actual, reconstructed_predicted)
    return (config_id, actual, predicted, ffn_sgd_records, ffn_network)
end

end
