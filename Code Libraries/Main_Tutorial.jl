workspace()

push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using ExperimentProcessTrainSAE


##Train SAE Networks############################################################

sae_experiment_set_name = "SAE Tutorial 1"
dataset_name = "AGL"
jsedata = ReadJSETop40Data()
dataset =  jsedata[:, [:AGL]]# ,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

data_sgd_validation_set_split = [0.8]
data_sgd_ogd_split = [0.6]
data_horizon_predictions = [5]
data_scaling_function = LimitedNormalizeData
data_horizon_aggregations = ([1,5,20], [5,20,60], [10,20,60])

sae_network_initialization_functions = (InitializationFunctions.DCUniformInit,InitializationFunctions.XavierGlorotUniformInit)
sae_network_hidden_layer_activation = LeakyReluActivation
sae_network_output_activation = LinearActivation
sae_network_encoding_activation = LinearActivation
sae_network_layer_sizes = ((20,20,20), (20,20), (20))
sae_network_encoding_layers = (3, 5)

sae_sgd_max_learning_rates = (0.005, 0.01, 0.05, 0.1)
sae_sgd_min_learning_rates = (0.001, 0.0001)
sae_sgd_learning_rate_epoch_length = (100, 200)
sae_sgd_minibatch_size = (32)
sae_sgd_learning_rate_max_epochs = (100, 1000)
sae_sgd_l1_lambda = (0.0)
sae_sgd_validation_set_split = [0.8]
sae_sgd_denoising_enabled = (false)
sae_sgd_denoising_variance = (0.0)

RunSAEExperiment(sae_experiment_set_name,
                            dataset_name,
                            dataset,
                            data_sgd_validation_set_split,
                            data_sgd_ogd_split,
                            data_horizon_predictions,
                            data_scaling_function,
                            data_horizon_aggregations,
                            sae_network_initialization_functions,
                            sae_network_hidden_layer_activation,
                            sae_network_output_activation,
                            sae_network_encoding_activation,
                            sae_network_layer_sizes,
                            sae_network_encoding_layers,
                            sae_sgd_max_learning_rates,
                            sae_sgd_min_learning_rates,
                            sae_sgd_learning_rate_epoch_length,
                            sae_sgd_minibatch_size,
                            sae_sgd_learning_rate_max_epochs,
                            sae_sgd_l1_lambda,
                            sae_sgd_validation_set_split,
                            sae_sgd_denoising_enabled,
                            sae_sgd_denoising_variance)


##Choose SAE Networks###########################################################



##Train FFN Networks############################################################

ffn_experiment_set_name = "FFN Tutorial 1"

sae_choices = (1, 2, 3, 4)

ffn_network_initialization_functions = (InitializationFunctions.DCUniformInit,InitializationFunctions.XavierGlorotUniformInit)
ffn_network_hidden_layer_activation = LeakyReluActivation
ffn_network_output_activation = LinearActivation
ffn_network_layer_sizes = ((20,20,20), (20,20), (20))

ffn_sgd_max_learning_rates = (0.005, 0.01, 0.05, 0.1)
ffn_sgd_min_learning_rates = (0.001, 0.0001)
ffn_sgd_learning_rate_epoch_length = (100, 200)
ffn_sgd_minibatch_size = (32)
ffn_sgd_learning_rate_max_epochs = (100, 1000)
ffn_sgd_l1_lambda = (0.0)
ffn_sgd_validation_set_split = [0.8]
ffn_sgd_denoising_enabled = (false)
ffn_sgd_denoising_variance = (0.0)

ogd_learning_rates = (0.1, 0.01)

RunFFNExperiment(ffn_experiment_set_name, sae_choices,
                            dataset_name,
                            dataset,
                            ffn_network_initialization_functions,
                            ffn_network_hidden_layer_activation,
                            ffn_network_output_activation,
                            ffn_network_layer_sizes,
                            ffn_sgd_max_learning_rates,
                            ffn_sgd_min_learning_rates,
                            ffn_sgd_learning_rate_epoch_length,
                            ffn_sgd_minibatch_size,
                            ffn_sgd_learning_rate_max_epochs,
                            ffn_sgd_l1_lambda,
                            ffn_sgd_validation_set_split,
                            ffn_sgd_denoising_enabled,
                            ffn_sgd_denoising_variance,
                            ogd_learning_rates)


##Run Diagnostics###############################################################
