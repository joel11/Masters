module NetworkTrainer

using RBM, NeuralNetworks, ActivationFunctions, InitializationFunctions, TrainingStructures, SGD, CostFunctions

export TrainRBMSAE, TrainEncoderRBNMFFNNetwork, TrainInitSAE

function CreateEncoderDataset(dataset::DataSet)
    training_input = dataset.training_input
    testing_input = dataset.testing_input
    #validation_input = dataset.validation_input
    #return DataSet(training_input, testing_input, validation_input, training_input, testing_input, validation_input)
    return DataSet(training_input, testing_input, training_input, testing_input, 0, 0)
end

function TrainInitSAE(config_id, category, dataset::DataSet, network_parameters::NetworkParameters, parameters::TrainingParameters, output_function::Function)
    encoder_data = CreateEncoderDataset(dataset)
    network = NeuralNetwork(network_parameters.layer_sizes, network_parameters.layer_activations, network_parameters.initialization)
    AddDecoder(network, network_parameters)
    network.layers[end].activation = output_function
    sgd_records = RunSGD(config_id, category, encoder_data, network, parameters)
    autoencoder = GetAutoencoder(network)
    return (autoencoder, sgd_records, network)
end

function TrainRBMSAE(config_id, category, dataset::DataSet, network_parameters::NetworkParameters, rbm_parameters::TrainingParameters, ffn_parameters::TrainingParameters)
    encoder_data = CreateEncoderDataset(dataset)
    original_functions = copy(network_parameters.layer_activations)

    #network_parameters.layer_activations = GenerateActivationFunctions(length(original_functions))
    rbm_network, rbm_records = RBM.TrainRBMNetwork(config_id,encoder_data, network_parameters, rbm_parameters)
    AddDecoder(rbm_network, network_parameters)
    sgd_records = RunSGD(config_id, category, encoder_data, rbm_network, ffn_parameters)
    autoencoder = GetAutoencoder(rbm_network)
    return (autoencoder, rbm_records, sgd_records, rbm_network)
end

#=function TrainEncoderRBNMFFNNetwork(dataset::DataSet, network_parameters::NetworkParameters, rbm_parameters::TrainingParameters, ffn_parameters::TrainingParameters)

    encoder_data = dataset#CreateEncoderDataset(dataset)
    rbm_network, rbm_records = TrainRBMNetwork(encoder_data, network_parameters, rbm_parameters)
    sgd_records = RunSGD(encoder_data, rbm_network, ffn_parameters)

    return (rbm_network, rbm_records, sgd_records)
end=#

function GenerateActivationFunctions(number_layers)
    activation_functions = Array{Function,1}()
    for in in 1:number_layers
        push!(activation_functions, SigmoidActivation)
    end
    return (activation_functions)
end

function AddDecoder(network::NeuralNetwork, network_parameters::NetworkParameters)
    function ReverseLayer(layer::NetworkLayer, initialization)
        unbiased_weights_t = WeightsWithoutBias(layer)'
        bias_weights = initialization(1, size(unbiased_weights_t)[2])
        new_weights = vcat(bias_weights, unbiased_weights_t)
        return NetworkLayer(new_weights, layer.activation)
    end

    for i in length(network.layers):-1:1
        new_layer = ReverseLayer(network.layers[i], network_parameters.initialization)
        AddLayer(network, new_layer)
    end

    #network.layers[length(network_parameters.layer_activations)].activation = network_parameters.layer_activations[end]
end

function GetAutoencoder(network::NeuralNetwork)
    layers = network.layers[1:(Int64.(length(network.layers)/2))]
    return (NeuralNetwork(layers))
    #return network
end



end
