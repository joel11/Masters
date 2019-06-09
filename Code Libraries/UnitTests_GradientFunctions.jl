workspace()

using DataFrames
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess
using GradientFunctions

parameters = TrainingParameters("FFN",
                                0.5, #max_learning_rate
                                0.5, #min_learning_rate
                                1,  #epoch_cycle_max
                                1, #minibatch_size
                                1, #max_epochs
                                (0.0, 0), #stopping_parameters
                                NonStopping, #stopping_function
                                0.0, #l1_lambda
                                MeanSquaredError(), #cost_function
                                [1.0], #training_splits
                                false, #is_denoising
                                0.0) #denoising_variance

weights1 = [0.35 0.35; 0.15 0.25; 0.2 0.3]
weights2 = [0.6 0.6; 0.4 0.5; 0.45 0.55]

function SingleSGDTest()

    layer_one = NetworkLayer(weights1, SigmoidActivation)
    layer_two = NetworkLayer(weights2, SigmoidActivation)
    network = NeuralNetwork([layer_one, layer_two])

    minibatch_input = [0.05 0.1]#; 0.01 0.4]
    minibatch_ouput = [0.01 0.99]#; 0.1 0.9]

    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    GradientDescentWeightUpdate(network,
                                minibatch_input,
                                minibatch_ouput,
                                parameters,
                                weight_updates,
                                zero_activations,
                                activations)

    expected_activations = [[0.05 0.1],[0.593269992 0.596884378],[0.75136507 0.772928465]]


    sgd1 = all(round.(expected_activations,4) .== round.(activations,4))
    sgd2 = round(weight_updates[1][2,1],5) .== round(0.000438568,5)
    sgd3 = round(weight_updates[2][2,1],5) .== round(0.082167041,5)

    for l in 1:length(network.layers)
        network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, 1, 1)
    end

    sgd4 = all(round.(network.layers[1].weights[2:end,:],5) .== round.([[0.149780716 0.24975114];[0.19956143 0.29950229]],5))
    sgd5 = all(round.(network.layers[2].weights[2:end,:],5) .== round.([[0.35891648 0.511301270];[0.408666186 0.561370121]],5))

    println("Single SGD 1: $sgd1")
    println("Single SGD 2: $sgd2")
    println("Single SGD 3: $sgd3")
    println("Single SGD 4: $sgd4")
    println("Single SGD 5: $sgd5")
end

###########

function MultiSGDTest()

    layer_one = NetworkLayer(weights1, SigmoidActivation)
    layer_two = NetworkLayer(weights2, SigmoidActivation)
    network = NeuralNetwork([layer_one, layer_two])

    minibatch_input1 = [0.05 0.1]
    minibatch_ouput1 = [0.01 0.99]
    minibatch_input2 = [0.01 0.4]
    minibatch_ouput2 = [0.1 0.9]
    minibatch_input3 = [0.05 0.1; 0.01 0.4]
    minibatch_ouput3 = [0.01 0.99; 0.1 0.9]


    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    GradientDescentWeightUpdate(network, minibatch_input1, minibatch_ouput1, parameters, weight_updates, zero_activations, activations)
    w1 = deepcopy(weight_updates)
    GradientDescentWeightUpdate(network, minibatch_input2, minibatch_ouput2, parameters, weight_updates, zero_activations, activations)
    w2 = deepcopy(weight_updates)
    GradientDescentWeightUpdate(network, minibatch_input3, minibatch_ouput3, parameters, weight_updates, zero_activations, activations)
    wboth = deepcopy(weight_updates)

    msgd1 = all((w1 .+ w2)./2 .== wboth)

    println("Multi SGD 1: $msgd1")
end

###########

function OGDTest()

    layer_one = NetworkLayer(weights1, SigmoidActivation)
    layer_two = NetworkLayer(weights2, SigmoidActivation)
    network = NeuralNetwork([layer_one, layer_two])

    minibatch_input1 = [0.05 0.1]
    minibatch_ouput1 = [0.01 0.99]
    minibatch_input2 = [0.01 0.4]
    minibatch_ouput2 = [0.1 0.9]

    weight_updates = Array{Array{Float64,2}}(length(network.layers))
    zero_activations = (fill(0, (length(network.layers),1)))
    activations = Array{Array{Float64,2},1}(length(network.layers)+1)

    GradientDescentWeightUpdate(network, minibatch_input1, minibatch_ouput1, parameters, weight_updates, zero_activations, activations)
    for l in 1:length(network.layers)
        network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, 1, 1)
    end

    GradientDescentWeightUpdate(network, minibatch_input2, minibatch_ouput2, parameters, weight_updates, zero_activations, activations)
    for l in 1:length(network.layers)
        network.layers[l].weights = CalculateNewWeights(network.layers[l].weights, weight_updates[l], parameters, 1, 1)
    end


    minibatch_input3 = [0.05 0.1; 0.01 0.4]
    minibatch_ouput3 = [0.01 0.99; 0.1 0.9]

    ogd_ds  = DataSet(nothing, minibatch_input3, DataFrame(), minibatch_ouput3, DataFrame(), nothing, nothing, nothing, nothing)

    ogd_par = OGDTrainingParameters("FFN-OGD", 0.5, true, MeanSquaredError(), 0)
    layer_one = NetworkLayer(weights1, SigmoidActivation)
    layer_two = NetworkLayer(weights2, SigmoidActivation)
    new_network = NeuralNetwork([layer_one, layer_two])

    ogd_return = RunOGD(-1, "test", ogd_ds, new_network, ogd_par)

    ogd_network = ogd_return[1][1].network


    layer_one = all(ogd_network.layers[1].weights .== network.layers[1].weights)
    layer_two = all(ogd_network.layers[2].weights .== network.layers[2].weights)

    all_layers = layer_one && layer_two

    println("OGD Test 1: $all_layers")
end


SingleSGDTest()
MultiSGDTest()
OGDTest()
