module AutomatedTests

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN


export PredictionAccuracy

function PredictionAccuracy(network, validation_input, validation_labels)
    validation_pred = Feedforward(network, validation_input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == validation_labels[i,:], 1:size(validation_labels)[1])))
    return(correct_predictions)
end

using MNIST
trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()

training_labels = fill(0.0, (10, length(traininglabels)))
validation_labels = fill(0.0, (10, length(validationlabels)))

i = 1

for i in 1:length(traininglabels)
    training_labels[Int64.(traininglabels[i])+1, i] = 1
end

for i in 1:length(validationlabels)
    validation_labels[Int64.(validationlabels[i])+1, i] = 1
end



scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255


training_data = scaled_training_data
validation_data = scaled_validation_data



function FFNClassification_SigmoidCCETest()
    srand(1234)
    layer_sizes = [784, 60, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation,  SigmoidActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    cost_function = CategoricalCrossEntropyError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9287
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidCCETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidSoftmaxCCETest()
    srand(2345)
    layer_sizes = [784, 60, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation,  SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    cost_function = CategoricalCrossEntropyError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9235
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidSoftmaxCCETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidMSETest()
    #2.5 + 1 - 9230
    #1.5 + 2
    #2.5 + 1; 30 - 9396
    srand(1080)
    cost_function = MeanSquaredError()
    layer_sizes = [784, 100,  10]
    layer_functions = [SigmoidActivation, SigmoidActivation]
    parameters = TrainingParameters(2.5, 30, 0.0,  1, 10, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9230
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidCETest()

    srand(2180)

    cost_function = CategoricalCrossEntropyError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9412
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidCETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSoftmaxCETest()

    srand(3069)
    cost_function = CategoricalCrossEntropyError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9317
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSoftmaxCETest $prediction_acc $expected_value $pass")
    return(pass)
end

#TO DO Add Softmax Derivative for this to work with Softmax Activation
function FFNClassification_ReluSigmoidMSETest()

    #1.8 6768
    #1.9 3930
    #2 : 7156
    #2.1: 5126
    #2.3: 2155

    srand(9876)
    cost_function = MeanSquaredError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(0.5, 30, 0.0,  1, 10, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(scaled_training_data, training_labels', scaled_validation_data, validation_labels', layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, validation_data, validation_labels')
    expected_value = 9230
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function RunTests()
    FFNClassification_SigmoidCCETest()
    FFNClassification_SigmoidSoftmaxCCETest()
    FFNClassification_SigmoidMSETest()

    FFNClassification_ReluSigmoidCETest()
    FFNClassification_ReluSoftmaxCETest()
end


end
