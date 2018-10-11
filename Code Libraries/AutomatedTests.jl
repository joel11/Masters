module AutomatedTests

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN

export PredictionAccuracy

function PredictionAccuracy(network, dataset)
    validation_pred = Feedforward(network, dataset.validation_input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == dataset.validation_output[i,:], 1:size(dataset.validation_output)[1])))
    return(correct_predictions)
end

function FFNClassification_SigmoidLLTest(dataset)
    srand(1234)
    layer_sizes = [784, 60, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation,  SigmoidActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    cost_function = LoglikelihoodError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9287
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidSoftmaxLLTest(dataset)
    srand(2345)
    layer_sizes = [784, 60, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation,  SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    cost_function = LoglikelihoodError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9235
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidMSETest(dataset)
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
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9230
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidLLTest(dataset)

    srand(2180)

    cost_function = LoglikelihoodError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9444
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSoftmaxLLTest(dataset)

    srand(3069)
    cost_function = LoglikelihoodError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9317
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

#TO DO Add Softmax Derivative for this to work with Softmax Activation
function FFNClassification_ReluSigmoidMSETest(dataset)

    #1.8 6768
    #1.9 3930
    #2 : 7156
    #2.1: 5126
    #2.3: 2155

    srand(9876)
    cost_function = MeanSquaredError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(2, 30, 0.0,  1, 10, NonStopping, false)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 7156
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function RunTests(dataset)

    results = []

    push!(results, FFNClassification_SigmoidLLTest(dataset))
    push!(results, FFNClassification_SigmoidSoftmaxLLTest(dataset))
    push!(results, FFNClassification_SigmoidMSETest(dataset))

    push!(results, FFNClassification_ReluSigmoidLLTest(dataset))
    push!(results, FFNClassification_ReluSoftmaxLLTest(dataset))
    push!(results, FFNClassification_ReluSigmoidMSETest(dataset))

    correct = sum(Int64.(results))
    total = length(results)

    println("Correct $correct / Total $total")
end

end
