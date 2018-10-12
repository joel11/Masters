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
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false, true, 0.0, 0.0)
    cost_function = LoglikelihoodError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9233
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidSoftmaxLLTest(dataset)
    srand(2345)
    layer_sizes = [784, 60, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation,  SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false, true, 0.0, 0.0)
    cost_function = LoglikelihoodError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9212
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidMSETest(dataset)

    srand(1080)
    cost_function = MeanSquaredError()
    layer_sizes = [784, 100,  10]
    layer_functions = [SigmoidActivation, SigmoidActivation]
    parameters = TrainingParameters(2.15, 30, 0.0,  1, 10, NonStopping, false, true, 0.0, 0.0)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 5757
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidLLTest(dataset)

    srand(2180)

    cost_function = LoglikelihoodError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false, true, 0.0, 0.0)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9366
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSoftmaxLLTest(dataset)

    srand(3069)
    cost_function = LoglikelihoodError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SoftmaxActivation]
    parameters = TrainingParameters(0.1, 30, 0.0,  2, 5, NonStopping, false, true, 0.0, 0.0)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)
    expected_value = 9275
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidMSETest(dataset)

    srand(9876)
    cost_function = MeanSquaredError()
    layer_sizes = [784, 100,  10]
    layer_functions = [ReluActivation, SigmoidActivation]
    parameters = TrainingParameters(1.6, 30, 0.0,  1, 10, NonStopping, false, true, 0.0, 0.0)
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records =
    TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)

    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 7439
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

function OverfittingExampls(dataset, output_dir)

    dataset.training_input = dataset.training_input[1:1000, :]
    dataset.training_output = dataset.training_output[1:1000, :]

    srand(1234)
    layer_sizes = [784, 30, 10]
    layer_functions = [SigmoidActivation, SigmoidActivation]
    parameters = TrainingParameters(0.5, 10, 0.0,  0, 300, NonStopping, true, true, 0.0, 0.0)
    cost_function = LoglikelihoodError()
    initialization = InitializationFunctions.XavierGlorotUniformInit

    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, layer_sizes, layer_functions, initialization, parameters, cost_function)
    prediction_acc = PredictionAccuracy(network, dataset)

    using OutputLibrary
    reload("OutputLibrary")
    WriteFFNGraphs(ffn_records, output_dir)


end

end
