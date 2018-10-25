module AutomatedTests

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD

export PredictionAccuracy

function PredictionAccuracy(network, dataset)
    validation_pred = Feedforward(network, dataset.validation_input)[end]
    predictions = reduce(hcat, map(i -> Int64.(validation_pred[i, :] .== maximum(validation_pred[i, :])), 1:size(validation_pred)[1]))'
    correct_predictions = sum(Int64.(map(i -> predictions[i, :] == dataset.validation_output[i,:], 1:size(dataset.validation_output)[1])))
    return(correct_predictions)
end

function FFNClassification_SigmoidLLTest(dataset)
    srand(1234)
    network_parameters = NetworkParameters([784, 15, 10], [SigmoidActivation, SigmoidActivation], InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 10, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    base_ffn_parm = TrainingParameters(2.1, 10, 0.0, 5, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, base_ffn_parm)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 9077
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidSoftmaxLLTest(dataset)

    srand(2345)
    rbm_parameters = TrainingParameters(0.1, 10, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    base_ffn_parm = TrainingParameters(1.1, 10, 0.0, 5, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network_parameters = NetworkParameters([784, 15, 10], [SigmoidActivation, SoftmaxActivation], InitializationFunctions.XavierGlorotUniformInit)
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, base_ffn_parm)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 9142
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_SigmoidMSETest(dataset)

    srand(3456)
    rbm_parameters = TrainingParameters(0.1, 10, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    base_ffn_parm = TrainingParameters(3.6, 10, 0.0, 5, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    network_parameters = NetworkParameters([784, 15, 10], [SigmoidActivation, SigmoidActivation], InitializationFunctions.XavierGlorotUniformInit)
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, base_ffn_parm)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 7115
    pass = prediction_acc == expected_value
    println("FFNClassification_SigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidLLTest(dataset)

    srand(2180)
    network_parameters = NetworkParameters([784, 15, 10], [ReluActivation, SigmoidActivation], InitializationFunctions.HeNormalInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 4, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 9210
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSoftmaxLLTest(dataset)

    srand(2181)
    network_parameters = NetworkParameters([784, 15, 10], [ReluActivation, SoftmaxActivation], InitializationFunctions.HeNormalInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 3, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 9309
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSoftmaxLLTest $prediction_acc $expected_value $pass")
    return(pass)
end

function FFNClassification_ReluSigmoidMSETest(dataset)

    srand(9876)
    network_parameters = NetworkParameters([784, 15, 10], [ReluActivation, SigmoidActivation], InitializationFunctions.HeNormalInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.6, 30, 0.0, 3, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 9214
    pass = prediction_acc == expected_value
    println("FFNClassification_ReluSigmoidMSETest $prediction_acc $expected_value $pass")
    return(pass)
end

function OGDTest(dataset)
    srand(2181)
    network = NeuralNetwork([784, 100, 10], [ReluActivation, SoftmaxActivation], InitializationFunctions.HeNormalInit)
    ffn_parm = TrainingParameters(0.1, 1, 0.0, 1,  NonStopping, true, true, 0.0, 0.0, LoglikelihoodError())
    record = RunOGD(dataset, network, ffn_parm)
    prediction_acc = PredictionAccuracy(network, dataset)

    expected_value = 7400
    pass = prediction_acc == expected_value
    println("OGDTest $prediction_acc $expected_value $pass")
    return(pass)
end


function RBMTest(dataset)
    ##RBM Testing ################################################################################
    srand(1180)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, CrossEntropyError())
    network_parameters = NetworkParameters([784, 500], [SigmoidActivation], InitializationFunctions.XavierGlorotUniformInit)
    layer = CreateRBMLayer(1, network_parameters)
    epoch_records = TrainRBMLayer(dataset.training_input, dataset.validation_input, layer, rbm_parameters)

    expected = 55.62030683485252
    val_error = epoch_records[end].test_cost
    pass = expected == val_error

    println("RMBTest $expected $val_error $pass")
    return(pass)

    #using OutputLibrary
    #PlotRBMInputOutput([epoch_records], dataset.validation_input, 25,"/Users/joeldacosta/Desktop/")
end

function RunTests(dataset)

    results = []

    push!(results, FFNClassification_SigmoidLLTest(dataset))
    push!(results, FFNClassification_SigmoidSoftmaxLLTest(dataset))
    push!(results, FFNClassification_SigmoidMSETest(dataset))

    push!(results, FFNClassification_ReluSigmoidLLTest(dataset))
    push!(results, FFNClassification_ReluSoftmaxLLTest(dataset))
    push!(results, FFNClassification_ReluSigmoidMSETest(dataset))

    push!(results, OGDTest(dataset))
    push!(results, RBMTest(dataset))

    correct = sum(Int64.(results))
    total = length(results)

    println("Correct $correct / Total $total")
end




function OverfittingExample(dataset, output_dir)

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

function PretrainingTests(dataset)
    ##Pretraining Testing########################################################################

    srand(1080)
    #network_parameters = NetworkParameters([784, 500, 500, 2000, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
        , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #9690

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [ReluActivation, ReluActivation, ReluActivation, SoftmaxActivation]
        , InitializationFunctions.HeUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #9797

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
        , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #9529

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
        , InitializationFunctions.HintonUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #5241

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SoftmaxActivation]
        , InitializationFunctions.HintonUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #9708

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation]
        , InitializationFunctions.HintonUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #9731

    ############################################################
    srand(1080)
    network_parameters = NetworkParameters([784, 250, 250, 500, 10], [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation]
        , InitializationFunctions.HintonUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, false, true, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, false, true, 0.0, 0.0, LoglikelihoodError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    prediction_acc = PredictionAccuracy(network, dataset)
    #5954
end

function LongPretrainTest(dataset)
    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 100, 200, 400, 784]
                                            , [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation
                                            ,  SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation]
                                            , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)


    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 100, 200, 400, 784]
                                            , [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation
                                            ,  SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation]
                                            , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
end

function RBMSAETests(dataset)
    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 6]
                                            , [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation]
                                            , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainRBMSAE(dataset, network_parameters, rbm_parameters, ffn_parameters)

    using OutputLibrary
    PlotInputOutput(network, dataset.validation_input, 20, "/Users/joeldacosta/Desktop/")

    ##With Linear Function
    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 6]
                                            , [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, LinearActivation]
                                            , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.1, 30, 0.0, 10, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainRBMSAE(dataset, network_parameters, rbm_parameters, ffn_parameters)

    using OutputLibrary
    PlotInputOutput(network, dataset.validation_input, 20, "/Users/joeldacosta/Desktop/")


    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 8]
                                            , [SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, SigmoidActivation, LinearActivation]
                                            , InitializationFunctions.XavierGlorotUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.01, 30, 0.0, 20, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records =
    TrainRBMSAE(dataset, network_parameters,rbm_parameters, ffn_parameters)


end

function InitSAETests_WithLinear(dataset)

    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 8, 25, 50, 100, 200, 400, 784]
                                            , [ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation
                                            ,  ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation, SigmoidActivation]
                                            , InitializationFunctions.HeUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.01, 30, 0.0, 30, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainEncoderRBNMFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    #MSE ±13

    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 8, 25, 50, 100, 200, 400, 784]
                                            , [ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation, LinearActivation
                                            ,  ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation, SigmoidActivation]
                                            , InitializationFunctions.HeUniformInit)
    rbm_parameters = TrainingParameters(0.1, 30, 0.0, 0, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    ffn_parameters = TrainingParameters(0.01, 30, 0.0, 30, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records, ffn_records = TrainEncoderRBNMFFNNetwork(dataset, network_parameters, rbm_parameters, ffn_parameters)
    #MSE 13.04;13.75

    srand(1080)
    network_parameters = NetworkParameters( [784, 400, 200, 100, 50, 25, 8]
                                            , [ReluActivation, ReluActivation, ReluActivation, ReluActivation, ReluActivation, LinearActivation]
                                            , InitializationFunctions.HeUniformInit)
    ffn_parameters = TrainingParameters(0.01, 30, 0.0, 30, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    network, rbm_records =
    TrainInitSAE(dataset, network_parameters, ffn_parameters, SigmoidActivation)
    #MSE 12.11; 12.96
end

end
