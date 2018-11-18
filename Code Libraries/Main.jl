workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
################################################################################
#using Plots
#plotlyjs()

################################################################################
## Setting up data

srand(9879)
deltas = [1, 7, 30]
predictions = [7]

data_raw = GenerateDataset(1, 7300)[:,[1,4]]
data_splits = SplitData(data_raw, [0.6, 0.8])
processed_data = map(x -> ProcessData(x, deltas, predictions), data_splits)

saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], [0.8, 1.0]), processed_data)

################################################################################
## SAE Training

sae_parameters = NetworkParameters([6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.HeUniformInit)
sgd_parameters = TrainingParameters(0.005, 10, 0.0, 1000, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
sae_network, sgd_records = TrainInitSAE(saesgd_data, sae_parameters, sgd_parameters, LinearActivation)

################################################################################
## FFN-SGD Training

encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, sae_network)

ffn_network = NeuralNetwork([4, 40, 40, 40, 2]
                        , [ReluActivation, ReluActivation, ReluActivation, LinearActivation]
                        , InitializationFunctions.HeUniformInit)
sgd_parameters2 = TrainingParameters(0.0005, 20, 0.0, 1000, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
sgd_records2 = RunSGD(encoded_dataset, ffn_network, sgd_parameters2)

#rec_output = Feedforward(ffn_network, encoded_dataset.testing_input)[end]
#allplots = []
#for i in 1:4
    #data = hcat(encoded_dataset.testing_output[:, i], rec_output[:, i])
    #pc = plot(data, ylabel = ("actual", "predicted"), ylim = (minimum(data), maximum(data)))
    #push!(allplots, pc)
#end
#savefig(plot(allplots..., layout = 4, size = (1500,800)), "/users/joeldacosta/desktop/recGraphs.html")

################################################################################
## OGD Training

encoded_ogd_dataset = GenerateEncodedSGDDataset(ogd_data, sae_network)
ogd_parameters = TrainingParameters(0.001, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
ogd_records = RunOGD(encoded_ogd_dataset, ffn_network, ogd_parameters)

################################################################################
## Holdout
## Use validation data from OGD dataset

encoded_holdout_dataset = GenerateEncodedSGDDataset(holdout_data, sae_network)
holdout_parameters = TrainingParameters(0.001, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
holdout_records, comparisons = RunOGD(encoded_holdout_dataset, ffn_network, holdout_parameters)

################################################################################
## Trading Strategy & Profit/Losses

function CalculateProfit(predicted, actual)
    rev = (abs(actual) > abs(predicted)) ? abs(predicted) : sign(predicted) * (actual - predicted)
    return rev
end

actual = comparisons[1]
predicted = comparisons[2]
returns = map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])



################################################################################
## CSCV


################################################################################
## PBO
