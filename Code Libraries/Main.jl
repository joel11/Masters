workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
################################################################################
#using Plots
#plotlyjs()

################################################################################
## Setting up data

deltas = [1, 7, 30]
predictions = [7]

data_raw = GenerateDataset(1, 7300)[:,[1,4]]
data_splits = SplitData(data_raw, [0.6, 0.8])
processed_data = map(x -> ProcessData(x, deltas, predictions), data_splits)

saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], [0.8, 1.0]), processed_data)

cscv_data = DataFrame()
mses = []
epoch_tests = [1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510]
srand(9879)
################################################################################
## SAE Training & Encoding

sae_parameters = NetworkParameters([6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.HeUniformInit)
sgd_parameters = TrainingParameters(0.005, 10, 0.0, 1000, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
sae_network, sgd_records = TrainInitSAE(saesgd_data, sae_parameters, sgd_parameters, LinearActivation)
encoded_dataset =  GenerateEncodedSGDDataset(saesgd_data, sae_network)

#Either overfitting or a result of randomness in training - weights / SGD minibatches
#If latter, same seed to start should resolve it, and should see gradual decrease as epochs increase
#Else.. former?

#[-6.91258, -6.34586, -5.97316, -5.65011, -5.29533]Any[6.26168e-5, 5.86508e-5, 5.51745e-5, 5.21172e-5, 4.9422e-5]

#outside srand(1234)
#[-6.7144, -1.43021, -0.449516, -0.0132619, -0.74805, -2.06555, 0.016782]
#Any[6.09703e-5, 2.76489e-5, 2.56272e-5, 2.26612e-5, 2.53971e-5, 2.88241e-5, 2.47536e-5]

#inside srand(1234)
#[-6.7144, -6.28141, -5.97316, -5.71669, -5.41888, -5.19055, -4.89301]Any[6.09703e-5, 5.79195e-5, 5.51745e-5, 5.26995e-5, 5.04643e-5, 4.84457e-5, 4.65943e-5]
#Therefore, not inherently due to intermittent minima/maxima found from SGD

#Outside srand & outside network creation (copied inside)
#cfn = NeuralNetwork([4, 40, 40, 40, 2], [ReluActivation, ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.HeUniformInit)
#[-6.7144, -6.28165, -5.97338, -5.71669, -5.42242, -5.19049, -4.89298]
#Any[6.09703e-5, 5.79244e-5, 5.51806e-5, 5.26999e-5, 5.04672e-5, 4.84449e-5, 4.65944e-5]
#Therefore; inconsistent changes are due to different network initializations taking different lengths to improve
#Should see relatively similar performance for long running epoch on different networks, once they've had time to converge
#^Not seen for outside seed and 1500 epoch runs. Perhaps then down to SGD minimatch effect nonetheless /w learning rate?


#Overnight run: as found, with lr/10 and mb size from 20->30

srand(1234)


for num_epochs in epoch_tests


    ################################################################################
    ## FFN-SGD Training
    #ffn_network = CopyNetwork(cfn)
    ffn_network = NeuralNetwork([4, 40, 40, 40, 2]
                            , [ReluActivation, ReluActivation, ReluActivation, LinearActivation]
                            , InitializationFunctions.HeUniformInit)
    #sgd_parameters2 = TrainingParameters(0.0005, 20, 0.0, num_epochs, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
    sgd_parameters2 = TrainingParameters(0.00005, 30, 0.0, num_epochs, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())


    sgd_records2 = RunSGD(encoded_dataset, ffn_network, sgd_parameters2)

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

    sqrs = (actual - predicted).^2
    m = sum(sqrs)/length(sqrs)
    push!(mses, m)

    returns = map(r -> mapreduce(c -> CalculateProfit(predicted[r, c], actual[r, c]), +, 1:size(actual)[2]), 1:size(actual)[1])

    cscv_data[parse(string("iteration_", num_epochs))] = returns
end

print(map(x -> sum(cscv_data[:, x]), 1:size(cscv_data)[2]))
print(mses)

################################################################################
## CSCV & PBO

distribution = RunCSCV(cscv_data, 16)
pbo = CalculatePBO(distribution)


################################################################################
## Vis
#rec_output = Feedforward(ffn_network, encoded_dataset.testing_input)[end]
#allplots = []
#for i in 1:4
    #data = hcat(encoded_dataset.testing_output[:, i], rec_output[:, i])
    #pc = plot(data, ylabel = ("actual", "predicted"), ylim = (minimum(data), maximum(data)))
    #push!(allplots, pc)
#end
#savefig(plot(allplots..., layout = 4, size = (1500,800)), "/users/joeldacosta/desktop/recGraphs.html")
