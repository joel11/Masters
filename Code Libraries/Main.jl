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
using FinancialFunctions

################################################################################
##Experiment Process

function RunFFNExperimentConfiguration(saesgd_dataset, ogd_dataset, holdout_dataset, sae_network, ffn_network_parameters, ffn_sgd_parameters, ogd_parameters, holdout_ogd_parameters)

    ################################################################################
    ## SAE Training & Encoding
    encoded_dataset =  GenerateEncodedSGDDataset(saesgd_dataset, sae_network)

    ################################################################################
    ## FFN-SGD Training
    ffn_network = NeuralNetwork(ffn_network_parameters.layer_sizes, ffn_network_parameters.layer_activations, ffn_network_parameters.initialization)
    ffn_sgd_records = RunSGD(encoded_dataset, ffn_network, ffn_sgd_parameters)

    ################################################################################
    ## OGD Training
    encoded_ogd_dataset = GenerateEncodedSGDDataset(ogd_dataset, sae_network)
    ogd_records = RunOGD(encoded_ogd_dataset, ffn_network, ogd_parameters)

    ################################################################################
    ## Holdout
    ## Use validation data from OGD dataset
    encoded_holdout_dataset = GenerateEncodedSGDDataset(holdout_dataset, sae_network)
    holdout_records, comparisons = RunOGD(encoded_holdout_dataset, ffn_network, holdout_ogd_parameters)

    ################################################################################
    ## Trading Strategy & Profit/Losses

    actual = comparisons[1]
    predicted = comparisons[2]
    mse = sum((actual - predicted).^2)/length(actual)
    model_returns = CalculateReturns(actual, predicted)

    return (mse, model_returns)
end

################################################################################
## Setting up data

deltas = [1, 7, 30]
predictions = [7]

data_raw = GenerateDataset(1, 7300)[:,[1,4]]
data_splits = SplitData(data_raw, [0.6, 0.8])
processed_data = map(x -> ProcessData(x, deltas, predictions), data_splits)

saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], [0.8, 1.0]), processed_data)

################################################################################
##Configuration

srand(1234)
sae_netpar = NetworkParameters([6, 20, 20, 4],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.XavierGlorotNormalInit)
sae_sgd_par = TrainingParameters(0.005, 10, 0.0, 100, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

ffn_net_par = NetworkParameters([4, 40, 40, 40, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.XavierGlorotNormalInit)
ffn_sgd_par = TrainingParameters(0.00005, 30, 0.0, 100, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

ogd_par = TrainingParameters(0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
holdout_ogd_par = TrainingParameters(0.1, 1, 0.0, 1, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())

##Iterations####################################################################

sae_network, sgd_records = TrainInitSAE(saesgd_data, sae_netpar, sae_sgd_par, LinearActivation)

return_data = DataFrame()
mses = []
lrates =  [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

#Xavier
#Any[2.02137e-5, 2.09241e-5, 2.02685e-5, 2.06369e-5, 2.11963e-5, 2.13587e-5, 0.00835043]
#[-7.73268, -7.30702, -7.4037, -7.49135, -6.53332, -7.03436, 8.26569]

ffn_net_par = NetworkParameters([4, 40, 40, 40, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.XavierGlorotNormalInit)
ffn_sgd_par = TrainingParameters(0.0, 30, 0.0, 100, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())


for i in 1:length(lrates)
    lr = lrates[i]
    println(lr)

    ffn_sgd_par.learning_rate = lr

    mse, returns = RunFFNExperimentConfiguration(saesgd_data, ogd_data, holdout_data, sae_network, ffn_net_par, ffn_sgd_par, ogd_par, holdout_ogd_par)
    push!(mses, mse)
    return_data[parse(string("iteration_", i))] = returns

end

distribution = RunCSCV(return_data, 16)
pbo = CalculatePBO(distribution)

println(pbo)
println(mses)
println(map(x -> sum(return_data[:, x]), 1:size(return_data)[2]))



#Either overfitting or a result of randomness in training - weights / SGD minibatches
#If latter, same seed to start should resolve it, and should see gradual decrease as epochs increase
#Else.. former?

#inside srand(1234)
#Therefore, not inherently due to intermittent minima/maxima found from SGD

#Outside srand & outside network creation (copied inside)
#cfn = NeuralNetwork([4, 40, 40, 40, 2], [ReluActivation, ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.HeUniformInit)
#[-6.7144, -6.28165, -5.97338, -5.71669, -5.42242, -5.19049, -4.89298]
#Any[6.09703e-5, 5.79244e-5, 5.51806e-5, 5.26999e-5, 5.04672e-5, 4.84449e-5, 4.65944e-5]
#Therefore; inconsistent changes are due to different network initializations taking different lengths to improve
#Should see relatively similar performance for long running epoch on different networks, once they've had time to converge
#^Not seen for outside seed and 1500 epoch runs. Perhaps then down to SGD minimatch effect nonetheless /w learning rate?

#ffn_net_par = NetworkParameters([4, 30, 30, 30, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.HeUniformInit)
#ffn_sgd_par = TrainingParameters(0.01, 40, 0.0, 1000, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
#[-6.12703, -6.01495, -6.96097, -7.03036, -7.47923, -6.88615]

#XavierNormal
#Any[0.0178584, 2.20732e-5, 2.17889e-5, 2.09695e-5, 2.05302e-5, 2.05762e-5]
#[8.26569, -7.70187, -7.5221, -7.40636, -7.64443, -7.08936]

#Hinton: Stable results
#ffn_net_par = NetworkParameters([4, 40, 40, 40, 2] ,[ReluActivation, ReluActivation, ReluActivation, LinearActivation] ,InitializationFunctions.HintonUniformInit)
#ffn_sgd_par = TrainingParameters(0.00005, 30, 0.0, 1000, NonStopping, true, false, 0.0, 0.0, MeanSquaredError())
#Any[9.24081e-5, 6.87923e-5, 2.02936e-5, 2.02621e-5, 2.02984e-5, 2.02737e-5]
#[1.11861, -0.475769, -7.53444, -7.55722, -7.56367, -7.55946]
#epoch_tests =  [1, 100, 1000, 1001, 1002, 1003]


################################################################################
#using Plots
#plotlyjs()
##Vis##############################################################################
## Vis
#rec_output = Feedforward(ffn_network, encoded_dataset.testing_input)[end]
#allplots = []
#for i in 1:4
    #data = hcat(encoded_dataset.testing_output[:, i], rec_output[:, i])
    #pc = plot(data, ylabel = ("actual", "predicted"), ylim = (minimum(data), maximum(data)))
    #push!(allplots, pc)
#end
#savefig(plot(allplots..., layout = 4, size = (1500,800)), "/users/joeldacosta/desktop/recGraphs.html")
