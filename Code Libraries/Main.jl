workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, StoppingFunctions, FFN, OGD
using DataGenerator, DataProcessor
################################################################################
#using Plots
#plotlyjs()

sd = FormatDataset(1, 3650, [1, 7, 30], [1, 7], 0.9, 0.95)

srand(9879)
sae_network = NetworkParameters([6, 20, 20, 2],[ReluActivation, ReluActivation, LinearActivation], InitializationFunctions.HeUniformInit)
sgd_parameters = TrainingParameters(0.005, 10, 0.0, 100, NonStopping, true, true, 0.0, 0.0, MeanSquaredError())
network, sgd_records = TrainInitSAE(sd, sae_network, sgd_parameters, LinearActivation)


#feedforward train & test data through SAE
rec_train =  GenerateEncodedSGDDataset(sd, network)

ylims = (-0.15, 0.15)
originalplot = plot(rec_train[1],ylabel = "Vals", xlabel = "Epoch", ylim=ylims)
recplot = plot(rec_train[end], ylim=ylims)
savefig(plot(originalplot, recplot, layout = 2, size = (1500,800)), string("/users/joeldacosta/desktop/recGraphs.html"))

#sgd_parameters = TrainingParameters(0.001, 10, 0.0, 500, NonStopping, true, true, 0.0, 0.0, MeanSquaredError())
#0.0070


#create train, test, validation output dataset: all stocks next day 1-delta fluctation
#SGD train prediction network: with train data + test data
#OGD train prediction network: test data

#feedforwad validation data through SAE + FFN
#get accuracy level with: test output vs validation output
#use validation output as input to trading strategy



#Use tradign strategy profitability for
