push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using ActivationFunctions
using InitializationFunctions
using TrainingStructures
using NeuralNetworks
using RBM
using AutoEncoder

using MNIST




#Movie Tests####################################################################
#=
trainingData = [[1 1 1 0 0 0];[1 0 1 0 0 0];[1 1 1 0 0 0];[0 0 1 1 1 0];[0 0 1 1 0 0];[0 0 1 1 1 0]]
learning_rate = 0.5
batch_size = 2
net = RBM.ShallowRBM(6, 2, NeuralNetwork.sigmoidActivation, RBM.XavierGlorotUniformInit)
errors = RBM.trainRBMBatch(trainingData, net, learning_rate, batch_size, 5000)
data = [0 0 0 1 1 0]
print(RBM.run_visible(net, [0 0 0 0 1 1]))
plot(errors)
print(RBM.run_hidden(net, [1 0]))
=#

#MNIST#########################################################################
#=
using Plots
plotlyjs()
gr()

function plot_image(image_vector)
    dim = Int64.(sqrt(size(image_vector)[1]))
    image = reshape(image_vector, dim, dim)
    Plots.heatmap(image)
end
=#

trainingdata, traininglabels = traindata()
validationdata, validationlabels = testdata()
scaled_training_data = (trainingdata')./255
scaled_validation_data = (validationdata')./255

parameters = TrainingParameters(0.1, 20, 0.0,  1)

#Test TrainRBMLayer
#layer = CreateRBMLayer(784, 100, NeuralNetworks.ActivationFunctions.SigmoidActivation, InitializationFunctions.XavierGlorotUniformInit)
#epoch_records = TrainRBMLayer(scaled_training_data, scaled_validation_data, layer, parameters)

#Test TrainRBMNetwork

#=
network, epoch_records_network =
    TrainRBMNetwork(scaled_training_data,
                    scaled_validation_data,
                    [784, 30, 20, 5],
                    [SigmoidActivation SigmoidActivation SigmoidActivation],
                    InitializationFunctions.XavierGlorotUniformInit,
                    parameters)


=#
#Test CreateAutoEncoder

encoder = CreateAutoEncoder(scaled_training_data, scaled_validation_data, [784, 30, 20, 5],
InitializationFunctions.XavierGlorotUniformInit, parameters)

#Create functions to map the epoch_records











#i = 89
#sample = scaled_training_data[i, :]
#print(traininglabels[i])
#plot_image(sample)

#plot_image(net.weights[2:end,17])

#rec = RBM.reconstruct(net, scaleddata[:,i]')
#plot_image(rec')






#hid = RBM.run_visible(net, scaleddata[:,i]', false)
#vis = RBM.run_hidden(net, hid, false)
#plot_image(vis')

#Plotting Weights
#histogram(weight_hist[2:end,2:end,2])

#plot(energy_ratios)
