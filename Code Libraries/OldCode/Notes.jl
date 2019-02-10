#Initialization Notes

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
