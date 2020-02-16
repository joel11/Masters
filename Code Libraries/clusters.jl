
workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using DataJSETop40
using BSON
using MLBase
using PlotlyJS

##########Cluster Analysis###############################################################################################################

function PrintClusterAnalysis()

    clusterOneResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p
                                inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 0
                                where sharpe_ratio is not null")
    clusterTwoResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p
                                inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 1
                                where sharpe_ratio is not null")

    println(string("Cluster One Skewness: ", string(skewness(Array(clusterOneResults[:sharpe_ratio])))))
    println(string("Cluster Two Skewness: ", string(skewness(Array(clusterTwoResults[:sharpe_ratio])))))

    println(string("Cluster One Kurtosis: ", string(kurtosis(Array(clusterOneResults[:sharpe_ratio])))))
    println(string("Cluster Two Kurtosis: ", string(kurtosis(Array(clusterTwoResults[:sharpe_ratio])))))

    println(string("Cluster One Mean: ", string(mean(Array(clusterOneResults[:sharpe_ratio])))))
    println(string("Cluster Two Mean: ", string(mean(Array(clusterTwoResults[:sharpe_ratio])))))

    println(string("Cluster One Variance: ", string(var(Array(clusterOneResults[:sharpe_ratio])))))
    println(string("Cluster Two Variance: ", string(var(Array(clusterTwoResults[:sharpe_ratio])))))
end

##OGD vs PL graph###############################################################################################################

function OGDvsMSEPlot()

    ogdpl_data = RunQuery("select training_cost --, total_pl
    from clusters c
    inner join epoch_records er on er.configuration_id = c.configuration_id and category = 'OGD'
    --inner join config_oos_pl pl on pl.configuration_id = c.configuration_id and total_pl is not null and training_cost is not null
    --where training_cost < 0.02")

    #cor(Array(ogdpl_data[:training_cost]),Array(ogdpl_data[:total_pl]))

    ogdpl_data[:,1] = Array(ogdpl_data[:,1])
    ogdpl_data[:,2] = Array(ogdpl_data[:,2])
    ogdpl_data[:rounded_mse] = round.(Array(ogdpl_data[:,1]),3)

    groups = by(ogdpl_data, [:rounded_mse], df -> mean(df[:total_pl]))

    trace1 = scatter(;x = Array(ogdpl_data[:training_cost]), y = Array(ogdpl_data[:total_pl]), mode = "markers", name = "Observations")

    trace2 = scatter(;x = Array(groups[:rounded_mse]), y = Array(groups[:x1]), mode = "line", name = "Mean")

    fig = Plot([trace1, trace2])
    savefig(plot(fig), string("/users/joeldacosta/desktop/msepl.html"))
end
