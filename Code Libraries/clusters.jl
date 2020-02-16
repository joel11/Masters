
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

##########Sharpe Ratio Plot##############################################################################################################

function ClusterDistributionPlot()

    clusterOneResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 0 where sharpe_ratio is not null")
    clusterTwoResults = RunQuery("select sharpe_ratio from config_oos_sharpe_ratio_cost p inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 1 where sharpe_ratio is not null")
    highestSR =get(RunQuery("select max(sharpe_ratio) from config_oos_sharpe_ratio_cost")[1,1])

    clusterOneResults[:grouping] = round.(Array(clusterOneResults[:sharpe_ratio]), 2)
    clusterTwoResults[:grouping] = round.(Array(clusterTwoResults[:sharpe_ratio]), 2)

    one_groups = by(clusterOneResults, [:grouping], df -> size(df, 1))
    two_groups = by(clusterTwoResults, [:grouping], df -> size(df, 1))

    one_groups[:x1] = one_groups[:x1]./size(clusterOneResults, 1).*100
    two_groups[:x1] = two_groups[:x1]./size(clusterTwoResults, 1).*100

    opacity_value = 0.8

    br_max = max(maximum(one_groups[:x1]),
                maximum(two_groups[:x1])) + 0.1

    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Cluster One [n=", size(clusterOneResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
    trace_two = bar(;y=two_groups[:x1], x=two_groups[:grouping], name=string("Cluster Two [n=", size(clusterTwoResults,1),"]"), opacity=opacity_value, xbins=Dict(:size=>0.001))
    bm  = scatter(;x=[highestSR, highestSR],y=[0, br_max], name="Highest SR", marker = Dict(:color=>"green"))

    data = [trace_one, trace_two, bm]

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Trials in Cluster </br> </b>"))
        , xaxis = Dict(:title => string("<b> Sharpe Ratio </b>"))
        , barmode="overlay")

    fig = Plot(data, l)
    savefig(plot(fig), string("/users/joeldacosta/desktop/Cluster Distributions.html"))
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


##########OGD MSE Plot##############################################################################################################

function ClusterOGDMSEPlot()

    clusterOneResults = RunQuery("select training_cost
                                    from epoch_records p
                                    inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 0
                                    inner join config_oos_sharpe_ratio_cost s on s.configuration_id = c.configuration_id and sharpe_ratio is not null
                                    where category = 'OGD'
                                    --and training_cost is not null
                                    and training_cost < 0.05")
    clusterTwoResults = RunQuery("select training_cost
                                from epoch_records p
                                inner join clusters c on p.configuration_id = c.configuration_id and c.cluster = 1
                                inner join config_oos_sharpe_ratio_cost s on s.configuration_id = c.configuration_id and sharpe_ratio is not null
                                where category = 'OGD'
                                --and training_cost is not null
                                and training_cost < 0.05")

    clusterOneResults[:grouping] = round.(Array(clusterOneResults[:training_cost]), 3)
    clusterTwoResults[:grouping] = round.(Array(clusterTwoResults[:training_cost]), 3)

    one_groups = by(clusterOneResults, [:grouping], df -> size(df, 1))
    two_groups = by(clusterTwoResults, [:grouping], df -> size(df, 1))

    one_groups[:x1] = one_groups[:x1]./size(clusterOneResults, 1).*100
    two_groups[:x1] = two_groups[:x1]./size(clusterTwoResults, 1).*100

    opacity_value = 0.8

    trace_one = bar(;y=one_groups[:x1], x=one_groups[:grouping], name=string("Cluster One [n=7415]"), opacity=opacity_value, xbins=Dict(:size=>0.0001))
    trace_two = bar(;y=two_groups[:x1], x=two_groups[:grouping], name=string("Cluster Two [n=14400]"), opacity=opacity_value, xbins=Dict(:size=>0.0001))

    data = [trace_one, trace_two]#, bm]

    l = Layout(width = 900, height = 600, margin = Dict(:b => 100, :l => 100)
        , yaxis = Dict(:title => string("<b> Percentage of Trials </br> </b>"))
        , xaxis = Dict(:title => string("<b> OGD MSE </b>"))
        , barmode="overlay")

    fig = Plot(data, l)
    savefig(plot(fig), string("/users/joeldacosta/desktop/Cluster MSE Distributions.html"))
end
