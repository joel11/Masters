#module ExperimetGraphs2

workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using RBM
using NeuralNetworks
using ActivationFunctions, InitializationFunctions, NetworkTrainer
using TrainingStructures
using SGD, CostFunctions, FunctionsStopping, FFN, OGD
using DataGenerator, DataProcessor
using DataFrames
using CSCV
using FinancialFunctions
using DatabaseOps
using ConfigGenerator
using ExperimentProcess
using DataJSETop40
using BSON


function UpdateTotalProfits(config_ids)

    #Original Setup
    #TotalProfits = DataFrame()
    #TotalProfits[:configuration_id] = config_ids
    #TotalProfits[:profit] = NaN

    TotalProfits = BSON.load("ProfitVals.bson")[:profits]

    current_configs = TotalProfits[:,1]
    needed_configs = collect(setdiff(Set(config_ids), Set(current_configs)))

    configs = mapreduce(x->string(x, ","), string, needed_configs)[1:(end-1)]
    query = string("select * from prediction_results where configuration_id in ($configs)")
    results = RunQuery(query)

    for i in 1:(length(needed_configs))

        c = needed_configs[i]
        println(c)
        query = string("select * from prediction_results where configuration_id in ($c)")
        results = RunQuery(query)
        profits = sum(CalculateReturnsOneD(Array(results[:,:actual]), Array(results[:,:predicted])))
        #profits = sum(CalculateReturnsOneD(Array(results[:,:actual]), Array(results[:,:actual])))

        TotalProfits = cat(1, TotalProfits, [c profits])
    end

    profit_array = Array(TotalProfits)
    file_name = string("ProfitVals.bson")
    values = Dict(:profits => profit_array)
    bson(file_name, values)
end


function ReadProfits()
    pa = BSON.load("ProfitVals.bson")

    TotalProfits = DataFrame()
    TotalProfits[:configuration_id] = pa[:profits][:,1]
    TotalProfits[:profit] = pa[:profits][:,2]
    return TotalProfits
end

config_ids = 504:622
UpdateTotalProfits(504:622)
maximum(TotalProfits[:profit])

##Prediction Plots#############################################################

#using Plots
#plotlyjs()
using PlotlyJS

#minmse = 510
#maxprof = 902

best_config = 902 #TotalProfits[Array{Bool}(TotalProfits[:profit] .== maximum(TotalProfits[:profit])),:][1][1]
best_query = string("select * from prediction_results where configuration_id in (510, 902)")
best_results = RunQuery(best_query)
best_groups = by(best_results, [:stock], df -> [df])
#recreation_plots = Vector{GenericTrace{Dict{Symbol,Any}}}()

names = Dict(902 => "Profit", 510 => "MSE")

for i in 1:size(best_groups,1)
    timesteps = best_groups[i,2][:time_step]
    config_groups = by(best_groups[i,2], [:configuration_id], df-> [df])

    actual = cumsum(config_groups[1,2][:actual])
    predicted_one = cumsum(config_groups[1,2][:predicted])
    predicted_two = cumsum(config_groups[2,2][:predicted])

    stock_name = get(best_groups[i,1])

    t0 = scatter(;y=actual, x = timesteps, name=string(stock_name, "_actual"), mode ="lines", xaxis = string("x", i), yaxis = string("y", i))
    t1 = scatter(;y=predicted_one, x = timesteps, name=string(stock_name, "_predicted_", names[get(config_groups[1][1])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))
    t2 = scatter(;y=predicted_two, x = timesteps, name=string(stock_name, "_predicted_", names[get(config_groups[1][2])]), mode="lines", xaxis = string("x", i), yaxis = string("y", i))

    recreation_plots = [t0, t1, t2]
    filename = string("recreation_", stock_name)
    savefig(plot(recreation_plots), string("/users/joeldacosta/desktop/", filename, ".html"))

    #push!(recreation_plots, t0)
    #push!(recreation_plots, t1)
end


sae_choices = (253, 265, 256, 260, 264)
min_config = minimum(TotalProfits[:,1])



################################################################################
##Layers vs. SGD Learning Rate Best MSE


layer_msequery = string("select er.configuration_id, learning_rate, min(testing_cost) min_test_cost, experiment_set_name
from epoch_records er
inner join configuration_run cr on cr.configuration_id = er.configuration_id
inner join training_parameters tp on tp.configuration_id = er.configuration_id
where er.configuration_id >= $min_config
and er.category = \"FFN-SGD\"
and tp. category = \"FFN\"
group by er.configuration_id, sae_config_id, learning_rate")
layer_mseresults = RunQuery(layer_msequery)



layer_mseresults[:,1] = Array{Int64,1}(layer_mseresults[:,1])
layer_mseresults[:,2] = Array{Float64,1}(layer_mseresults[:,2])
layer_mseresults[:,3] = Array{Float64,1}(layer_mseresults[:,3])
layer_mseresults[:,4] = Array{String,1}(layer_mseresults[:,4])
layer_mseresults[:layers] = map(getlayerstruc, Array(layer_mseresults[:experiment_set_name]))
layer_mseresults = join(TotalProfits, layer_mseresults, on = :configuration_id)


comb_mse = by(layer_mseresults, [:layers, :learning_rate], df -> maximum(df[:profit]))
#comb_mse = by(layer_mseresults, [:layers, :learning_rate], df -> mean(df[:min_test_cost]))

layers_order = Array(unique(comb_mse[:layers]))

layers_order = Array(sort(map(l -> string(split(l, "x")[2], "x", split(l, "x")[1]), layers_order)))
rates_order = Array(unique(comb_mse[:learning_rate]))

vals = Array{Float64,2}(fill(NaN, length(layers), length(rates)))
for r in 1:size(comb_mse, 1)
    lval = string(split(comb_mse[r,1], "x")[2], "x", split(comb_mse[r,1], "x")[1])
    l_index = findfirst(layers_order, lval)
    r_index = findfirst(rates_order, comb_mse[r,2])
    vals[l_index, r_index] = comb_mse[r,3]
end

yvals = map(i -> string("lr ", i), rates_order)
trace = heatmap(z = vals, y = yvals, x = layers_order)
data = [trace]
savefig(plot(data), string("/users/joeldacosta/desktop/layers_lr_heatmap_max_profit.html"))









################################################################################
##All configuration Profits
x0 = Array(TotalProfits[:profit])
trace = histogram(;x=x0, name="Configuration Profits")
data = [trace]
savefig(plot(data), string("/users/joeldacosta/desktop/profit pdf.html"))
################################################################################
##OGD Learning Rates BoxPlot####################################################

ogd_mse_query = "select tp.configuration_id, learning_rate, avg(training_cost) training_cost
            from training_parameters tp
            inner join epoch_records er on er.configuration_id = tp.configuration_id
            where tp.configuration_id >= $min_config
                and tp.category = \"FFN\"
                and er.category = \"OGD\"
            group by tp.configuration_id, learning_rate
            having training_cost not null"

ogd_mse_results = RunQuery(ogd_mse_query)
ogd_mse_results[:,1] = Array{Int64,1}(ogd_mse_results[:,1])
ogd_mse_results[:,2] = Array{Float64,1}(ogd_mse_results[:,2])
ogd_mse_results[:,3] = Array{Float64,1}(ogd_mse_results[:,3])
ogd_mse_groups = by(ogd_mse_results, [:learning_rate], df -> [df])
general_boxplot(ogd_mse_groups, "OGD learning rate", "OGD mse", :training_cost)


################################################################################
##OGD Learning Rates BoxPlot####################################################

lr_query = "select configuration_id, learning_rate from training_parameters where configuration_id >= $min_config and category = \"FFN-OGD\""
learning_results = RunQuery(lr_query)
learning_results[:,1] = Array{Int64,1}(learning_results[:,1])
learning_results[:,2] = Array{Float64,1}(learning_results[:,2])

learning_returns = join(TotalProfits, learning_results, on = :configuration_id)
groups = by(learning_returns, [:learning_rate], df -> [df])

general_boxplot(groups, "OGD learning rate", "OGD profit", :profit)

################################################################################
##FFN Learning Rates MSE Boxplot

lr_msequery = "select tp.configuration_id, learning_rate, min(testing_cost) min_test_cost
            from training_parameters tp
            inner join epoch_records er on er.configuration_id = tp.configuration_id
            where tp.configuration_id >= $min_config and tp.category = \"FFN\"
                and er.category = \"FFN-SGD\"
            group by tp.configuration_id, learning_rate
            "
lrmse_results = RunQuery(lr_msequery)
lrmse_results[:,1] = Array{Int64,1}(lrmse_results[:,1])
lrmse_results[:,2] = Array{Float64,1}(lrmse_results[:,2])
lrmse_results[:,3] = Array{Float64,1}(lrmse_results[:,3])

lr_groups = by(lrmse_results, [:learning_rate], df -> [df])

general_boxplot(lr_groups, "learning rate", "learning_rate_mse", :min_test_cost)

################################################################################
##FFN Learning Rates Profit Boxplot


lr_query = "select configuration_id, learning_rate from training_parameters where configuration_id >= $min_config and category = \"FFN\""
learning_results = RunQuery(lr_query)
learning_results[:,1] = Array{Int64,1}(learning_results[:,1])
learning_results[:,2] = Array{Float64,1}(learning_results[:,2])

learning_returns = join(TotalProfits, learning_results, on = :configuration_id)
groups = by(learning_returns, [:learning_rate], df -> [df])

general_boxplot(groups, "learning rate", "learning_rate_profits", :profit)

################################################################################
##Layers Profit Boxplot

function getlayerstruc(set_name)
    return ascii(split(split(set_name, "_")[1])[end])
end

function general_boxplot(layer_groups, prefix, filename, variable_name)

    y_vals = layer_groups[1,2][:,variable_name]
    trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[1,1]))
    data = [trace]

    for i in 2:size(layer_groups,1)
        y_vals = layer_groups[i,2][:,variable_name]
        trace = box(;y=y_vals, name = string(prefix, " ", layer_groups[i,1]))
        push!(data, trace)
    end

    savefig(plot(data), string("/users/joeldacosta/desktop/", filename, ".html"))
end

layers_query = string("select configuration_id, experiment_set_name from configuration_run where configuration_id >= $min_config")
layers_results = RunQuery(layers_query)

layers_results[:layers] = map(getlayerstruc, Array(layers_results[:experiment_set_name]))
layers_results[:,1] = Array{Int64,1}(layers_results[:,1])
layers_results[:,2] = Array{String,1}(layers_results[:,2])
layer_returns = join(TotalProfits, layers_results, on = :configuration_id)
groups = by(layer_returns, [:layers], df -> [df])

general_boxplot(groups, "layers", "layer_boxes", :profit)

################################################################################
##Layer Best MSE Boxplot


layer_msequery = string("
select er.configuration_id, min(testing_cost) min_test_cost, experiment_set_name
from epoch_records er
inner join configuration_run cr on cr.configuration_id = er.configuration_id
where er.configuration_id >= $min_config
and category = \"FFN-SGD\"
group by er.configuration_id, sae_config_id")
layer_mseresults = RunQuery(layer_msequery)
layer_mseresults[:,1] = Array{Int64,1}(layer_mseresults[:,1])
layer_mseresults[:,2] = Array{Float64,1}(layer_mseresults[:,2])
layer_mseresults[:,3] = Array{String,1}(layer_mseresults[:,3])
layer_mseresults[:layers] = map(getlayerstruc, Array(layer_mseresults[:experiment_set_name]))

layer_msegroups = by(layer_mseresults, [:layers], df -> [df])

general_boxplot(layer_msegroups, "layers", "layer_bestmse_boxes", :min_test_cost)

################################################################################
##Layer Last MSE Boxplot
layer_last_query = string("
select er.configuration_id, avg(testing_cost) min_test_cost, experiment_set_name
from epoch_records er
inner join configuration_run cr on cr.configuration_id = er.configuration_id
where er.configuration_id >= $min_config
and category = \"FFN-SGD\"
and epoch_number > 1900
group by er.configuration_id, experiment_set_name
order by min_test_cost")

layer_lastmseresults = RunQuery(layer_last_query)
layer_lastmseresults[:,1] = Array{Int64,1}(layer_lastmseresults[:,1])
layer_lastmseresults[:,2] = Array{Float64,1}(layer_lastmseresults[:,2])
layer_lastmseresults[:,3] = Array{String,1}(layer_lastmseresults[:,3])
layer_lastmseresults[:layers] = map(getlayerstruc, Array(layer_lastmseresults[:experiment_set_name]))

layer_lastmsegroups = by(layer_lastmseresults, [:layers], df -> [df])

general_boxplot(layer_lastmsegroups, "layers", "layer_lastmse_boxes", :min_test_cost)



################################################################################
##SAE Boxplot


function sae_boxplot(profit_groups, filename, variable_name)

    encodings = Dict()
    for id in profit_groups[1]
        encodings[id] = OutputSize(GetAutoencoder(ReadSAE(id)[1]))
    end

    y_vals = profit_groups[1,2][:,variable_name]
    trace = box(;y=y_vals, name = string("encoding ", encodings[profit_groups[1,1]]))
    data = [trace]

    for i in 2:size(profit_groups,1)
        y_vals = profit_groups[i,2][:,variable_name]
        trace = box(;y=y_vals, name = string("encoding ", encodings[profit_groups[i,1]]))
        push!(data, trace)
    end

    d = Dict()
    for t in data
        d[parse(Int, SubString(string(t[:name]),10))] = t
    end

    data = map(i -> d[i], sort(map(i -> Int(i), keys(d))))

    savefig(plot(data), string("/users/joeldacosta/desktop/", filename, ".html"))
end


sae_query = string("select configuration_id, sae_config_id from configuration_run where configuration_id >= $min_config")
sae_results = RunQuery(sae_query)

sae_results[:,1] = Array{Int64,1}(sae_results[:,1])
sae_results[:,2] = Array{Int64,1}(sae_results[:,2])
sae_returns = join(TotalProfits, sae_results, on = :configuration_id)
groups = by(sae_returns, [:sae_config_id], df -> [df])

sae_boxplot(groups, "sae boxplots")



################################################################################
##SAE Best Epoch


sae_msequery = string("
select er.configuration_id, min(testing_cost) min_test_cost, sae_config_id
from epoch_records er
inner join configuration_run cr on cr.configuration_id = er.configuration_id
where er.configuration_id >= $min_config
and category = \"FFN-SGD\"
group by er.configuration_id, sae_config_id")
sae_mseresults = RunQuery(sae_msequery)
sae_mseresults[:,1] = Array{Int64,1}(sae_mseresults[:,1])
sae_mseresults[:,2] = Array{Float64,1}(sae_mseresults[:,2])
sae_mseresults[:,3] = Array{Int64,1}(sae_mseresults[:,3])
mse_groups = by(sae_mseresults, [:sae_config_id], df -> [df])
sae_boxplot(mse_groups, "sae_mse_boxplots", :min_test_cost)





################################################################################
##SAE Last 100 Epochs

sae_last_query = string("
select er.configuration_id, epoch_number, avg(testing_cost) min_test_cost, sae_config_id
from epoch_records er
inner join configuration_run cr on cr.configuration_id = er.configuration_id
where er.configuration_id >= $min_config
and category = \"FFN-SGD\"
and epoch_number > 1900
group by er.configuration_id
order by min_test_cost")
sae_lastresults = RunQuery(sae_last_query)
sae_lastresults[:,1] = Array{Int64,1}(sae_lastresults[:,1])
sae_lastresults[:,2] = Array{Int64,1}(sae_lastresults[:,2])
sae_lastresults[:,3] = Array{Float64,1}(sae_lastresults[:,3])
sae_lastresults[:,4] = Array{Int64,1}(sae_lastresults[:,4])
last_groups = by(sae_lastresults, [:sae_config_id], df -> [df])
sae_boxplot(last_groups, "sae_lastmse_boxplots", :min_test_cost)










################################################################################
##Init SAE Training
init_query = string("
select er.configuration_id, min(testing_cost) min_test_cost, initialization init
from epoch_records er
inner join network_parameters np on np.configuration_id = er.configuration_id
where np.category = \"SAE\"
group by er.configuration_id, init
having min_test_cost not null")
init_results = RunQuery(init_query)

init_results[:,1] = Array{Int64,1}(init_results[:,1])
init_results[:,2] = Array{Float64,1}(init_results[:,2])
init_results[:,3] = Array{String,1}(init_results[:,3])
init_groups = by(init_results, [:init], df -> [df])
general_boxplot(init_groups, "Initialization", "init_boxplots", :min_test_cost)








################################################################################
##Init SAE Training



var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))
data_config = DatasetConfig(ds, datasetname,  5000,  [1, 7, 30],  [0.6],  [0.8, 1.0],  [2], var_pairs)


ds = GenerateDataset(167, 5000, var_pairs)

price_plot = plot(Array(ds), name = ["stock 1" "stock 2" "stock 3" "stock 4" "stock 5" "stock 6"])
savefig(price_plot, "/users/joeldacosta/desktop/PriceGraphs.html")



#end
