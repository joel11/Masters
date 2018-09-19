
module OutputLibrary

using TrainingStructures,  FFN, RBM, NeuralNetworks
using Plots
plotlyjs()

export WriteOutputGraphs, PlotRBMInputOutput

function WriteOutputGraphs(network, rbm_records, ffn_records, validation_data, output_dir)

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    #Add File Names / Folder
    PlotInputOutput(network, validation_data, 10, output_dir)
    PlotRBMInputOutput(rbm_records, validation_data, 10, output_dir)

    PlotActivationGraphs(rbm_records, output_dir)
    PlotRBMWeights(rbm_records, 2, output_dir)

    #PlotFFNWeights(ffn_records, output_dir)
    #PlotRBMWeights(rbm_records, output_dir)

    mc_plot = PlotEpochLines(rbm_records, ffn_records, MeanCostErrors, "Mean Cost Errors")
    ce_plot = PlotEpochLines(rbm_records, ffn_records, CrossEntropyErrors, "Cross Entropy Errors")
    rt_plot = PlotEpochLines(rbm_records, ffn_records, RunTimes, "Run Times")
    ve_plot = PlotEpochLines(rbm_records, ffn_records, ValidationErrors, "Validation Errors")
    wr_plot = PlotEpochLines(rbm_records, ffn_records, WeightRateChanges, "Weight Rate Changes")

    savefig(plot(mc_plot, ce_plot, rt_plot, ve_plot, wr_plot, layout = 5, size = (800,800)), string(output_dir , "LineGraphs.html"))
end

##Input / Output###############################################################

function PlotInputOutput(network, validation, number_samples, output_dir)
    function get_plot(data)
        return(heatmap(reshape(data, (28,28))))
    end

    samples = validation[rand(1:size(validation)[1], number_samples), :]
    output = Feedforward(network, samples)[end]

    pairs = map(i -> (hcat(samples[i,:], output[i,:])), 1:number_samples)
    combos = reduce(hcat, pairs)
    plots = map(x -> get_plot(combos[:,x]), 1:size(combos)[2])
    savefig(plot(plots..., size = (800, 800)),  string(output_dir, "InputOutput.html"))
end

function PlotRBMInputOutput(rbm_records, validation, number_samples, output_dir)
    function get_plot(data)
        return(heatmap(reshape(data, (28,28))))
    end

    samples = validation[rand(1:size(validation)[1], number_samples), :]
    max_epoch = length(rbm_records[1])

    for l in 1:length(rbm_records)
        layer_network = NeuralNetwork(map(l -> rbm_records[l][max_epoch].network.layers[1], 1:l))
        output = ReconstructVisible(layer_network, validation)

        pairs = map(i -> (hcat(samples[i,:], output[i,:])), 1:number_samples)
        combos = reduce(hcat, pairs)
        plots = map(x -> get_plot(combos[:,x]), 1:size(combos)[2])
        savefig(plot(plots..., size = (800, 800)),  string(output_dir, "RBM_InputOutput_", l, ".html"))
    end
end

##Weights as maps###############################################################

function PlotRBMWeights(rbm_records, max_layer, output_dir)
    epochs = length(rbm_records[1])
    layers = length(rbm_records)

    #layer 1
    heatmaps = reduce(vcat, map(e -> PlotRBMWeights(1, e, 10, rbm_records), 1:epochs))

    for l in 2:max_layer
        heatmaps = vcat(heatmaps, reduce(vcat, map(e -> PlotRBMWeights(l, e, 10, rbm_records), 1:epochs)))
    end
    savefig(plot(heatmaps..., size = (800, 800)),  string(output_dir, "RBMWeights.html"))
    #savefig(plot(plots..., layout = number_of_weights, title = "L$layer_num : E$epoch_num"),  "testfile.html")
end

function PlotRBMWeights(layer_num, epoch_num, number_of_weights, rbm_records)

    weights = rbm_records[layer_num][epoch_num].network.layers[1].weights[2:end,:]
    random_weights = weights[:,rand(1:size(weights)[2], number_of_weights)]
    plot_size = Int64.(floor(sqrt(size(weights)[1])))

    plots = []

    for i in 1:number_of_weights
        push!(plots, heatmap(reshape(random_weights[1:plot_size^2,i], (plot_size,plot_size)), title = "L$layer_num : E$epoch_num"))
    end

    return (plots)
end

##Weights ######################################################################


function PlotActivationGraphs(rbm_records, output_dir)

    recs = map(x -> map(r -> r.hidden_activation_likelihoods , x), rbm_records)
    heatmapnums = reduce(vcat, map(a -> reduce(vcat, a), recs))
    plots = map(heatmap, heatmapnums)
    savefig(plot(plots..., size = (1000, 1000)),  string(output_dir, "HiddenActivations.html"))
end

function PlotFFNWeights(ffn_records, output_dir)
    number_epochs = length(ffn_records)
    p = map( x-> PlotFFNWeightEpoch(ffn_records, x), 1:number_epochs)
    savefig(plot(p...,  layout = number_epochs, size=(900,900)), string(output_dir, "FFNWeights_histogram.html"))
end

function PlotFFNWeightEpoch(ffn_records, epoch_number)

    weights =  map( x -> x.weights, ffn_records[epoch_number].network.layers)

    colours = ["red", "blue", "green", "yellow", "purple", "brown", "orange", "teal", "gold", "silver", "violet", "cyan"]
    weight_plot = histogram(weights[1], alpha = 0.1, color=colours[1], title = "FFN Weights Epoch $epoch_number")
    for l in 2:length(weights)
        histogram!(weight_plot, weights[l], alpha = 0.1, color = colours[l])
    end

    return (weight_plot)
end

function PlotRBMWeights(rbm_records, output_dir)

    number_epochs = length(rbm_records[1])
    p = map( x-> PlotRBMWeightEpoch(rbm_records, x), 1:number_epochs)
    savefig(plot(p...,  layout = number_epochs, size=(900,900)), string(output_dir, "RBMWeights_histogram.html"))
end

function PlotRBMWeightEpoch(rbm_records, epoch_number)
    colours = ["red", "blue", "green", "yellow", "purple", "brown", "orange"]

    rbm_weights = map(x -> x[epoch_number].network.layers[1].weights, rbm_records)
    weight_plot = histogram(rbm_weights[1], alpha = 0.1, color=colours[1], title = "RBM Weights Epoch $epoch_number")
    for i in 2:length(rbm_weights)
        histogram!(weight_plot, rbm_weights[i], alpha = 0.1, color = colours[i])
    end

    return(weight_plot)
end

################################################################################


function PlotEpochLines(rbm_records, ffn_records, value_function, ylab)
    rbm_values = CatEpochAttributes(rbm_records, value_function)
    ffn_values = value_function(ffn_records)

    max_y =  max(maximum(rbm_values[:,3]), maximum(ffn_values[:,2])) + 1

    first_epoch = rbm_values[rbm_values[:,1] .== 1, :]
    epoch_plot = plot(first_epoch[:,3], ylims = (0, max_y), labels = "RBM 1",ylabel = ylab, xlabel = "Epoch")

    for i in 2:maximum(rbm_values[:,1])
        epoch_vals = rbm_values[rbm_values[:,1] .== i, :]
        plot!(epoch_plot, epoch_vals[:,3],  labels = "RBM $i")
    end

    plot!(epoch_plot, ffn_values[:,2],  labels = "FFN")

    return (epoch_plot)
end


function PlotEpochLines2(rbm_records, ffn_records, value_function, ylab)
    rbm_values = CatEpochAttributes(rbm_records, value_function)
    ffn_values = value_function(ffn_records)

    max_error =  max(maximum(rbm_values[1][:,3]), maximum(ffn_values[:,2])) + 1
    max_x =  max(maximum(rbm_values[1][:,2]), maximum(ffn_values[:,1]))

    rbm_traces = map(d -> PlotlyJS.scatter(;x=d[:,2], y=d[:,3], mode="lines", name= string("RBM ", Int64.(d[1,1]))), rbm_values)
    ffn_trace = PlotlyJS.scatter(;x=ffn_values[:,1], y = ffn_values[:,2], mode="lines", name = "FFN")

    #layout = PlotlyJS.Layout(xaxis_range=[1, max_x], yaxis_range=[0, max_error])

    epoch_plot = PlotlyJS.plot([rbm_traces; ffn_trace])#, layout)

    return (epoch_plot)
end

function CatEpochAttributes(records, attribute_function)
    epoch_records = map(attribute_function, records)
    return(reduce(vcat, map((x, y) -> [fill(x, size(y)[1]) y], (1:size(records)[1]), epoch_records)))
end

function WeightRateChanges(records::Array{EpochRecord,1})
    function NumberWeightChanges(e::EpochRecord)
        return reduce(vcat, e.weight_change_rates')# hcat(1:length(e.weight_change_rates), reduce(vcat, e.weight_change_rates'))
    end
    return(reduce(vcat,map((x, y) -> [fill(x, size(y)[1]) y], (1:size(records)[1]), map(NumberWeightChanges, records))))
end

function MeanCostErrors(records::Array{EpochRecord,1})
    return(reduce(vcat,map(x -> [x.epoch_number x.mean_cost_error], records)))
end

function ValidationErrors(records::Array{EpochRecord,1})
    return(reduce(vcat,map(x -> [x.epoch_number x.validation_cost_error], records)))
end

function RunTimes(records::Array{EpochRecord,1})
    return(reduce(vcat,map(x -> [x.epoch_number x.run_time], records)))
end

function EnergyRatios(records::Array{EpochRecord,1})
    return(reduce(vcat,map(x -> [x.epoch_number x.energy_ratio], records)))
end

function CrossEntropyErrors(records::Array{EpochRecord,1})
    return(reduce(vcat,map(x -> [x.epoch_number x.cross_entropy_error], records)))
end

end
