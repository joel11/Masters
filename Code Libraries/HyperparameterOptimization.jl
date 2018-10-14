module HyperparameterOptimization

export HyperparameterRangeSearch, GraphHyperparameterResults, ChangeLearningRate

using NetworkTrainer
using Plots
plotlyjs()


function ChangeLearningRate(parameters,val)
    parameters.learning_rate = val
end


function HyperparameterRangeSearch(dataset, network_parameters, rbm_parameters, base_ffn_parameters, attribute_change_function, values)
    results = []

    for i in values

        ChangeLearningRate(base_ffn_parameters, i)
        network, rbm_records, ffn_records = TrainFFNNetwork(dataset, network_parameters, rbm_parameters, base_ffn_parameters)

        run_times = map(x -> x.run_time, ffn_records)
        test_accuracy = map(x -> x.test_accuracy, ffn_records)
        test_cost = map(x -> x.test_cost, ffn_records)
        epoch_numbers = map(x -> x.epoch_number, ffn_records)

        push!(results, (i, epoch_numbers, test_cost, test_accuracy, run_times))
    end

    return (results)
end


function GraphHyperparameterResults(results, output_dir, file_name, val_name)

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    #Run Times
    max_y_runtimes = maximum(reduce(vcat, map(x -> x[5], results)))*1.1
    min_y_times =  minimum(reduce(vcat, map(x -> x[5], results)))*0.9

    runtime_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    runtime_series = map(x -> x[5], results)

    runtime_plot = plot(runtime_series, ylims = (min_y_times, max_y_runtimes), labels = runtime_labels, ylabel = "Run Times", xlabel = "Epoch")

    #Cost
    cost_max_y = maximum(reduce(vcat, map(x -> x[3], results)))*1.1
    cost_min_y =  minimum(reduce(vcat, map(x -> x[3], results)))*0.9

    cost_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    cost_series = map(x -> x[3], results)

    cost_plot = plot(cost_series, ylims = (cost_min_y, cost_max_y), labels = cost_labels, ylabel = "Costs", xlabel = "Epoch")

    #Accuracy

    accuracy_labels = reduce(hcat, map(x -> string(val_name, "=", string.(x[1])), results))
    accuracy_series = map(x -> x[4], results)

    accuracy_plot = plot(accuracy_series, ylims = (0, 1.1), labels = accuracy_labels, ylabel = "Accuracy", xlabel = "Epoch")

    savefig(plot(cost_plot, accuracy_plot, runtime_plot, layout = 3, size = (1000,1000)), string(output_dir , file_name, ".html"))
end




end
