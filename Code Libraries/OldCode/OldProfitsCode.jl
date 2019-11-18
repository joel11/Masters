
function CreateProfitsFile(filename)
    #Original Setup
    TotalProfits = DataFrame()
    TotalProfits[:configuration_id] = []
    TotalProfits[:profit] = []

    #profit_array = Array(TotalProfits)
    #file_name = string("ActualTotalProfits", ".bson")
    #values = Dict(:profits => profit_array)
    #bson(file_name, values)
end

function UpdateTotalProfits(config_ids, over_ride, dataset, filename)

    #Original Setup
    #TotalProfits = DataFrame()
    #TotalProfits[:configuration_id] = []
    #TotalProfits[:profit] = []

    TotalProfits = BSON.load(string(filename, ".bson"))[:profits]

    current_configs = TotalProfits[:,1]
    if over_ride
        needed_configs = config_ids
    else
        needed_configs = collect(setdiff(Set(config_ids), Set(current_configs)))
    end

    println(length(needed_configs))

    ids = mapreduce(c -> string(c, ","), (x, y) -> string(x, y), needed_configs)[1:(end-1)]
    tic()
    println("Datapull")
    results = RunQuery("select * from prediction_results where configuration_id in ($ids) and predicted is not null")

    prediction_ids = get.(unique(results[:configuration_id]))

    println(string(toc()))

    for c in prediction_ids
        println(c)
        tic()

        config_results = results[(Array(results[:configuration_id]) .== c),:]
        try
            profits = GenerateTotalProfit(c, dataset, config_results)
            index = findin(TotalProfits[:,1], c)
            if length(index) > 0
                TotalProfits[index[1],2] = profits
            else
                TotalProfits = cat(1, TotalProfits, [c profits])
            end
        catch y
            println(string("Error on: ", c))
            continue
        end

        println(string(toc()))
    end

    profit_array = Array(TotalProfits)
    file_name = string(filename, ".bson")
    values = Dict(:profits => profit_array)
    bson(file_name, values)
end

function ReadProfits(filename)
    pa = BSON.load(string(filename,".bson"))

    TotalProfits = DataFrame()
    TotalProfits[:configuration_id] = pa[:profits][:,1]
    TotalProfits[:profit] = pa[:profits][:,2]
    return TotalProfits
end
