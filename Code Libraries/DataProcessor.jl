push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using DataGenerator

function LogDiff(x1, x2)
    return log(e, x2) - log(e, x1)
end

function GenerateLogFluctuations(series, delta)
    fluctuations = []

    for i in 1:delta
        push!(fluctuations, nothing)
    end

    for i in (delta+1):length(series)
        push!(fluctuations, LogDiff(series[i-delta], series[i]))
    end

    return fluctuations
end

function ProcessSeriesDataset(price_series::Dict, deltas)
    new_series = Dict()

    for r in price_series
        for d in deltas
            new_series[string(r[1], "_", d)] = GenerateLogFluctuations(r[2], d)
        end
    end

    return new_series
end

function ValidateGeneration(price_series, deltas, num_stocks)
    for sn in 1:num_stocks
        for d in deltas

            s = price_series[string("stock", sn)]
            sd = new_series[string("stock", sn, "_", d)]

            correct = true

            for i in (1+d):length(sd)
                correct = correct && LogDiff(s[(i-d)], s[i]) == sd[i]
            end

            println(sn, " ", d, " ", correct)

        end
    end
end

#price_series = GenerateDataset(1)
#deltas = [1, 30, 180]
#new_series = ProcessSeriesDataset(price_series, deltas)
#ValidateGeneration(price_series, deltas, 9)
