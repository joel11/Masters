module DataGenerator
using DataFrames

#Expand this to a OHLC format

export GenerateDataset

function GenerateTimeSeries(S0, mu, sigma, steps)
    T = 1.0
    dt = T / steps # dt

    prices = [S0]
    for m in 1:steps
        z = randn()
        st = prices[end] * exp((mu - 0.5 * ^(sigma,2)) * dt + sigma * sqrt(dt) * z)
        push!(prices, st)
    end

    return(prices)
end

function GenerateGroup(start_prices, variations, trend, steps)
    prices = []

    for i in 1:length(start_prices)
        push!(prices, GenerateTimeSeries(start_prices[i], trend, variations[i], steps))
    end

    return (prices)
end

function GenerateDataset(seed, steps)

    srand(seed)

    bull_prices = Float64.(rand(10:60, 3))
    bull_vars = [0.15, 0.5, 0.25]

    bear_prices = Float64.(rand(10:60, 3))
    bear_vars = [0.15, 0.4, 0.25]

    stable_prices = Float64.(rand(10:60, 3))
    stable_vars  = [0.09, 0.1, 0.15]

    bull_pricegroup = GenerateGroup(bull_prices, bull_vars, 0.9, steps)
    bear_pricegroup = GenerateGroup(bear_prices, bear_vars, -0.9, steps)
    stable_pricegroup = GenerateGroup(stable_prices, stable_vars, 0.2, steps)

    all_prices = vcat(bull_pricegroup, bear_pricegroup, stable_pricegroup)

    df = DataFrame(all_prices)
    colnames = map(x -> string("stock", x), 1:size(df)[2])
    names!(df.colindex, map(parse, colnames))

    return df
end

end
