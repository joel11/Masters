module DataGenerator
using DataFrames

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

function GenerateDataset(seed, steps, variation_pairs)
    srand(seed)

    df = DataFrame()

    for i in 1:length(variation_pairs)
        start_price = Float64.(rand(10:60))
        ts_prices = GenerateTimeSeries(start_price, variation_pairs[i][1], variation_pairs[i][2], steps)
        df[parse(string("stock", i))] = ts_prices
    end

    return df
end

function PlotPrices()

    #using Plots
    #plotlyjs()

    #all_pairs = ((0.9, 0.15), (0.9, 0.4), (0.9, 0.25), (-0.9, 0.15), (-0.9, 0.4), (-0.9, 0.25), (0.2, 0.09), (0.2, 0.1), (0.2, 0.15))
    all_pairs = ((0.9, 0.1), (0.9, 0.01), (0.9, 0.01), (-0.9, 0.1), (-0.9, 0.01), (-0.9, 0.01), (0.05, 0.01), (0.05, 0.01), (0.05, 0.01))
    var_pairs = (all_pairs[1], all_pairs[4])

    s = Int64.(round(rand()*100))
    ds = GenerateDataset(s, 5500, var_pairs)

    price_plot = plot(Array(ds))
    savefig(price_plot, "/users/joeldacosta/desktop/PriceGraphs.html")

end

end
