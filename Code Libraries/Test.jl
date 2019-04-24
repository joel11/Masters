using Distributions
using DataFrames

d2 = DataFrame()
d2[:one] = [2, 4, 0, -1, 4]
d2[:two] = [13.5, -7, 1, 23, 9]
d2[:a] = [1, 2, 3, -5, -10]

function AddNoiseToDataFrame(df, variance)

    function AddNoise(x)
        return rand(Normal(x, variance))
    end

    newdf = DataFrame(AddNoise.(Array(df)))
    names!(newdf, names(df))

    return newdf
end
