module DataJSETop40

using DataFrames
using CSV

export ReadJSETop40Data

function ReadJSETop40Data()

    function MapNulls(col_data)
        col = []

        for i in 1: length(col_data)
            push!(col, isnull(col_data[i]) ? 0.0 : get(col_data[i]))
        end

        return col
    end

    dat = CSV.read("data/JSE_top40_2003_2018.csv")

    cols = names(dat)
    newdf = DataFrame()

    for i in 2:length(cols)
        newdf[cols[i]] = cumprod(MapNulls(dat[cols[i]]))
    end

    return (newdf)
end


end
