using DataFrames

using MLBase

actual =    [1, 1,   1,   0, 0,   0, 0,     0];
model =     [1, 1,   0,   0, 0,   1, 1,     0];

df = DataFrame(confusmat(2, gt .+ 1, pred .+ 1))

names!(df, [:model_no_trade, :model_trade])


writetable("/users/joeldacosta/desktop/confusion.csv", df)
