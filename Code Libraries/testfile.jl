A = map(x->x, 1:40)
B = map(x->x, 31:70)
function lag(A, n)
    return map(x->sum(A[x]/A[(x-n)]*100), (1+n):length(A))
end


nsteps = [1, 3, 7]

d1 = reduce(hcat, map(x -> lag(A, x)[(1+maximum(nsteps)-x):end], nsteps))'
d2 = reduce(hcat, map(x -> lag(B, x)[(1+maximum(nsteps)-x):end], nsteps))'
data = vcat(d1, d2)




input = data
steps = [1, 3]
n_stocks = 2

function FormatPredictionDataset(data, nsteps, prediction_steps, number_stocks, partition_one, partition_two)

    groups = map(x -> data[(length(nsteps)*(x-1)+1):(length(nsteps)*(x-1)+length(prediction_steps)), :], 1:number_stocks)
    output = reduce(vcat, map(data -> reduce(hcat,map(x -> data[x, (1+prediction_steps[x]):end][1:(size(data)[2]-maximum(prediction_steps))] ,1:length(prediction_steps)))', groups))
    input = data[:, (1:size(output)[2])]

end
