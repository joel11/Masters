a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
msize = 3
num_batches = Int64.(floor(size(a)[1]/msize))

for i in 1:(num_batches)
    print(a[((i-1)*msize+1):i*msize])
end

#1:3
#4:6
#7:9
#10:12




wone  = [[1 2];[3 4]]
wtwo = [[5 6];[7 8]]
wthree = [[1 1];[1 2]]



a = Array{Int,3}(size(wone)[1],size(wone)[2],0)
a = cat(3, a, wone)
a = cat(3, a, wtwo)
