Pkg.add("Distributions")
Pkg.add("DataFrames")


array1 = [1, 2, 3]


trans = array1'
transpose(array1)

array4 = [[1, 2, 3],[4,5,6]]
array4'

array5= [[1 2 3]; [4 5 6]]
length(array5)

repmat([1, 2], 3)

collect(linspace(0, 10, 5))

typeof(collect(0:1:5))

##Slicing##########################################

array12 = rand(10:20, 10, 5)
array12[:, 1]
array12[:, [2,5]]
array12[:, 2:4]
array12[:, 2:end]
find(array12[:, 1] .> 12


array13 = [1, 2, 3, 4]
push!(array13, 5)
unshift!(array13, 0)
array13


##Comprehensions###################################

array14 = []
for i in 1:5
    push!(array14, 3*i)
end

array15 = [3 * i for i in 1:5]
array14 == array15

sum(array14)
mean(array14)
std(array15)

5+array14
3*array14

array14 .* array15
array14 * array15'


##NA##################################################
array18 = [1, 2, NaN, 4, 5]
sum(array18)
isnan(array18)

using DataFrames
array19 = @data([1, 3, 4, 5, NA, 7, NA, 2])
dropna(array19)

##Tuples################################################

tup = (1, 2, 3, "hello")
typeof(tup)
a, b, c, seven = (1, 3, 5, 7)
tup2 = (1, 3, 5, 7, 9, 13, 15);

tup2[end]
tup2[2:4]
#tup2[1] = 4

tup3 = ((1, 2, 3), 1, 2, (3, 100, 1))
tup3[2]

##Dictionaries############################################

dict1 = Dict(1 => 77, 2=>66, 3=>1)
dict1 = Dict{Any, Any}(1 => 77, 2=>66, 3=>"three")
dict3 = Dict{Any, Any}("a" => 1, (2, 3) => "hello")
dict4 = Dict(:A => 300, :B => 305, :C =>309)
dict4[:B]

get(dict4, :H, "test")
in((:A => 300), dict4)
haskey(dict4, :D)
dict4[:D] = 301
haskey(dict4, :D)

dict3[:C] = 1000
dict3

delete!(dict4, :A)
keys(dict4)
values(dict4)
length(dict4)

for (k, v) in dict4
    println("The key $k has val $v")
end

p_vals = ["Abs", "Col", "chols"]
p_dict = Dict{AbstractString, AbstractString}()

for (s, n) in enumerate(p_vals)
    p_dict["x_$s"] = n
end


for k in sort(collect(keys(dict4)))
    println("$k is $dict4[k]")
end

sort(collect(keys(dict4)))
