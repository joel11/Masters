using DataFrames


da2=@data([1, 3, 5, 8, 10, 12])

mapreduce(x -> x^2, +, da2)

df1 = DataFrame(A = 1:5, b = ["Y", "Y", "N", "Y", "N"])

df2 = DataFrame();

df2[:A] = 1:5
df2[:B] = ["This", "is", "a", "new", "row"]

size(df2)


##Content of DF#######################################

df3 = DataFrame();
rand(3)

df3[:A] = rand(15)
df3[:B] = rand(15)
df3[:C] = rand(15)
df3[:D] = rand(15);

head(df3)
tail(df3, 2)

names(df3)

showcols(df3)
eltypes(df3)
dump(df3)
describe(df3)

df3[[:A, :C]]
df3[3:5, :]
df3[[3, 5], [:A, :C]]


##Joins & Groups##############################################
subjects = DataFrame(Number = [100,101, 102, 103], Stage = ["I", "III", "II", "I"])
treatment = DataFrame(Number = [103, 102, 101, 100], Treatment = ["A", "B", "A", "B"])

df5 = join(subjects,treatment, on = :Number)

subjects = DataFrame(Number = [100, 101, 102, 103, 104, 105], Stage = ["I", "III", "II", "I", "II", "II"])

df7 = join(subjects, treatment, on = :Number, kind = :outer)

##Grouping & Splicing

df8 = DataFrame(Group = rand(["A", "B", "C"], 15),
Variable1 = randn(15), Variable2 = rand(15))

by(df8, :Group, size)

by(df8, :Group, df -> DataFrame(Count = size(df)[1]))

aggregate(df8, :Group, [mean, std])

groupby(df8, :Group)

for i in groupby(df8, :Group)
    print(i)
end

groupby(df8, :Group)[2]

#sub(df8, [5, 7])

##Sorting & Duplicates & NA##########################################

sort!(df8, cols = [:Group, :Variable1], rev = true)
sort!(df8, cols = [:Group, :Variable1, :Variable2], rev = (false, false, true))

df9 = DataFrame(A = [1, 2, 2, 3, 4, 5], B = [11, 12,12,13,14,15], C = ["A", "B", "B", "C", "D", "E"])
unique!(df9)

df10 = DataFrame(A = 1:10, B = 11:20, C=21:30)
df10[3, :A] = NA
df10[4, :B] = NA
df10[[3, 9], :C] = NA

complete_cases(df10)

complete_cases!(df10)

isna(df10[:A])
findin(isna(df10[:A]), true)
find(isna(df10[:A]))

rows, cols = size(df10)

for i in 1:cols
    deleterows!(df10, find(isna(df10[:, i])))
end

df10


##Renaming
rename!(df10, :C, :QWE)
df10

##Converting

array_a = convert(Array, df10[:A])
array_a
