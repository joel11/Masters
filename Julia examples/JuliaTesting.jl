Array{Int64}(4)
["asd", 1]

arr = Array(Integer,3,2)
arr[1,2] = 3
arr[1,1] = 2
#arr[2,1]

a, b, c = 1, 3 - 1, abs(-1.1)

muladd(2, 2, 3)

twotimes(v) = v*2

function testfunc(v)
    println("this is a test of $v printing")
end

testfunc(10)

########################################################################

wikiEVDraw = readdlm("wikipediaEVDraw.csv", ',')

datetimerep = DateTime(wikiEVDraw[1,1], "d u y")

wikiEVDraw[1,:]

for num = 3:7
    println("value $num")
end

values = [23, "test", 2.3]

for v in values
    println(v)
end

size(m)[1]



col1 = wikiEVDraw[:,1]
for i in 1:(size(wikiEVDraw)[1])
    col1[i] = DateTime(wikiEVDraw[i,1], "d u y")
end

dayssincemar22(x) = Dates.datetime2rata(x) - Dates.datetime2rata(col1[54])
epidays = Array{Int64}(54)

for i = 1:54
    epidays[i] = dayssincemar22(col1[i])
end

wikiEVDraw[:,1] = epidays

########################################################################
using PyPlot


plot(epidays, ".")
savefig("test.png")


a = rand()
if a > 0.5
    println("bigger")
end
