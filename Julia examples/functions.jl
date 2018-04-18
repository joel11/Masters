#Functions##################################################

f(x) = x^2
methods(f)

function mltpl(x, y)
    print("First is $x Second is $y")
    return x * y
end

mltpl(3, 4)


function mltpl3(x, y)
    print("First is $x Second is $y")
    x + y
    x * y
end

mltpl3(3, 4)

function math_func(a, b)
    a + b, a - b, a * b
end

ans1, ans2, ans3 = math_func(3, 4)
ans2

##Arguments##################################################
function func(a, b, c = 100)
    print("We have $a, $b, $c")
end

func(1, 2)

function func2(a, b, c = 100; p = 100, q = "red")
    println("First ordered is $a")
    println("Second ordered is $b")
    println("Thir oreders was optional")
    println("Optional 100? is $c")
    println("keyword p: $p")
    println("keyword q: $q")
    a * b
end

func2(10, 1, p=2, 3,  q = "test")

#Variable Arguments############################################

function func3(args...)
    println(args)
end

func3("this", 1, "test")

function surg(string_array...)
    string_items = join( string_array, ", ", " and ")
    println("All: $string_items")
end

surg("one", "two", "three")
surg("one")

function argues(a, b, c...)
    println("$a, $b and $c")
end

argues(1, 4, 5, "julia")

function fun_func(; a...)
    a
end

fun_func(var1 = "julia", var2 = 4, val1 =3)


##Arrays to functions##########################################

xvals = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
function sqr(a)
    return a^2
end

map(sqr, xvals)

array_1 = [3, 4]
tuple_1 = (3, 4);

function h(x, y)
    return 3*x + 2*y
end

h(array_1...)

array_primes = [2, 3, 5, 7, 11, 13]

function add_ele(a)
    push!(a, 23)
    a[1] = 1
    a
end

add_ele(array_primes)

function m(x::Int)
    3 * x
end

m("a")

function m2{T <: Real}(x::T)
    print("$x is of type $T")
end

m2(3)
m2(3.5)


##Stabby Functions####################################

map(x -> 2x^2 + 3x - 2, [1, 2, 3, 4, 5])

map([1, 2, 3, 4, 5]) do x
    2x^2 + 3x - 2
end


##Functions as arguments##############################

function string_func(s)
    str = s()
    print("I love $str")
end

function luv()
    return "Julia"
end

string_func(luv)
