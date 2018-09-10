function gen_pow(x)
    function run_pow(y)
        return (y^x)
    end
    return (run_pow)
end


square = gen_pow(2)


cube = gen_pow(3)

square(3)
cube(4)


for i in 1:5
    if i ==4
        break
    end
    println(i)
end
