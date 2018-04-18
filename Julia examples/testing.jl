tempvar = Array{Any}(4)
fill!(tempvar, "test")


Number <: Any
subtypes(Any)
subtypes(Number)

(2+2)::Int64

function testfunc(a)
    v::Int64 = 16
    return v
end


#Can use immutable keyword instead of type
type Vector_2D
        x::Float64
        y::Float64
end

va = Vector_2D(1, 4)
va.y

methods(Vector_2D)
fieldnames(Vector_2D)
setfield!(va, :x, 2.0)

va.x = 5

type Vector3D{T}
    x::T
    y::T
    z::T
end

t = Vector3D(4, 5, 6)


is(5, 5.0)
==(5, 5.0)

import Base.+
+(a::Vector_2D, b::Vector_2D) = Vector_2D(a.x + b.x, a.y + b.y)

va + va

##More COmplex Parameters
type Relook{N, T<:Real}
        duration::T
end

Relook(5)
Relook{4, Int16}(40)

#########


type BloodPressure
    systolic::Int64
    diastolic::Int64

    function BloodPressure(s, d)
        s < 0 && throw(ArgumentError("Negative not allowed"))
        s <= d && throw(ArgumentError("s > d"))
        isa(s, Integer) || throw(ArgumentError("Only int"))
        isa(d, Integer) || throw(ArgumentError("Only int"))
        new(s, d)
    end

    function TestFunc()
        return s < d
    end
end

q =  BloodPressure(4, 3)
