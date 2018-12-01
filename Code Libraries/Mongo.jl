workspace()
#Pkg.add("BinaryBuilder")
#Pkg.add("BinaryProvider")



using BinaryProvider
Pkg.update("Mongoc")
#Pkg.add("AbstractString")
Pkg.build("Mongoc")
using Mongoc
using LibBSON
using Mongo
