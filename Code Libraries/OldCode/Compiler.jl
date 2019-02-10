using PackageCompiler

# This command will use the `runtest.jl` of `ArgParse` + `SnoopCompile` to find out what functions to precompile!
# `force = false` to not force overwriting Julia's current system image
compile_package("ArgParse", "SnoopCompile", force = false, reuse = false)

# Build again, reusing the snoop file
compile_package("ArgParse", "SnoopCompile", force = false, reuse = true)

# You can define a file that will get run for snooping explicitly like this:
# this makes sure, that binary gets cached for all functions called in `for_snooping.jl`
compile_package(("ArgParse", "for_snooping.jl"))

# If you used force and want your old system image back (force will overwrite the default system image Julia uses) you can run:
#revert()

# Or if you simply want to get a native system image e.g. when you have downloaded the generic Julia install:
#force_native_image!()

# Build an executable
build_executable(
    "testcompile.jl"
    #, # Julia script containing a `julia_main` function, e.g. like `examples/hello.jl`
    #snoopfile = "call_functions.jl", # Julia script which calls functions that you want to make sure to have precompiled [optional]
    #builddir = "/users/joeldacosta/Desktop/" # that's where the compiled artifacts will end up [optional]
)

# Build a shared library
build_shared_lib("testcompile.jl")
