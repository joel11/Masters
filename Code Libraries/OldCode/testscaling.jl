workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")
using DataProcessor, DataJSETop40

jsedata = ReadJSETop40Data()
exp_data = jsedata[:, [:ACL, :AGL, :AMS, :CRH, :CFR , :SOL]]
scaled_data = exp_data#normalizeset(exp_data)

data_raw = scaled_data
data_splits = SplitData(data_raw, [0.6])
processed_data = map(x -> ProcessData(x, [2], [2]), data_splits)
#saesgd_data, ogd_data, holdout_data = map(x -> CreateDataset(x[1], x[2], ep.data_config.training_splits), processed_data)
saesgd_data = (CreateDataset(processed_data[1][1], processed_data[1][2], [0.7, 1.0]))
#saesgd_data = NormalizeDataset(CreateDataset(processed_data[1][1], processed_data[1][2], ep.data_config.training_splits))
ns = NormalizeDatasetForTanh(saesgd_data)







minimum(ns.training_input)
minimum(ns.testing_input)
#minimum(Array(ns.training_output))
#minimum(Array(ns.testing_output))

maximum(ns.training_input)
maximum(ns.testing_input)
#maximum(Array(ns.training_output))
#maximum(Array(ns.testing_output))



layers =   (#("8 - ReLU", [input, 8, encoding], [ReluActivation,  LinearActivation]),
            #("15 - ReLU", [input, 15, encoding], [ReluActivation,  LinearActivation]),
            #("50 - ReLU", [input, 50, encoding], [ReluActivation,  LinearActivation]),
            #("100 - ReLU", [input, 100, encoding], [ReluActivation,  LinearActivation])
            #("30 - ReLU", [input, 30, encoding], [ReluActivation,  LinearActivation]),
            #("8x8 - ReLU", [input, 8, 8, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            #("15x15 - ReLU", [input, 15, 15, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            #("25x25 - ReLU", [input, 25, 25, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            #("50x50 - ReLU", [input, 50, 50, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            #("100x100 - ReLU", [input, 100, 100, encoding], [ReluActivation, ReluActivation,  LinearActivation])
            ("30x15 - ReLU",     [input, 30,  15, encoding],     [ReluActivation, ReluActivation, LinearActivation]),
            ("50x30x10 - ReLU",  [input, 50,  30, 10, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation]),
            ("100x50x20 - ReLU", [input, 100, 50, 20, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation])
            #("40x40 - ReLU", [input, 40, 40, encoding], [ReluActivation, ReluActivation,  LinearActivation]),
            #("8x8x8 - ReLU", [input, 8, 8, 8, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation]),
            #("15x15x15 - ReLU", [input, 15, 15, 15, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation]),
            #("30x30x30 - ReLU", [input, 30, 30, 30, encoding], [ReluActivation, ReluActivation, ReluActivation, LinearActivation])
            )
