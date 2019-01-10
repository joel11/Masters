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
