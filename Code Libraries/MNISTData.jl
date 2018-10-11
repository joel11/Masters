module MNISTData

using MNIST
using TrainingStructures
export GenerateData



function GenerateData()

    trainingdata, traininglabels = traindata()
    validationdata, validationlabels = testdata()

    training_labels = fill(0.0, (10, length(traininglabels)))
    validation_labels = fill(0.0, (10, length(validationlabels)))

    i = 1

    for i in 1:length(traininglabels)
        training_labels[Int64.(traininglabels[i])+1, i] = 1
    end

    for i in 1:length(validationlabels)
        validation_labels[Int64.(validationlabels[i])+1, i] = 1
    end

    scaled_training_data = (trainingdata')./255
    scaled_validation_data = (validationdata')./255

    ds = DataSet(scaled_training_data, scaled_validation_data, training_labels', validation_labels')
    return (ds)
end

end
