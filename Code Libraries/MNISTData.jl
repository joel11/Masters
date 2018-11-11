module MNISTData

using MNIST
using TrainingStructures
export GenerateData



function GenerateData()

    trainingset, trainingsetlabels = traindata()

    trainingdata = trainingset[:, 1:1000]
    traininglabels = trainingsetlabels[1:1000]

    testingdata = trainingset[:, 50001:60000]
    testinglabels = trainingsetlabels[50001:60000]

    validationdata, validationlabels = testdata()

    training_labels = fill(0.0, (10, length(traininglabels)))
    testing_labels = fill(0.0, (10, length(testinglabels)))
    validation_labels = fill(0.0, (10, length(validationlabels)))


    for i in 1:length(traininglabels)
        training_labels[Int64.(traininglabels[i])+1, i] = 1
    end

    for i in 1:length(testinglabels)
        testing_labels[Int64.(testinglabels[i])+1, i] = 1
    end

    for i in 1:length(validationlabels)
        validation_labels[Int64.(validationlabels[i])+1, i] = 1
    end

    scaled_training_data = (trainingdata')./255
    scaled_testing_data = (testingdata')/255
    scaled_validation_data = (validationdata')./255

    ds = DataSet(scaled_training_data, scaled_testing_data, scaled_validation_data, training_labels', testing_labels', validation_labels')
    return (ds)
end

end
