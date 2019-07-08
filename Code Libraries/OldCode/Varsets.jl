
#var_pairs = ((0.9, 0.5), (0.9, 0.2), (-0.8, 0.55), (-0.8, 0.15), (0.05, 0.4), (0.05, 0.1))

var_pairs = ((0.9, 0.5),
             (0.7, 0.2),
             (0.05, 0.4),
             (0.05, 0.5),
             (0.04, 0.1),
             (0.02, 0.15),
             (0.01, 0.05),
             (-0.8, 0.55),
             (-0.4, 0.15),
             (-0.1, 0.2))

function GenerateOneAssetConfig()
    return [(0.6, 0.05)]
end

function GenerateTwoAssetConfig()

    means = (0.05, 0.6)
    variances = (0.02, 0.4)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2])))
    end

    for v in variances
            push!(allcombos, ((means[1],v),(means[2], v)))
    end

    return allcombos
end

function GenerateThreeAssetConfig()

    means = (0.05, 0.2, 0.6)
    variances = (0.02, 0.1, 0.4)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2]),(m, variances[3])))
    end

    for v in variances
        push!(allcombos, ((means[1], v),(means[2], v),(means[3], v)))
    end

    return allcombos
end

function GenerateFourAssetConfig()

    means = (0.05, 0.2, 0.4, 0.8)
    variances = (0.02, 0.1, 0.3, 0.6)

    allcombos = []
    for m in means
        push!(allcombos, ((m, variances[1]),(m, variances[2]),(m, variances[3]),(m, variances[4])))
    end

    for v in variances
        push!(allcombos, ((means[1],v),(means[2],v),(means[3],v),(means[4],v)))
    end

    return allcombos
end
