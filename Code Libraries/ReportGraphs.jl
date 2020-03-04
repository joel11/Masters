workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using ExperimentGraphs
using DatabaseOps
using DataJSETop40

category_query_test = "select min(configuration_id) minid,
       max(configuration_id) maxid,
       count(distinct configuration_id) total_configs,
       case
       when experiment_set_name like 'Iteration2_1 Leaky ReLU Tests%' then 'Iteration2_1 Leaky ReLU Tests'
       when experiment_set_name like 'Iteration2 Leaky ReLU SAE%' then 'Iteration2 Leaky ReLU SAE'
       when experiment_set_name like 'Iteration2_1 Smaller Linear Tests%' then 'Iteration2_1 Smaller Linear Tests'
       when experiment_set_name like 'Iteration2_1 MAPE Tests FFN%' then 'Iteration2_1 MAPE Tests FFN'
       when experiment_set_name like 'Iteration2_1 CV Tests FFN%' then 'Iteration2_1 CV Tests FFN'
       when experiment_set_name like 'Iteration2_1 No SAE Tests FFN%' then 'Iteration2_1 No SAE Tests FFN'
       when experiment_set_name like 'Iteration2_1 Linear Tests FFN%' then 'Iteration2_1 Linear Tests FFN'
       when experiment_set_name like 'Iteration2_1 Tests FFN%' then 'Iteration2_1 Tests FFN'
       when experiment_set_name like 'Test Iteration2 Tests FFN%' then 'Test Iteration2 Tests FFN'

       when experiment_set_name like 'Iteration2 LTD SAE%' then 'Iteration2 LTD SAE'
       when experiment_set_name like 'Iteration2 SAE%' then 'Iteration2 SAE'
       when experiment_set_name like 'Denoising 2%' then 'Denoising 2'
       when experiment_set_name like 'Denoising 1%' then 'Denoising 1'
       when experiment_set_name like 'Denoising tests%' then 'Denoising tests'
       when experiment_set_name like 'Mape tests%' then 'Mape tests'
       when experiment_set_name like 'PT 3%' then 'Pretraining 3'
       when experiment_set_name like 'PT 2%' then 'Pretraining 2'

       when experiment_set_name like 'Linear Tests 2 Std%' then 'Linear Tests 2 Std'
       when experiment_set_name like 'Linear Tests 1%' then 'Linear Tests 1'
       when experiment_set_name like 'Linear Tests 2%' then 'Linear Tests 2'
       when experiment_set_name like 'Linear Tests 3%' then 'Linear Tests 3'

       when experiment_set_name like 'Iteration3 SAE LeakyRelu v2%' then 'Iteration3 SAE LeakyRelu v2'
       when experiment_set_name like 'Iteration3 SAE LeakyRelu Implementation2v1%' then 'Iteration3 SAE LeakyRelu Implementation2v1'
       when experiment_set_name like 'Iteration3 FFN Leaky ReLU Tests v2%' then 'Iteration3 FFN Leaky ReLU Tests v2'
       when experiment_set_name like 'Iteration3 FFN Validation Set Tests 2%' then 'Iteration3 FFN Validation Set Tests 2'
       when experiment_set_name like 'Iteration3 FFN Validation Set Tests%' then 'Iteration3 FFN Validation Set Tests'
       when experiment_set_name like 'Iteration3 FFN Max Epoch Tests%' then 'Iteration3 FFN Max Epoch Tests'
       when experiment_set_name like 'Iteration3_2 SAE LeakyRelu vs Relu%' then 'Iteration3_2 SAE LeakyRelu vs Relu'
       when experiment_set_name like 'Iteration3_2 FFN LeakyRelu vs Relu%' then 'Iteration3_2 FFN LeakyRelu vs Relu'
       when experiment_set_name like 'Iteration3_2 FFN LeakyRelu New Validation Set Test%' then 'Iteration3_2 FFN LeakyRelu New Validation Set Test'
       when experiment_set_name like 'Iteration3_2 Denoising On-Off Tests Real Data%' then 'Iteration3_2 Denoising On-Off Tests Real Data'
       when experiment_set_name like 'Iteration3_2 Regularization Tests Real Data%' then 'Iteration3_2 Regularization Tests Real Data'
       when experiment_set_name like 'Iteration3_2 SAE Regularization Tests Synthetic Data%' then 'Iteration3_2 SAE Regularization Tests Synthetic Data'
       when experiment_set_name like 'Iteration3_2 FFN Regularization Tests Synthetic Data %' then 'Iteration3_2 FFN Regularization Tests Synthetic Data'
       when experiment_set_name like 'Iteration3_2 SAE Real Data Regularization Redo%' then 'Iteration3_2 SAE Real Data Regularization Redo'
       when experiment_set_name like 'Iteration3_2 SAE Synthetic Data Regularization Redo%' then 'Iteration3_2 SAE Synthetic Data Regularization Redo'
       when experiment_set_name like 'Iteration3_2 SAE 2 Asset Tests%' then 'Iteration3_2 SAE 2 Asset Tests'
       when experiment_set_name like 'Iteration3_2 SAE 2 Asset Same Mean%' then 'Iteration3_2 SAE 2 Asset Same Mean'
       when experiment_set_name like 'Iteration3_2 FFN 2 Asset Six Combo%' then 'Iteration3_2 FFN 2 Asset Six Combo'
       when experiment_set_name like 'Iteration3_2 SAE 2 Asset Six Combo%' then 'Iteration3_2 SAE 2 Asset Six Combo'
       when experiment_set_name like 'Iteration3_2 FFN 3 Asset Combo%' then 'Iteration3_2 FFN 3 Asset Combo'
       when experiment_set_name like 'Iteration3_2 SAE 3 Asset Combo%' then 'Iteration3_2 SAE 3 Asset Combo'
       when experiment_set_name like 'Iteration3_2 FFN 4 Asset Combo%' then 'Iteration3_2 FFN 4 Asset Combo'
       when experiment_set_name like 'Iteration3_2 SAE 4 Asset Combo%' then 'Iteration3_2 SAE 4 Asset Combo'
       when experiment_set_name like 'Iteration3_3 SAE Redo 2 Asset%' then 'Iteration3_3 SAE Redo 2 Asset'
       when experiment_set_name like 'Iteration3_3 FFN 2 Asset Combo%' then 'Iteration3_3 FFN 2 Asset Combo'
       when experiment_set_name like 'Iteration3_4 SAE Redo 2 Asset%' then 'Iteration3_4 SAE Redo 2 Asset'
       when experiment_set_name like 'Iteration3_4 FFN 2 Asset Combo%' then 'Iteration3_4 FFN 2 Asset Combo'
       when experiment_set_name like 'Iteration3_4 SAE Redo 3 Asset%' then 'Iteration3_4 SAE Redo 3 Asset'
       when experiment_set_name like 'Iteration3_4 FFN 3 Asset Combo%' then 'Iteration3_4 FFN 3 Asset Combo'
       when experiment_set_name like 'Iteration3_4 SAE Redo 4 Asset%' then 'Iteration3_4 SAE Redo 4 Asset'
       when experiment_set_name like 'Iteration3_5 SAE Redo 4 Asset%' then 'Iteration3_5 SAE Redo 4 Asset'
       when experiment_set_name like 'Iteration3_12 SAE 1 Asset%' then 'Iteration3_12 SAE 1 Asset'
       when experiment_set_name like 'Iteration3_12 FFN 1 Asset%' then 'Iteration3_12 FFN 1 Asset'
       when experiment_set_name like 'Iteration3_12 SAE 2 Asset%' then 'Iteration3_12 SAE 2 Asset'
       when experiment_set_name like 'Iteration3_12 FFN 2 Asset%' then 'Iteration3_12 FFN 2 Asset'
       when experiment_set_name like 'Iteration3_12 SAE 3 Asset%' then 'Iteration3_12 SAE 3 Asset'
       when experiment_set_name like 'Iteration3_12 FFN 3 Asset%' then 'Iteration3_12 FFN 3 Asset'
       when experiment_set_name like 'Iteration3_12 SAE 4 Asset%' then 'Iteration3_12 SAE 4 Asset'
       when experiment_set_name like 'Iteration3_12 FFN 4 Asset%' then 'Iteration3_12 FFN 4 Asset'
       when experiment_set_name like 'Iteration3_13 SAE 2 Asset%' then 'Iteration3_13 SAE 2 Asset'
       when experiment_set_name like 'Iteration3_13 FFN 2 Asset%' then 'Iteration3_13 FFN 2 Asset'
       when experiment_set_name like 'Iteration3_13 SAE 3 Asset%' then 'Iteration3_13 SAE 3 Asset'
       when experiment_set_name like 'Iteration3_13 FFN 3 Asset%' then 'Iteration3_13 FFN 3 Asset'
       when experiment_set_name like 'Iteration3_13 SAE 4 Asset%' then 'Iteration3_13 SAE 4 Asset'
       when experiment_set_name like 'Iteration3_13 FFN 4 Asset%' then 'Iteration3_13 FFN 4 Asset'
       when experiment_set_name like 'Iteration3_6 FFN 1 Asset%' then 'Iteration3_6 FFN 1 Asset'
       when experiment_set_name like 'Iteration3_13 FFN Validation Set Tests%' then 'Iteration3_13 FFN Validation Set Tests'
       when experiment_set_name like 'Iteration3_15 FFN Validation 3 Test%' then 'Iteration3_15 FFN Validation 3 Test'
       when experiment_set_name like 'Iteration3_15 SAE LR Real Data Test%' then 'Iteration3_15 SAE LR Real Data Test'
       when experiment_set_name like 'Iteration4_1 SAE Tests%' then 'Iteration4_1 SAE Tests'
       when experiment_set_name like 'Iteration4_2 FFN Tests%' then 'Iteration4_2 FFN Tests'
       when experiment_set_name like 'Iteration4_3 SAE Epoch Tests%' then 'Iteration4_3 SAE Epoch Tests'
       when experiment_set_name like 'Iteration5_1 SAE AGL Test%' then 'Iteration5_1 SAE AGL Test'
       when experiment_set_name like 'Iteration5_1 AGL FFN Tests%' then 'Iteration5_1 AGL FFN Tests'
       when experiment_set_name like 'Iteration5_2 AGL FFN Tests%' then 'Iteration5_2 AGL FFN Tests'
       when experiment_set_name like 'Iteration5_3 AGL Test FFN Tests%' then 'Iteration5_3 AGL Test FFN Tests'
       when experiment_set_name like 'Iteration5_4 AGL Test FFN Tests%' then 'Iteration5_4 AGL Test FFN Tests'

       when experiment_set_name like 'Iteration5_6 AGL Test FFN Tests%' then 'Iteration5_6 AGL Test FFN Tests'
       when experiment_set_name like 'Iteration5_7 AGL Test FFN Tests%' then 'Iteration5_7 AGL Test FFN Tests'

           else null end as esn
        from configuration_run
        group by esn
        having esn is not null
        order by maxid desc"

category_query = "select min(configuration_id) minid,
       max(configuration_id) maxid,
       count(distinct configuration_id) total_configs,
       case
       when experiment_set_name like 'Iteration1 SAE Actual10 Test%' then 'Iteration1 SAE Actual10 Test'
       when experiment_set_name like 'Iteration2 SAE Actual10 Test%' then 'Iteration2 SAE Actual10 Test'
       when experiment_set_name like 'Iteration3 SAE Actual10 Test%' then 'Iteration3 SAE Actual10 Test'
       when experiment_set_name like 'Iteration4 SAE Actual10 Test%' then 'Iteration4 SAE Actual10 Test'
       when experiment_set_name like 'Iteration5 SAE Actual10 Test%' then 'Iteration5 SAE Actual10 Test'
       when experiment_set_name like 'Iteration6 SAE Actual10 Test%' then 'Iteration6 SAE Actual10 Test'

       when experiment_set_name like 'Iteration2_1 FFN Actual10%' then 'Iteration FFN Actual10 Tests'
       when experiment_set_name like 'Iteration3_1 FFN Actual10%' then 'Iteration FFN Actual10 Tests'
       when experiment_set_name like 'Iteration4_1 FFN Actual10%' then 'Iteration FFN Actual10 Tests'
       when experiment_set_name like 'Iteration5_1 FFN Actual10%' then 'Iteration FFN Actual10 Tests'
       when experiment_set_name like 'Iteration6_1 No SAE FFN Actual10%' then 'Iteration No SAE FFN Actual10 Tests'
       when experiment_set_name like 'Iteration7_1 No SAE FFN Actual10%' then 'Iteration No SAE FFN Actual10 Tests'

       when experiment_set_name like 'Iteration13_1 Large FFN Actual10 %' then 'Iteration FFN Large Actual10 Tests'
       when experiment_set_name like 'Iteration14_1 Large FFN Actual10 %' then 'Iteration2 FFN Large Actual10 Tests'

       when experiment_set_name like 'Iteration15%' then 'Iteration3 FFN Large Actual10 Tests'

       else null end as esn
    from configuration_run
    group by esn
    having esn is not null
    order by maxid desc"

category_ids_test = RunQuery(category_query_test, true)
category_ids = RunQuery(category_query)

function SelectConfigIDs(setnames, testDatabase = false)

    category_idSet = testDatabase ? category_ids_test : category_ids

    ids = []
    for set in setnames
        minid = get(category_idSet[Array(category_idSet[:esn]) .== set, :minid][1])
        maxid = get(category_idSet[Array(category_idSet[:esn]) .== set, :maxid][1])
        for i in minid:maxid
            push!(ids, i)
        end
    end

    return ids
end

function PrintConfigTable(setnames, testDatabase)
    config_ids = SelectConfigIDs(setnames, testDatabase)
    ids = TransformConfigIDs(config_ids)

    l2lambda_clause = testDatabase ? "" : "ifnull(tp.l2_lambda,0),"

    query = "select  tp.category, sae_config_id, cr.seed_used,
            steps, deltas, process_splits, training_splits, prediction_steps, scaling_function,
            ('(' || layer_sizes || ')') layer_sizes ,
            ('(' || layer_activations || ')') layer_activations, initialization, encoding_activation, output_activation,
            tp.learning_rate, tp.minibatch_size, tp.max_epochs, tp.l1_lambda, $l2lambda_clause tp.min_learning_rate, tp.epoch_cycle_max, tp.is_denoising, tp.denoising_variance,
            ifnull(tp2.learning_rate, 0.0) ogd_learning_rate
            from configuration_run cr
            inner join dataset_config dc on dc.configuration_id = cr.configuration_id
            inner join network_parameters np on np.configuration_id = cr.configuration_id
            inner join training_parameters tp on tp.configuration_id = cr.configuration_id and tp.category in ('FFN', 'SAE')
            left join training_parameters tp2 on tp2.configuration_id = cr.configuration_id and tp2.category = 'FFN-OGD'
            where cr.configuration_id in ($ids)"

    configs = RunQuery(query, testDatabase)

    config_string = "\\begin{longtable}[H]{|p{0.3\\linewidth}|p{0.7\\linewidth}|} \\hline \\textbf{Parameter} &\\textbf{Values} \\\\\\hline"

    for n in names(configs)

        if string(typeof(unique(configs[:, n])[1])) != "Nullable{Any}"
            vals = mapreduce(v -> string(v, ";"), string, unique(Array(configs[:,n])))[1:(end-1)]

            #{AGL} & {Anglo American} & {Resources}  \\\hline
            config_string = string(config_string, "{", replace(string(n), "_", "\\_"), "} & {",
            ProcessLayerActivations(n, vals), "} \\\\\\hline \n")
        end
    end

    config_string = string(config_string, "{\\textbf{Total Samples}} & {\\textbf{", size(configs,1), "}} \\\\\\hline \n")
    config_string = string(config_string, "\\end{longtable}")

    println("\n\n")
    println(config_string)
end

function ProcessLayerActivations(val_name, vals)
    if string(val_name) == "layer_activations"

        vals = "(SigmoidActivation,SigmoidActivation);(ReluActivation,ReluActivation);(LinearActivation,LinearActivation);(SigmoidActivation,SigmoidActivation,SigmoidActivation);(ReluActivation,ReluActivation,ReluActivation);(LinearActivation,LinearActivation,LinearActivation);(SigmoidActivation,SigmoidActivation,SigmoidActivation,SigmoidActivation)"

        vals = replace(replace(replace(vals, "(", ""), ")", ""), ';', ',')

        activations = mapreduce(i-> string(i, ','), (x, y) -> string(x, y), ascii.(unique(split(vals, ','))))[1:(end-1)]

        return activations

    elseif string(val_name) == "layer_sizes"
        return replace(vals, ";", "; ")
    end

    return vals
end

function Results_8_1_Intro()
    OGD_MSE_vs_PL()
end

function Results_8_2_PrimaryDeterminants()

    jsedata = ReadJSETop40Data()
    dataset = jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]
    delta_values = ([1,5,20], [5,20,60], [10,20,60])
    actualDist = GenerateDeltaDistribution(delta_values, (), 5000, dataset, 75, "Actual", "ActualMSE")

    sae_setnames = ["Iteration1 SAE Actual10 Test", "Iteration2 SAE Actual10 Test", "Iteration3 SAE Actual10 Test",
                "Iteration4 SAE Actual10 Test", "Iteration5 SAE Actual10 Test", "Iteration6 SAE Actual10 Test"]
    sae_config_ids = SelectConfigIDs(sae_setnames) #config12
    PrintConfigTable(sae_setnames, false)

    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13
    PrintConfigTable(ffn_setnames, false)

    MSE_Deltas(sae_config_ids, "Actual  ", "ActualMSE", false)
    PL_DataDeltas(ffn_config_ids, "Actual", "ActualPL")
    PL_OGD_LearningRate(ffn_config_ids, "Actual", "ActualPL", false)
    OOS_PL_OGDLR_Deltas(ffn_config_ids, 30, "IS Actual", "ActualPL", false, median, false)
    IS_PL_Encoding(ffn_config_ids, 30, "IS Actual", "ActualPL", false, median)
    MSE_Min_EncodingSize_Activation(sae_config_ids, "encoding", "Actual ", "ActualMSE", false, median)

    PL_SAE_Encoding_Size(ffn_config_ids, 30, "Actual", "ActualPL", false, nothing)
    #PL_SAE_Encoding_Size(config_ids, 30, "0.005 Actual", "ActualPL", false, 0.005)
    #PL_SAE_Encoding_Size(config_ids, 30, "0.01 Actual", "ActualPL", false, 0.01)
    #PL_SAE_Encoding_Size(config_ids, 30, "0.05 Actual", "ActualPL", false, 0.05)
    #PL_SAE_Encoding_Size(config_ids, 30, "0.1 Actual", "ActualPL", false, 0.1)
    PL_SAE_Encoding_SizeLines(ffn_config_ids, 30, "All Actual", "ActualPL", false, nothing, median)
    OOS_PL_OGDLR_Delta_Encoding(ffn_config_ids, 30, "OOS Actual", "ActualPL", false, median, 5)
    OOS_PL_OGDLR_Delta_Encoding(ffn_config_ids, 30, "OOS Actual", "ActualPL", false, median, 10)
    OOS_PL_OGDLR_Delta_Encoding(ffn_config_ids, 30, "OOS Actual", "ActualPL", false, median, 15)
    OOS_PL_OGDLR_Delta_Encoding(ffn_config_ids, 30, "OOS Actual", "ActualPL", false, median, 20)
    OOS_PL_OGDLR_Delta_Encoding(ffn_config_ids, 30, "OOS Actual", "ActualPL", false, median, 25)
end

function Results_8_3_HistoricalData()

    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13
    large_ffn_names = ["Iteration2 FFN Large Actual10 Tests"]
    large_ffn_config_ids = SelectConfigIDs(large_ffn_names)
    PrintConfigTable(large_ffn_names, false)

    PL_MaxEpochs(ffn_config_ids, "Actual ", "ActualPL", false)
    PL_ValidationSplit(large_ffn_config_ids, "Actual ", "ActualPL", false)
end

function Results_8_4_WeightInit()

    setnames = ["Pretraining 3"]#, "Pretraining 3"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    MSE_Pretraining(config_ids, "Actual Sigmoid PT", "ActualMSE", true)


    setnames = ["Iteration4_1 SAE Tests"] #3266 configurations
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    MSE_Init(config_ids, nothing, "Synthetic 10 ", "SyntheticMSE", true)

    setnames = ["Iteration5_1 SAE AGL Test"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    MSE_Init(config_ids, nothing, "AGL all ", "ActualMSE", true)

    setnames = ["Iteration4_2 FFN Tests"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    PL_Init(config_ids, "Synthetic 10 ", "SyntheticPL", true)

    setnames = ["Iteration5_1 AGL FFN Tests"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    PL_Init(config_ids, "AGL All ", "ActualPL", true)
end

function Results_8_5_SyntheticData()

    ##MSE#######################################################################
    #synthetic std: [1,5,20] = 0.1452, [5,20,60]=0.1522, [10,20,60] = 0.154
    jsedata = ReadJSETop40Data()
    dataset = jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]

    delta_values = ([1,5,20], [5,20,60], [10,20,60])
    var_pairs = ((0.9, 0.5),(0.7, 0.2),(0.05, 0.4),(0.05, 0.5),(0.04, 0.1),(0.02, 0.15),(0.01, 0.05),(-0.8, 0.55),(-0.4, 0.15),(-0.1, 0.2))

    synthDist = GenerateDeltaDistribution(delta_values, var_pairs, 5000, nothing, 75, "Test", "SyntheticMSE")


    sae_setnames_test = ["Iteration4_1 SAE Tests"] #config 7
    sae_config_ids_test = SelectConfigIDs(sae_setnames_test, true)

    MSE_Deltas(sae_config_ids_test, "TestSet ", "SyntheticMSE", true)

    sae_setnames_test = ["Iteration4_2 FFN Tests"] #config9
    sae_config_ids_test = SelectConfigIDs(sae_setnames_test, true)
    PL_DataDeltas(sae_config_ids_test, "Test", "SyntheticPL", true)

    setnames = ["Iteration4_2 FFN Tests"]
    config_ids = SelectConfigIDs(setnames, true)
    PL_SAE_Encoding_SizeLines(config_ids, 30, "All Synthetic", "SyntheticPL", true, nothing, median)
end

function Results_8_6_Complexity()

    setnames = ["Linear Tests 1", "Linear Tests 2 Std"]
    config_ids = SelectConfigIDs(setnames, true) #1050 samples
    PrintConfigTable(setnames, true)
    MSE_Scaling_Filters(config_ids, true, 1000, false, nothing, "Scaling ", "ActualMSE", true)

    setnames = ["Iteration2_1 Tests FFN", "Iteration2_1 Linear Tests FFN"]#, "Iteration2_1 MAPE Tests FFN"]
    config_ids = SelectConfigIDs(setnames, true)
    #PrintConfig(setnames, true)
    PL_Scaling(config_ids, "Linear ", "SyntheticPL", true)

    sae_setnames = ["Iteration1 SAE Actual10 Test", "Iteration2 SAE Actual10 Test", "Iteration3 SAE Actual10 Test", "Iteration4 SAE Actual10 Test", "Iteration5 SAE Actual10 Test", "Iteration6 SAE Actual10 Test"]
    sae_config_ids = SelectConfigIDs(sae_setnames) #config12
    MSE_Reg_EncodingLines(sae_config_ids, "Actual", "ActualMSE", false, median)

    MSE_Lambda1(sae_config_ids, "Actual", "ActualMSE", false)#config12

    gaussian_denoising_sets = ["Denoising 2" , "Denoising 1"]#, "Denoising tests"]
    gaussian_sae_ids = SelectConfigIDs(gaussian_denoising_sets, true)
    PrintConfigTable(gaussian_denoising_sets, true)
    MSE_Denoising(gaussian_sae_ids, "Actual Gaussian", "ActualMSE", true)

    masking_sets = ["Iteration3_2 Denoising On-Off Tests Real Data"]
    maskig_ids = SelectConfigIDs(masking_sets, true)
    PrintConfigTable(masking_sets, true)
    MSE_Denoising(maskig_ids, "Actual Masking ", "ActualMSE", true)



    synth_sae_setnames = ["Iteration4_1 SAE Tests"] #config 7
    synth_sae_ids = SelectConfigIDs(synth_sae_setnames, true)

    sae_setnames = ["Iteration1 SAE Actual10 Test", "Iteration2 SAE Actual10 Test", "Iteration3 SAE Actual10 Test", "Iteration4 SAE Actual10 Test", "Iteration5 SAE Actual10 Test", "Iteration6 SAE Actual10 Test"]
    sae_config_ids = SelectConfigIDs(sae_setnames) #config12
    MSE_LearningRate_MaxMin(sae_config_ids, nothing, "Actual", "ActualMSE", false) #config12
    MSE_LearningRate_MaxMin(synth_sae_ids, nothing, "Synth", "SyntheticMSE", true) #config 7

    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests", "Iteration2 FFN Large Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13
    PrintConfigTable(ffn_setnames, false)

    PL_EpochCycle(ffn_config_ids, "IS Actual", "ActualPL", false, true)
    PL_EpochCycle(ffn_config_ids, "OOS Actual", "ActualPL", false, false)

    PL_L1Reg(ffn_config_ids, "Actual OOS", "ActualPL", false, false)#config13
    PL_L1Reg(ffn_config_ids, "Actual IS", "ActualPL", false, true)#config13

    PL_Denoising(ffn_config_ids, "OOS Actual", "ActualPL", false, false)#config13
    PL_Denoising(ffn_config_ids, "IS Actual", "ActualPL", false, true)#config13
end

function Results_8_7_Network()

    sae_setnames = ["Iteration1 SAE Actual10 Test", "Iteration2 SAE Actual10 Test", "Iteration3 SAE Actual10 Test",
                    "Iteration4 SAE Actual10 Test", "Iteration5 SAE Actual10 Test", "Iteration6 SAE Actual10 Test"]
    sae_config_ids = SelectConfigIDs(sae_setnames) #config12
    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests", "Iteration2 FFN Large Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13
    PrintConfigTable(ffn_setnames, false)

    MSE_LayerSizesLines(sae_config_ids, "Actual ", "ActualMSE", false) #config12
    PL_NetworkSizeLines(ffn_config_ids, "Actual ", "ActualPL", false) #config13

    setnames = ["Linear Tests 1", "Linear Tests 2 Std"]
    config_ids = SelectConfigIDs(setnames, true) #1050 samples
    #PrintConfig(setnames, true)
    MSE_Output_Activation(config_ids, false, 0.1, false, nothing, "Scaling ", "ActualMSE", true)

    setnames = ["Linear Tests 1"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)

    MSE_Min_EncodingSize_Activation(config_ids, "encoding", "Encoding ", "ActualMSE", true)
    MSE_Min_EncodingSize_Activation(config_ids, "hidden", "Hidden ", "ActualMSE", true)

    setnames = ["Iteration2_1 Tests FFN", "Iteration2_1 Linear Tests FFN","Iteration2_1 Smaller Linear Tests"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    PL_Activations(config_ids, "Linear Small ", "SyntheticPL", true)
end

function Results_8_8_MMS()
    jsedata = ReadJSETop40Data()
    dataset = jsedata[:, [:AGL,:BIL,:IMP,:FSR,:SBK,:REM,:INP,:SNH,:MTN,:DDT]]


    #AllProfitsPDF(dataset)

    #AllProfitsPDF(dataset, true)

    SharpeRatiosPDF()

    PlotConfusion()
end

function Results_8_9_PBO()
    PlotCombinationSizes()

    PlotPBOBySplits()

    #ExperimentCSCVProcess(28880:53000, splits)
    #ExperimentCSCVProcess(50393:53000, splits)
end

function Result_8_10_DSR()
    ClusterOGDMSEPlot()

    ClusterDistributionPlot()
end

function Appendix()

    setnames = ["Iteration4_1 SAE Tests"]
    config_ids = SelectConfigIDs(setnames, true)
    MSE_Min_EncodingSize_Activation(config_ids, "encoding", "Synthetic ", "SyntheticMSE", true, median)


    setnames = ["Iteration1 SAE Actual10 Test", "Iteration2 SAE Actual10 Test", "Iteration3 SAE Actual10 Test", "Iteration4 SAE Actual10 Test",
                "Iteration5 SAE Actual10 Test", "Iteration6 SAE Actual10 Test"]
    sae_config_ids = SelectConfigIDs(setnames, false)
    MSE_Init(sae_config_ids, nothing, "Actual10 All ", "ActualMSE", false)

    MSE_EpochCycle(sae_config_ids, "Actual", "ActualMSE",false)

    synth_sae_setnames = ["Iteration4_1 SAE Tests"] #config 7
    synth_sae_ids = SelectConfigIDs(synth_sae_setnames, true)
    MSE_EpochCycle(synth_sae_ids, "Synth", "SyntheticMSE",true)

    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13

    synth_ffn_setnames_test = ["Iteration4_2 FFN Tests"] #config9
    synth_ffn_ids = SelectConfigIDs(synth_ffn_setnames_test, true)
    PL_LearningRates_MaxMin(synth_ffn_ids, "Synth", "SyntheticPL", true) #config9
    PL_LearningRates_MaxMin(ffn_config_ids, "OOS Actual", "ActualPL", false, false) #config13

    MSE_LayerSizes(sae_config_ids, nothing, "Actual ", "ActualMSE") #config12
    MSE_LayerSizes(synth_sae_ids, nothing, "Synth ", "SyntheticMSE", true)

    ffn_setnames = ["Iteration FFN Actual10 Tests", "Iteration No SAE FFN Actual10 Tests", "Iteration2 FFN Large Actual10 Tests"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames) #config13

    PL_NetworkSize(ffn_config_ids, "Actual ", "ActualPL", false) #config13
    PL_NetworkSize(synth_ffn_ids, "Synth ", "SyntheticPL", true)

    PL_OGD_LearningRate(synth_ffn_ids, "Synth", "SyntheticPL", true)
    PL_L1Reg(synth_ffn_ids, "Synth", "SyntheticPL", true)

    setnames = ["Iteration3_2 SAE LeakyRelu vs Relu"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    MSE_ActivationsEncodingSizes(config_ids, nothing, "Leaky Relu v Relu ", "SyntheticMSE", true)

    setnames = ["Iteration3_2 FFN LeakyRelu vs Relu"]
    config_ids = SelectConfigIDs(setnames, true)
    PrintConfigTable(setnames, true)
    PL_Activations(config_ids, "Leaky Relu v Relu ", "SyntheticPL", true)
end

Results_8_1_Intro()
Results_8_2_PrimaryDeterminants()
Results_8_3_HistoricalData()
Results_8_4_WeightInit()
Results_8_5_SyntheticData()
Results_8_6_Complexity()
Results_8_7_Network()
Results_8_8_MMS()
Results_8_9_PBO()
Result_8_10_DSR()
Appendix()
