workspace()
push!(LOAD_PATH, "/Users/joeldacosta/Masters/Code Libraries/")

using ExperimentGraphs2
using DatabaseOps


#Iteration 1 <= 3703
#Iteration 2

function SelectConfigIDs(setnames)
    ids = []
    for set in setnames
        minid = get(category_ids[Array(category_ids[:esn]) .== set, :minid][1])
        maxid = get(category_ids[Array(category_ids[:esn]) .== set, :maxid][1])
        for i in minid:maxid
            push!(ids, i)
        end
    end

    return ids
end


category_query = "select min(configuration_id) minid,
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
       when experiment_set_name like 'Linear Tests%' then 'Linear Tests'
       when experiment_set_name like 'Linear Tests%' then 'Linear Tests'
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

       else null end as esn
    from configuration_run
    group by esn
    having esn is not null
    order by maxid desc"
category_ids = RunQuery(category_query)




## Redone/ Fine
function It3_SAE_L1RegSyntheticlData()
    setnames = ["Iteration3_2 SAE Regularization Tests Synthetic Data"]
    config_ids = SelectConfigIDs(setnames)
    SAE_Lambda1_MinTest_BxMSE(config_ids)
end

function It3_L1RegRealData()
    setnames = ["Iteration3_2 Regularization Tests Real Data"]
    config_ids = SelectConfigIDs(setnames)
    SAE_Lambda1_MinTest_BxMSE(config_ids)
end

function It3_DenoisingOnOff()
    setnames = ["Iteration3_2 Denoising On-Off Tests Real Data"]
    config_ids = SelectConfigIDs(setnames)

    Denoising_BxMSE(config_ids)
end

function It3_ValidationSetPercentages()
    setnames = ["Iteration3_2 FFN LeakyRelu New Validation Set Test"]
    config_ids = SelectConfigIDs(setnames)

    OGD_ValidationSet_Profits_bx(config_ids)
end

function It3_SAE_OGD_LearningRate_Activations()
    sae_setnames = ["Iteration3_2 SAE LeakyRelu vs Relu"]

    sae_config_ids = SelectConfigIDs(sae_setnames)
    #SAE MSE
    SAE_MaxLR_MinTest_BxMSE(sae_config_ids)

    ffn_setnames = ["Iteration3_2 FFN LeakyRelu vs Relu"]
    ffn_config_ids = SelectConfigIDs(ffn_setnames)
    #FFN LR Profits
    FFN_LR_BxProfit(ffn_config_ids)
    #OGD LR MSE
    OGD_LR_AvgTrain_BxMSE(ffn_config_ids)
    #OGD LR Profits
    OGD_LR_BxProfit(ffn_config_ids)
end

function It3_FFN_Mape_vs_MSE()
    setnames = ["Iteration2_1 Tests FFN", "Iteration2_1 Linear Tests FFN", "Iteration2_1 MAPE Tests FFN"]
    config_ids = SelectConfigIDs(setnames)
    OGD_SAE_Selection_Profits_bx(config_ids)
end

function It3_FFN_SmallerNetwork_Linear_vs_Relu()
    setnames = ["Iteration2_1 Smaller Linear Tests"]
    config_ids = SelectConfigIDs(setnames)
    OGD_NetworkSizeOutputActivation_Profits_Bx(config_ids)
end

function It3_FFN_LeakyRelu_vs_Relu()
    setnames = ["Iteration3_2 FFN LeakyRelu vs Relu"]
    config_ids = SelectConfigIDs(setnames)
    OGD_Activations_Profits_Bx(config_ids)
end

function It3_SAE_LeakyRelu_vs_Relu()
    setnames = ["Iteration3_2 SAE LeakyRelu vs Relu"]
    config_ids = SelectConfigIDs(setnames)
    SAE_ActivationsNetworkSizes_MinMSE(config_ids)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids, 3)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids, 6)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids, 9)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids, 12)
    SAE_ActivationsEncodingSizes_MinMSE(config_ids, 15)
end


#Configs Needed

#Written Iteration 1
#None

#Written Iteration 2
#Fig 2 & 3; Pre Training Effects  (SAE MSE on 2 stocks)
#Fig 4 & 5: SAE Standardizing &  ReLU Output; Effects of Linear Activations  (SAE MSE on 10 stocks)
#Fig 9: Denoising SAEs (SAE MSE on 10 stocks)
#Fig 10: Effects of Limited Scaling (on Synthetic P&L)

#Carry Into Written Iteration 3:
# SAE Encoding Size Effects on SAE MSE
# SAE Encoding Size Effects on FFN P&L, including No SAE
# SAE Performance by network Size (SAE MSE on 10 stocks)
