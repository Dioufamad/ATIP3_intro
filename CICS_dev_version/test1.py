tag_task_type = "Classif" ##tag caught
classif_seeds = list(range(1))
classif_msn = 3
classif_msn_by_class = 0
classif_thres = 0.5
classif_list_cat_fts = ["SNV", "CNA", "SNVwCNA","CNAwSNV","SNVwCNAwGEXA","SNVwGEXAwCNA","CNAwSNVwGEXA","CNAwGEXAwSNV","GEXAwSNVwCNA","GEXAwCNAwSNV"]
classif_omc_search_type = "OMClight"
models_compared = [classif_omc_search_type, "Allfts"] # for the classification operations, a list of the models compared (2 at the moment) # for testing : do nothing it is done with previously line; just use this line

data_profiles_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/SLATE/SLATE_dev_version/slate_data/datasets_to_process_folder/real_val_prof_test" # the profiles data
data_drugs_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/SLATE/SLATE_dev_version/slate_data/table_of_treatments_details" # the drugs data
classif_profiles_folder = data_profiles_path
classif_drugs_folder = data_drugs_path
classif_Resp_col_name = "BestResCategory"
classif_Samples_col_name = "Model"
classif_reduc_data_decision = "yes"
classif_reduc_data_sn = 10
classif_CV_folds_number = 10
classif_chosen_pw = "par"

import locale
import numpy as np # linear algebra and exploit arrays faster and easier computations
import sys # to make all stdout display go to a log file (our .o in the results batch of files)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from slate_engines.data_engine1_mgmt import data_mgmt_1,data_mgmt_2,data_mgmt_6,unifier_creator,data_loadout_right_corresponder,data_mgmt_5,feature_values_type_caracterisation,reduction_of_dataset_for_testing_purpose
from slate_engines.data_engine2_allocation import add_entry_in_dict,il1_multiprocessing_handler,il1_sequential_processing_handler,stratKfolds_making
from slate_engines.learning_algs_engine import classifier_introduction,classifier_model_training,classifier_model_prediction,classifier_as_Keras_DNN_intro_train_pred,classifier_as_SVM_intro_train_pred # all for RF ML alg choice using tag_name, training and prediction
from slate_engines.learning_algs_engine import prediction_calling,raw_predictions_pusher,called_predictions_pusher # for predictions treatments
# from slate_engines.Classif_algs_engine import ## to copy and use to separate here the others functions for any another classifier
from slate_engines.metrics_engine import calculate_mcc_w_storing,pd_ml_classif_report_on_cm_binary
from slate_engines.watcher_engine import timer_started,duration_from,bcolors,roc_curve_updater_after_one_iteration_of_the_mdl,roc_curve_finisher_after_all_iterations_of_the_mdl,average_roc_curve_init,average_roc_curve_finisher,df_of_results_for_metrics_one_mdl_creator,df_of_results_for_metrics_all_mdls_creator,df_of_results_for_FS_one_mdl_creator
from slate_engines.fs_engine import ranker_by_pval_v2,OMC_founder_in_dict_MCs_MCCs
from multiprocessing import cpu_count, current_process # 1st is for the num_cores acquisition, 2nd is for telling a process who carried-out a job
import matplotlib.pyplot as plt # used to make plots # for roc curves
from uncertainties import ufloat # to write the mean auc accross seeds with the incertainty in one cell and correctly
# from functools import reduce # reduce fonctions to obtain persistent list in list of lists # only in python3 has it been here #unused in lmatest versions of the tool
import argparse #to manage the arguments of the script
from pathlib import Path # to manage paths as into arguments
from sklearn.preprocessing import LabelEncoder

extracted_data1 = data_mgmt_1(classif_profiles_folder, classif_drugs_folder, classif_Resp_col_name,classif_Samples_col_name)

data_loadout_left_all = extracted_data1[0]
data_loadout_right = extracted_data1[1]
data_loadouts_origin_files = extracted_data1[2]
set_of_all_condittions_found = extracted_data1[3]
set_of_all_TreatmentIDs_found = extracted_data1[4]
dict_TreatmentID_TreatmentName = extracted_data1[5]
set_of_all_profiles_found = extracted_data1[6]
Resp_col_name = extracted_data1[7]
Samples_col_name1 = extracted_data1[8]
Samples_col_name2 = extracted_data1[9]
Unifier_key_col_name = extracted_data1[10]
Condittion_col_name = extracted_data1[11]
TreatmentID_col_name = extracted_data1[12]
Layer_probed_col_name = extracted_data1[13]

ctype = "BRCA"
cframe = data_loadout_left_all.loc[data_loadout_left_all[Condittion_col_name] == ctype] #take only rows of drug_response that are that ctype

drug = 'Treatment17'
tag_drugname = dict_TreatmentID_TreatmentName[drug] # collect a tag for the drug name for the result files
drugframe = cframe.loc[cframe[TreatmentID_col_name] == drug]  # take only rows of drug_response that are that drug

featuretype = 'GEX'
feature_val_type = feature_values_type_caracterisation(featuretype,classif_list_cat_fts) #!! use this value in functions to know how to treat datasets values #featuretype and classif_list_cat_fts can go

unifier = unifier_creator(ctype, drug, featuretype)
featureframe = drugframe.loc[drugframe[Layer_probed_col_name] == featuretype]
corresponding_data_loadout_right = data_loadout_right_corresponder(featuretype,data_loadout_right)
dframe_and_profile_data_archived_and_index_starting_fts_cols = data_mgmt_2(unifier, featureframe, corresponding_data_loadout_right, Samples_col_name1, Samples_col_name2, Unifier_key_col_name)
dframe = dframe_and_profile_data_archived_and_index_starting_fts_cols[0]
profile_data_archived = dframe_and_profile_data_archived_and_index_starting_fts_cols[1] # for keeping track
index_starting_fts_cols = dframe_and_profile_data_archived_and_index_starting_fts_cols[2]

dframe_old = dframe
dframe = reduction_of_dataset_for_testing_purpose(dframe,Resp_col_name,classif_reduc_data_sn)
print("Features values formatting : categoricals as booleans and reals as floats...")
dframe = data_mgmt_5(dframe, index_starting_fts_cols, feature_val_type)
print("Sorting the dataframe entries following response column values and remaking a new index...")
dframe = data_mgmt_6(dframe, Resp_col_name)