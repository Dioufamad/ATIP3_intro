# =============================== CICS - Case Implicated Candidates Search ========================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>IMPORTS FOR CLASSIFICATION TASK<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import locale
import numpy as np # linear algebra and exploit arrays faster and easier computations
import sys # to make all stdout display go to a log file (our .o in the results batch of files)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from slate_engines.data_engine1_mgmt import data_mgmt_1,data_mgmt_2,data_mgmt_6,unifier_creator,data_loadout_right_corresponder,data_mgmt_5,feature_values_type_caracterisation,reduction_of_dataset_for_testing_purpose
from slate_engines.data_engine2_allocation import add_entry_in_dict,il1_multiprocessing_handler,il1_sequential_processing_handler,stratKfolds_making
from slate_engines.Classif_algs_engine import classifier_introduction,classifier_model_training,classifier_model_prediction,classifier_as_Keras_DNN_intro_train_pred,classifier_as_SVM_intro_train_pred # all for RF ML alg choice using tag_name, training and prediction
from slate_engines.Classif_algs_engine import prediction_calling,raw_predictions_pusher,called_predictions_pusher # for predictions treatments
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
from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>IMPORTS FOR REGRESSION TASK OR NOT NEEDED<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# from slate_engines.data_engine1_mgmt import data_mgmt_3
# import random
# from slate_engines.data_engine2_allocation import testset_indexes_spacer,testset_indexes_selector, trainingset_indexes_selector, set_creator_w_rows_index
# from slate_engines.Classif_algs_engine import classifier_introduction,classifier_model_training,classifier_model_prediction,classifier_introduction2,classifier_model_training2,classifier_model_prediction2,classifier_introduction2_dflt0 #1 XGBoost_C_1 #2 RF_dflt1 # RF intro dflt0
# from slate_engines.fs_engine import length_features_list,maximal_complexity_as_half_tr,list_of_complexities_ext,list_of_complexities_ltd,stratification1,feat_selection1
# from slate_engines.metrics_engine import r2,rmse,spearmanr_test,spearmanr_test_dec,r2_dec,rmse_dec,restriction_of_MCCs_wide,df_multiplier_in_rows
# from slate_engines.fs_engine import eliminate_non_variable_fts
# from random import shuffle
#-------------------------------END OF IMPORTS-----------------------------------------
#-----> Variables used accross the script for regression
# cv_standard = 5  #for K-fold cross-validation
# testset_size = 10 # size of the test set
# min_cons_corr = -2000 # minimum conserved correlation ie for the prediction to be conserved # usaually 0.25
# num_cores = 30 #for the Number of cores used in some classfiers like xgboost
#----->
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>ENVIRONNEMENT DEFINITIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # for setting the characters format
# add here a section to fixated shebang style
warnings.filterwarnings("ignore") #for the behaviour of the warnings alerts
globalstart = timer_started() # start a clock to get the time after all the analysis
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

### >>>>>>>>>>>>>>>>>>>>>>>>>>>VARIABLES INITALISATION 1/2 : ARGUMENTS MANAGEMENTS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
our_args_parser = argparse.ArgumentParser(prog='ClassHD',description="Welcome in the Classification benchmark on HD data.", epilog="Thank you and adress for contributions")
################### creating grouped args....
# group = our_args_parser.add_mutually_exclusive_group() ##!! customise and use later for what to show and what to retain
# group.add_argument("-v", "--verbose", action="store_true")
# group.add_argument("-q", "--quiet", action="store_true")
################## creating args....
#-----> The present version of the tool
our_args_parser.add_argument("-v",'--version', action='version', version='%(prog)s 2.0')
#-----> Name the analysis
our_args_parser.add_argument("-t","--Trial_number", type=str, help="(str) name given to the analysis that will be ran", required=True) # default action is action=store
#-----> Number of cores to use in case of multiprocessing (geared towards the loops)
max_cores_minus2 = (cpu_count() - 2) # leave 2 cores to work on the side # use 10 for tests
# max_cores = cpu_count() # full usage
our_args_parser.add_argument("-xproc","--Multiprocessing_cores", type=int, default=max_cores_minus2, help="(int) (default is max_cores_minus2) Number of cores to use in case of multiprocessing") # default action is action=store # required=True is taken off to be able to use default value
#-----> Super Learning Task(s) Type choice:
our_args_parser.add_argument("-sl","--SL_tasks_to_perform", choices=["Classif", "Regr", "Both"], nargs='+',help="(str choice) the types of supervised learning algorithms to performs",required=True) # required=True if not default='Classif'
#-----> ML algorithm(s) choices :
RF_marks = ["RF_Mark1Vpar","RF_Mark2Vpar","RF_Mark3Vpar","RF_Mark4Vpar"] # best = RF_Mark2Vpar
XGB_marks = ["XGB_Mark1Vpar","XGB_Mark2Vseq","XGB_Mark2Vpar"] # best = XGB_Mark2Vpar
# GBM_marks = ["GBM_Mark1Vpar"]
DNN_marks = ["DNN_Mark1Vseq","DNN_Mark2Vseq","DNN_Mark3Vseq","DNN_Mark4Vseq"] # best = DNN_Mark2Vseq
SVM_marks = ["SVM_Mark1Vpar","SVM_Mark2Vpar"] # best = SVM_Mark1Vpar for lk and SVM_Mark2Vpar for rbf
classif_algs_implemented = RF_marks+XGB_marks+DNN_marks+SVM_marks
our_args_parser.add_argument("-ca","--Classif_algs", choices=classif_algs_implemented, nargs='+', help="(str choice) the classification algorithm to use if classification has to be carried out") # required=True if not by default Classif # using action=append regroup all the entry(a list already) and add it to a list (so we get a list of a list) so better leave like this
regr_algs_implemented = ["RF_Mark1_par","RF_Mark2_par","RF_Mark3_par","RF_Mark4_par"]
our_args_parser.add_argument("-ra","--Regr_algs", choices=regr_algs_implemented, nargs='+', help="(str choice) the regression algorithm to use if regression has to be carried out")
#-----> Study(ies) to carry out choices :
classif_studies_to_carry_out = ["CAB","C-FSB","Both"]
our_args_parser.add_argument("-cs","--Classif_studies", choices=classif_studies_to_carry_out, nargs='+', help="(str choice) in classification, the studies to carry out (compare algorithms with metrics or feature selection based on composition)")
regr_studies_to_carry_out = ["RAB","R-FSB","Both"]
our_args_parser.add_argument("-rs","--Regr_studies", choices=regr_studies_to_carry_out, nargs='+', help="(str choice) in regression, the studies to carry out (compare algorithms with metrics or feature selection based on composition)")
#------> Random seeds list choices : used to built a list of int from 0 to "chosen int"-1 (ex : 6 -> [0,1,2,3,4,5] # use 1 or 2 for testing and 10 for real trials
our_args_parser.add_argument("-cla_seeds","--Classif_seeds_values", type=int, help="(int) the number of seeds to use for classification operations") # default action is action=store
our_args_parser.add_argument("-reg_seeds","--Regr_seeds_value", type=int, help="(int) the number of seeds to use for regression operations") # default action is action=store
#------> Default threshold value  when calling predictions after predictions probabilities obtained (default is 0.5)
our_args_parser.add_argument("-cla_pred_thres","--Classif_predsprobs_calling_threshold", type=float, default=0.5, help="(float) (default is 0.5) in classification operations, the default threshold value  when calling predictions after predictions probabilities obtained") # default action is action=store
our_args_parser.add_argument("-reg_pred_thres","--Regr_predsprobs_calling_threshold", type=float, default=0.5, help="(float) (default is 0.5) in regression operations, the default threshold value  when calling predictions after predictions probabilities obtained") # default action is action=store
### (Limits of data characteristics to allow the analysis)
#------> Defining a list of the feature types to consider categorical (others feature types are considered real or mixed)
default_list_profiles_known_as_categ = ["SNV", "CNA", "SNVwCNA","CNAwSNV","SNVwCNAwGEXA","SNVwGEXAwCNA","CNAwSNVwGEXA","CNAwGEXAwSNV","GEXAwSNVwCNA","GEXAwCNAwSNV"] ##! make a rule later to estimate if profile is categ or not
our_args_parser.add_argument("-list_catfts","--list_categ_fts", nargs='+', default=default_list_profiles_known_as_categ, help="(str choice) (default is [SNV, CNA]) a list of the feature types to consider categorical for formating there value True or False (others feature types are considered real and managed as floats") # required=True if not by default Classif
#------> Minimal_samples_number for classification following litterature # = 35 normally ; put 3 for testing; put 30 to be able to analyse any file because lowest samples # encoutered in PDX__data is 31
our_args_parser.add_argument("-cla_msn","--Classif_MSN", type=int, default=30, help="(int) (default is 35) the minimal samples number for a proper classification following litterature") # default action is action=store
our_args_parser.add_argument("-reg_msn","--Regr_MSN", type=int, default=30, help="(int) (default is 35) the minimal samples number for a proper regression following litterature") # default action is action=store
#------> Minimal_samples_number with the class as response for classification following litterature # normal is 5; put 0 for testing
our_args_parser.add_argument("-cla_msn_class","--Classif_MSN_by_class", type=int, default=0, help="(int) (default is 0) the minimal samples number with the class as response for a proper classification following litterature") # default action is action=store
our_args_parser.add_argument("-reg_msn_class","--Regr_MSN_by_class", type=int, default=0, help="(int) (default is 0) the minimal samples number with the class as response for a proper regression following litterature") # default action is action=store
#------> the type of OMC research to carry out
OMC_research_types_implemented = ["OMC", "OMClight"] # MC = optimal model complexity
our_args_parser.add_argument("-cla_omc_search","--Classif_OMC_search_type_to_perform", choices=OMC_research_types_implemented, type=str, default="OMClight", help="(str choice) (default is OMClight) in classification operations, the type of OMC research to carry out. Difference in the 2 types proposed is that Allfts complexity is also tested in the OMC")
our_args_parser.add_argument("-reg_omc_search","--Regr_OMC_search_type_to_perform", choices=OMC_research_types_implemented, type=str, default="OMClight", help="(str choice) (default is OMClight) in regression operations, the type of OMC research to carry out. Difference in the 2 types proposed is that Allfts complexity is also tested in the OMC")
#------> the type of processing way to execute for multiple jobs with the same algorithm (parallel or sequential)
pw__implemented = ["par", "seq"] # MC = optimal model complexity
our_args_parser.add_argument("-cla_pw","--Classif_processing_way", choices=pw__implemented, type=str, default="native", help="(str choice) (default is pw) in classification operations, the type of processing way to execute for multiple jobs with the same algorithm (parallel or sequential). The chosen processing way will be run instead if different from the one preferentiably implemented for chosen classifier")
our_args_parser.add_argument("-reg_pw","--Regr_processing_way", choices=pw__implemented, type=str, default="native", help="(str choice) (default is pw) in regression operations, the type of processing way to execute for multiple jobs with the same algorithm (parallel or sequential). The chosen processing way will be run instead if different from the one preferentiably implemented for chosen regressor")
#------> Paths to data sources
# Input data files are available in the "./PDX__data/" & "./inp/" directories. lets define their locations if not given as arguments.
# Any results is written in the "./outputs/" directory.
### default folders to use as arguments option if nothing if given by user
# data_profiles_path = "/home/diouf/ClassHD_work/actual_repo/ClassHD/CICS_dev_version/slate_data/datasets_to_process_folder/real_val_prof_test" # the profiles data #old
# data_drugs_path = "/home/diouf/ClassHD_work/actual_repo/ClassHD/CICS_dev_version/slate_data/table_of_treatments_details" # the drugs data # old

data_profiles_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/SLATE/SLATE_dev_version/slate_data/datasets_to_process_folder/real_val_prof_test" # the profiles data
data_drugs_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/SLATE/SLATE_dev_version/slate_data/table_of_treatments_details" # the drugs data

### Paths to capture from what is given by a user (previously was using type=lambda d: Path(d).absolute() changed to type=Path to force user to give full path of folders)
our_args_parser.add_argument("-cla_drugs_path","--Classif_drugs_folder", type=Path, default=data_drugs_path, help="(path) (default is PDX test data) for classification, path to the drugs data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
our_args_parser.add_argument("-cla_profiles_path","--Classif_profiles_folder", type=Path, default=data_profiles_path, help="(path) (default is PDX test data) for classification, path to the profiles data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
our_args_parser.add_argument("-reg_drugs_path","--Regr_drugs_folder", type=Path, default=data_drugs_path, help="(path) (default is PDX test data) for regression, path to the drugs data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
our_args_parser.add_argument("-reg_profiles_path","--Regr_profiles_folder", type=Path, default=data_profiles_path, help="(path) (default is PDX test data) for regression, path to the profiles data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
#------> The column names for the response column and the samples column in order to localise no matter their position if they are not already put in the positions required
our_args_parser.add_argument("-cla_resp_col","--Classif_Resp_col_name", type=str, default="BestResCategory", help="(str) (default is BestResCategory) the name of the Response values column. To be given if different from default") # default action is action=store
our_args_parser.add_argument("-cla_samples_col","--Classif_Samples_col_name", type=str, default="Model", help="(str) (default is Model) the name of the Samples names column. To be given if different from default")
our_args_parser.add_argument("-reg_resp_col","--Regr_Resp_col_name", type=str, default="BestResCategory", help="(str) (default is BestResCategory) the name of the Response values column. To be given if different from default") # default action is action=store
our_args_parser.add_argument("-reg_samples_col","--Regr_Samples_col_name", type=str, default="Model", help="(str) (default is Model) the name of the Samples names column. To be given if different from default")
#------> Reduction of dataset to analyse size for testing purpose
# choose if the reduction is done or not....
regarding_reduction_decisions_to_choose_from = ["yes","y","no","n"]
our_args_parser.add_argument("-cla_reduc_data","--Classif_reduce_dataset", choices=regarding_reduction_decisions_to_choose_from, type=str, default="no", help="(str choice) (default is no) in classification operations, wether or not to reduce dataset to analyse to a lower number of samples, for faster testing purpose")
our_args_parser.add_argument("-reg_reduc_data","--Regr_reduce_dataset", choices=regarding_reduction_decisions_to_choose_from, type=str, default="no", help="(str choice) (default is no) in regression operations, wether or not to reduce dataset to analyse to a lower number of samples, for faster testing purpose")
# ....and if it is done to what number of samples to reduce it to
our_args_parser.add_argument("-cla_reduc_data_sn","--Classif_reduced_dataset_samples_number", type=int, default=10, help="(int < 30) (default is 10) in classification operations, for faster testing purpose, reduction of dataset to analyse is made here by giving number of samples to use")
our_args_parser.add_argument("-reg_reduc_data_sn","--Regr_reduced_dataset_samples_number", type=int, default=10, help="(int < 30) (default is 10) in regression operations, for faster testing purpose, reduction of dataset to analyse is made here by giving number of samples to use")
#------> Cross-validation folds number choice
our_args_parser.add_argument("-cla_XCV","--Classif_cross_validation_folds_number", type=int, default=10, help="(2 <= int <= number of samples) (default is 10) in classification operations, the number of folds to make in the cross-validation")
our_args_parser.add_argument("-reg_XCV","--Regr_cross_validation_folds_number", type=int, default=10, help="(2 <= int <= number of samples) (default is 10) in regression operations, the number of folds to make in the cross-validation")
#------> Making a log file or not
decisions_of_making_log_file_to_choose_from = ["yes","y","no","n"]
our_args_parser.add_argument("-cla_log","--Classif_make_log", choices=decisions_of_making_log_file_to_choose_from, type=str, default="yes", help="(str choice) (default is no) in classification operations, wether or not to make a log file")
our_args_parser.add_argument("-reg_log","--Regr_make_log", choices=decisions_of_making_log_file_to_choose_from, type=str, default="yes", help="(str choice) (default is no) in regression operations, wether or not to make a log file")

############## parsing them....
our_args = our_args_parser.parse_args()
# for tests put here arguments values

###..checking the exceptions...
# if our_args.SL_tasks_to_perform is None:
# 	our_args_parser.error('At least, 1 supervised learning task has to be chosen, choosing from  "Classif", "Regr", "Both".') # dealt with a required=True
# if an sl_task is required and alg of same task has to be inputed
if (("Classif" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform)) and (our_args.Classif_algs is None) :
	our_args_parser.error('A classification is to be performed. Please give the classification algorithm(s) to test.')
if (("Regr" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform)) and (our_args.Regr_algs is None):
	our_args_parser.error('A regression is to be performed. Please give the classification algorithm(s) to test.')
# see at least if one alg has been given
if (our_args.Classif_algs is None) and (our_args.Regr_algs is None):
	our_args_parser.error('At least, 1 supervised learning algorithm has to be chosen, a classifier or a regressor.')
# see at least if one study has been required
if (our_args.Classif_studies is None) and (our_args.Regr_studies is None):
	our_args_parser.error('At least, 1 study has to be requested, a comparison of algorithms with metrics or a comparison of feature selections based on their respective compositions.')
# if an sl_task is detected, at least one study using it has to been required
if (our_args.Classif_algs is not None) and (our_args.Classif_studies is None) :
	our_args_parser.error('A classification alg is to be tested. Please give the study(ies) to perform.')
if (our_args.Regr_algs is not None) and (our_args.Regr_studies is None) :
	our_args_parser.error('A regression alg is to be tested. Please give the study(ies) to perform.')
# if a study is to be done, the seeds values for those run have to be given
if (our_args.Classif_studies is not None) and (our_args.Classif_seeds_values is None) :
	our_args_parser.error('At least one classification study is to be carried out. Please give the number of seeds for classification study(ies) to perform.')
if (our_args.Regr_studies is not None) and (our_args.Regr_seeds_value is None) :
	our_args_parser.error('At least one regression study is to be carried out. Please give the number of seeds for regression study(ies) to perform.')
# # if a study is to be done, the default threshold value  when calling predictions after predictions probabilities are obtained have to be given # handled by a default value
# if (our_args.Classif_studies is not None) and (our_args.Classif_predsprobs_calling_threshold is None):
# 	our_args_parser.error('At least one classification study is to be carried out. Please give the default threshold value when calling predictions after predictions probabilities are obtained.')
# if (our_args.Regr_studies is not None) and (our_args.Regr_predsprobs_calling_threshold is None):
# 	our_args_parser.error('At least one regression study is to be carried out. Please give the default threshold value when calling predictions after predictions probabilities are obtained.')
# if a study is to be done, the default threshold values have to be both in the interval [0:1]
if (our_args.Classif_predsprobs_calling_threshold < 0) | (our_args.Classif_predsprobs_calling_threshold > 1) :
	our_args_parser.error('any threshold value for probabilities has to be in the interval [0:1]. The threshold value for the classification is at fault here.')
if (our_args.Regr_predsprobs_calling_threshold < 0) | (our_args.Regr_predsprobs_calling_threshold > 1) :
	our_args_parser.error('any threshold value for probabilities has to be in the interval [0:1]. The threshold value for the regression is at fault here.')
# # if a study is to be done, a list of the feature types to consider categorical (others feature types are considered real or mixed) has to be given # handled by the use of a default list
# if ((our_args.Classif_studies is not None) | (our_args.Regr_studies is not None)) and (our_args.list_categ_fts is None):
# 	our_args_parser.error('At least one SL study is to be carried out. Please give a list of the feature types to consider categorical (others feature types are considered real or mixed).')
# # if a study is to be done, the minimal samples number to properly conduct the SL operation have to be given
# if (our_args.Classif_studies is not None) and (our_args.Classif_MSN is None):
# 	our_args_parser.error('At least one classification study is to be carried out. Please give the minimal samples number to properly conduct the classification study(ies) to perform.')
# if (our_args.Regr_studies is not None) and (our_args.Regr_MSN is None):
# 	our_args_parser.error('At least one regression study is to be carried out. Please give the minimal samples number to properly conduct the regression study(ies) to perform.')
# # if a study is to be done, the minimal samples number with the class as response to properly conduct the SL operation have to be given
# if (our_args.Classif_studies is not None) and (our_args.Classif_MSN_by_class is None):
# 	our_args_parser.error('At least one classification study is to be carried out. Please give the minimal samples number with the class as response to properly conduct the classification study(ies) to perform.')
# if (our_args.Regr_studies is not None) and (our_args.Regr_MSN_by_class is None):
# 	our_args_parser.error('At least one regression study is to be carried out. Please give the minimal samples number with the class as response to properly conduct the regression study(ies) to perform.')
# if a study is to be done, the paths to the drugs and the profiles data is to be given ##!! solved by the giving a default value as the known test PDX data
# if a study is to be done, the default reduction is no reduction.....
# if a reduction is chosen or not the default reduction value to 10 samples is present but only be used if reduction is chosen. can be modified through this exception related argument
# if a reduction is chosen , the value of number of samples kept has to be inferior to 30
if (our_args.Classif_reduce_dataset in ["yes","y"]) & ((our_args.Classif_reduced_dataset_samples_number < 7) | (our_args.Classif_reduced_dataset_samples_number > 30)) :
	our_args_parser.error('in classification operations, if a reduction of the dataset is chosen, the value of number of samples kept has to be in interval ]7:30[.')
if (our_args.Regr_reduce_dataset in ["yes", "y"]) & ((our_args.Regr_reduced_dataset_samples_number < 7) | (our_args.Classif_reduced_dataset_samples_number > 30)):
	our_args_parser.error('in regression operations, if a reduction of the dataset is chosen, the value of number of samples kept has to be in interval ]7:30[.')
# if the default 10XCV has to be changed, the int given has to be superior or equal to 2. NB : the 2nd condittion about the value not exceeding the number of samples is forced later
if our_args.Classif_cross_validation_folds_number < 2 :
	our_args_parser.error('in classification operations, if the default 10XCV has to be changed, the int given has to be superior or equal to 2.')
if our_args.Regr_cross_validation_folds_number < 2 :
	our_args_parser.error('in regression operations, if if the default 10XCV has to be changed, the int given has to be superior or equal to 2.')

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

### >>>>>>>>>>>>>>>>>>>>>>>>>>>VARIABLES INITALISATION 2/2 : FIXATED VARIABLES<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# not used but exploit this space for hard coding variables if necessary
basedir = str(Path()) #for setting the working directory
# NB : the output filename can have appointed tags to identify easier the analysis it reports on (done throughout the script and can be foung with commented "# tag caught")
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>DESCRIPTION OF THE ANALYSIS BEFORE IT STARTS
### Describing the analysis ordered with such arguments...
## (from here on now, tags will be taken in checkpoints throughout the script to use them in results descriptions (titles of figures, output filenames)
#Name of the trial that will be run
tag_num_trial = our_args.Trial_number # "Trial14" was the last # tag_num_trial = "Trial_test" for testing
tag_num_xproc = our_args.Multiprocessing_cores # capture # tag_num_xproc = 10 for testing or tag_num_xproc = 1

#>>>>>>>>>>>>>>>>>>>>>>>>>>>Redirecting stdout to .o file or not<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
classif_decision_make_log = our_args.Classif_make_log
if classif_decision_make_log in ["yes", "y"]:
	original_out = sys.stdout
	sys.stdout = open(basedir + "/" + "outputs" + "/" + "Output_following_" + tag_num_trial + ".o", 'w')
print('This is the file following the course of the run:')
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

###...and describe the analysis built
print("This analysis is named : " + tag_num_trial + " and is supported by " + str(tag_num_xproc) + " cores") #tag caught # tag caught
print("It will carry out these SL tasks : {}.".format(', '.join(our_args.SL_tasks_to_perform))) #tag will be caught when entering classif or regr operations
if ("Classif" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform) :
	print("For the classification task, each of these algorithms will be tested : {}".format(', '.join(our_args.Classif_algs)))  #{}".format(', '.join(str(v) for v in our_args.Classif_algs)))
	print("...For those classification algorithms, the studies requested are : {}".format(', '.join(our_args.Classif_studies)))  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
	print("...Each of those classification studies will be run on {} seeds.".format(our_args.Classif_seeds_values))  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
	print("...and the OMC search to carry out is : ", our_args.Classif_OMC_search_type_to_perform)  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
elif ("Regr" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform) :
	print("For the regression task, each of these algorithms will be tested : {}".format(', '.join(our_args.Regr_algs)))  # our_args.Regr_algs is None so use a none exception to manage it later (do a regression or not)
	print("...For those regression algorithms, the studies requested are : {}".format(', '.join(our_args.Regr_studies)))  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
	print("...Each of those regression studies will be run on {} seeds.".format(our_args.Regr_seeds_value))  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
	print("...and the OMC search to carry out is : ", our_args.Regr_OMC_search_type_to_perform)  # {}".format(', '.join(str(v) for v in our_args.Classif_algs)))
##!! pursue the decription later on
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>CLASSIFICATION OPERATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print(bcolors.OKGREEN + "Engaging Classification task of the analysis named "+tag_num_trial+"..." + bcolors.ENDC)
before_all_classif_ops_timer_start = timer_started() ##!! start a timer here for all classif_algs that will be ran
if ("Classif" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform) :
	##!! start a timer here for all classif part that will be ran
	tag_task_type = "Classif" ##tag caught
	classif_seeds = list(range(our_args.Classif_seeds_values)) # for the classification operations, get the seeds list to use # classif_seeds = list(range(2)) for testing or classif_seeds = list(range(1))
	classif_msn = our_args.Classif_MSN # for the classification operations, get the the minimal samples number # classif_msn = 3 for testing
	classif_msn_by_class = our_args.Classif_MSN_by_class  # for the classification operations, get the minimal samples number with the class as response # classif_msn_by_class = 0 for testing
	classif_thres = our_args.Classif_predsprobs_calling_threshold  # for the classification operations, get the default threshold value  when calling predictions after predictions probabilities obtained # classif_thres = 0.5 for testing
	classif_list_cat_fts = our_args.list_categ_fts  # for the classification operations, a list of the feature types to consider categorical (others feature types are considered real or mixed) # classif_list_cat_fts = ["SNV", "CNA", "SNVwCNA","CNAwSNV","SNVwCNAwGEXA","SNVwGEXAwCNA","CNAwSNVwGEXA","CNAwGEXAwSNV","GEXAwSNVwCNA","GEXAwCNAwSNV"] for testing
	classif_omc_search_type = our_args.Classif_OMC_search_type_to_perform  # for the classification operations, a str to easily choose omc or omclight # for testing : classif_omc_search_type = "OMClight"
	models_compared = [classif_omc_search_type, "Allfts"] # for the classification operations, a list of the models compared (2 at the moment) # for testing : do nothing it is done with previously line; just use this line
	classif_profiles_folder = our_args.Classif_profiles_folder # classif_profiles_folder = data_profiles_path for testing
	classif_drugs_folder = our_args.Classif_drugs_folder # classif_drugs_folder = data_drugs_path for testing
	classif_Resp_col_name = our_args.Classif_Resp_col_name # the name of the Response values column # for testing classif_Resp_col_name = "BestResCategory"
	classif_Samples_col_name = our_args.Classif_Samples_col_name # the name of the Samples names column # for testing  classif_Samples_col_name = "Model"
	classif_reduc_data_decision = our_args.Classif_reduce_dataset # in classif, reduce or not dataset # for testing purpose classif_reduc_data_decision = "yes"
	classif_reduc_data_sn = our_args.Classif_reduced_dataset_samples_number # in classif, if reducing dataset for testing purpose, numlber sammples to keep # for testing purpose classif_reduc_data_sn = 10
	classif_CV_folds_number = our_args.Classif_cross_validation_folds_number # in classif, if changing the CV folds number use for testing classif_CV_folds_number = 10
	classif_chosen_pw = our_args.Classif_processing_way # in classif, the chosen processing way of the classifier if to be on many jobs # for testing classif_chosen_pw = "par"
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>Data extraction step 1 / 2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	### data_extraction strategy : data_management_1 (NIBR-PDXE data)
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

	### NB : Data extraction step 2 / 2 happens along the analysis so its sparsed here and there in the rest of the analysis
	# >>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	before_all_classifiers_timer_start = timer_started() ##!! start a timer here for all classif_algs that will be ran
	for classifier in our_args.Classif_algs : # classifier = "RF_Mark2Vpar" for testing or classifier = "DNN_Mark2Vseq"
		tag_alg = classifier.split("_")[0]
		tag_alg_mark = classifier.split("_")[1]
		tag_alg_pw = tag_alg_mark.split("V")[1]
		# make a decision before going forward on the processing way of the alg (this is to give the user the occasion to enforce the processing way of a specific alg mark)
		alg_pw = "no_classif_pw_decision_yet"
		if (classif_chosen_pw == "native") | (classif_chosen_pw == tag_alg_pw):
			alg_pw = tag_alg_pw
		else:
			alg_pw = classif_chosen_pw
		before_a_classifier_timer_start = timer_started() ##!! start a timer here for all the runs this classif_alg will do

		#### Method : Loop on the ctype, drug and featuretype to find the proper info and make a left table.
		### A right table will come from the profile data corresponding or not to left table data.
		### Afterwards, restrict the left and right tables to corresponding data using a unifier (commonly found in both tables) that will be the basis to unify both tables

		####loop1 : on cancertypes (# loop in the ctypes and define each time a drug_response dataframe (cframe) by restricting to only actual considered ctype)
		before_any_ctype_timer_start = timer_started()  # start a clock to get the time after all ctypes
		for ctype in set_of_all_condittions_found:  # ctype = "BRCA" for testing
			tag_ctype = ctype # collect a tag for it for the result files
			cframe = data_loadout_left_all.loc[data_loadout_left_all[Condittion_col_name] == ctype] #take only rows of drug_response that are that ctype
			####loop2 : on treatments ( # loop in the drugs (numbers here) and define each time a specific_cancer-drug_response dataframe (drugframe) by restricting to only actual considered ctype's and drug used)
			before_any_drugs_tested_on_a_ctype_timer_start = timer_started()  # start a clock to get the time after all drugs tested on a ctype
			for drug in set_of_all_TreatmentIDs_found: # drug = 'Treatment17' for testing or drug = 'Treatment18'
				tag_num_drug = drug  # collect a tag for the drug number for the result files
				tag_drugname = dict_TreatmentID_TreatmentName[drug] # collect a tag for the drug name for the result files
				drugframe = cframe.loc[cframe[TreatmentID_col_name] == drug]  # take only rows of drug_response that are that drug
				# starttime = timer_started()  # start a clock to get the time after each feature in a profile
				if len(drugframe) >= classif_msn:  # verify if at least n=35 samples are available as the smallest value encountered for a classification task in oncology
					print(bcolors.WARNING + "Number of samples presented is suffisant for classification. Restricting of data to analysed subcases is starting..." + bcolors.ENDC)
					# intialisation of a dict to keep track of the profiles found for a case (ctype-drug)
					profile_data_found_for_case = {}
					####loop3 : on profile_types (# loop in the features and each time do the appending of the proper feature values to the specific_cancertype-drug_specific dataframe. in each case, the appending is based on both dtaframes containing the unique cosmic ids)
					before_any_profile_in_a_case_timer_start = timer_started()  # start a clock to get the time after all profiles in case
					for featuretype in set_of_all_profiles_found: # featuretype = 'GEX' for testing; use also featuretype = 'CNA'
						before_one_profile_in_a_case_timer_start = timer_started()  # start a clock to get the time after each profile in case
						tag_profilename = featuretype # collect a tag for it for the result file
						feature_val_type = feature_values_type_caracterisation(featuretype,classif_list_cat_fts) #!! use this value in functions to know how to treat datasets values #featuretype and classif_list_cat_fts can go
						featureframe = drugframe.loc[drugframe[Layer_probed_col_name] == featuretype]
						unifier = unifier_creator(ctype, drug, featuretype)
						corresponding_data_loadout_right = data_loadout_right_corresponder(featuretype,data_loadout_right)
						# lets create dframe ie the table of the full data of a subcase (ctype-drug-profile(s)) and that is to divide in training and testing
						dframe_and_profile_data_archived_and_index_starting_fts_cols = data_mgmt_2(unifier, featureframe, corresponding_data_loadout_right, Samples_col_name1, Samples_col_name2, Unifier_key_col_name)
						dframe = dframe_and_profile_data_archived_and_index_starting_fts_cols[0]
						profile_data_archived = dframe_and_profile_data_archived_and_index_starting_fts_cols[1] # for keeping track
						index_starting_fts_cols = dframe_and_profile_data_archived_and_index_starting_fts_cols[2]
						add_entry_in_dict(profile_data_found_for_case,featuretype,profile_data_archived) # for keeping track
						print(bcolors.UNDERLINE + "Entire dataframe is obtained for following subcase: ", ctype, drug, featuretype + bcolors.ENDC)
						# reducing the dataset for faster testing purpose if operator activate the option
						if classif_reduc_data_decision in ["yes", "y"]:
							print(bcolors.WARNING + "Reducing the dataset to",classif_reduc_data_sn,"samples for faster testing purpose..." + bcolors.ENDC)
							dframe = reduction_of_dataset_for_testing_purpose(dframe,Resp_col_name,classif_reduc_data_sn)
							# similar to this but perfected
							# dframe = dframe.iloc[list(range(7)), :]  # testing  # reduction of frame for tests (7 samples, n/2 is 3 so complexities are 2,3)
							# dframe = dframe.iloc[list(range(10)), :]  # testing  # reduction of frame as previously but with more samples to have all classes ##!! rectify the use of reduction as a function and activated from a function managed with an argument
						## formatting the dframe
						print("Features values formatting : categoricals as booleans and reals as floats...")
						dframe = data_mgmt_5(dframe, index_starting_fts_cols, feature_val_type)
						# lets change values of response in the appropriate type (floats for regression and categorial for classification)
						# dframe = data_mgmt_3(dframe,tag_task_type,Resp_col_name) # choose 2nd agr as tag_task_type (defined at start of script as one of "Classif" or "Regr") ###testing uncommented to keep real classes
						# not a tuple returned but just one df so no indexing into a tuple
						# lets eliminate features with only zeros as values, sort the dataframe entries following repsonse col values and remake a new index
						print("Sorting the dataframe entries following response column values and remaking a new index...")
						dframe = data_mgmt_6(dframe, Resp_col_name)

						#-------------reflections
						# data_mgmt_6 can also be at the start of the task 1 and was initialy. moved before the loop on each seed to make it just one time
						# also lets extract the name of the response column because it will be used very often

						# after having obtained dframe and stylised it as required for classfication or regression,lets compute on it
						# dframe will be used for division into training and test set

						if len(dframe) >= classif_msn:
							print(bcolors.WARNING + "Number of samples is suffisant after data formatting. Full analysis of the subcase (", ctype, "-", drug, "-", featuretype, ") is starting..." + bcolors.ENDC)
							#>>>>>>>>>>>RESPONSE COLUMN & FEATURES ACCOUNTING
							dataBin = dframe.loc[:,[Resp_col_name]]  # get the 1st column of data ... # anciently it was dframe[Resp_col_name] but gives a series instead of a df
							# RespClasses = sorted(set(dataBin)) # ...and get the classes # not working so use one of the 2 following lines
							RespClasses = sorted(dataBin.iloc[:, 0].unique())
							binary_classes_le = LabelEncoder()  # the encoder
							binary_classes_le.fit(RespClasses)  # encode the classes to memorize
							encoded_classes = binary_classes_le.classes_
							# RespClasses = sorted(dataBin.Resp_col_name.unique())
							##!!! (they are two for now but universalise later to more classes)
							##!!! for now the alphabetical order of classes is sufficient to call them 0 and 1 as resp to neg class and pos class (find a way later to decide of what class is pos and which is neg (actual workings or reverse it)
							# count and divise by 2 the numbers of values in the 2 classes (to have at least 2 datapoints and dodge a bit HD problem)
							count = len(set(dframe["Sample_id"])) # count the lines of response columns without count the eventuals duplicated lines
							min_complexity_of_tested_MCs = 2  # !!! use this in main and functions
							max_complexity_of_tested_MCs = int((count / 2))  ## Set max_complexity_of_tested_MCs = N/2 #round does gives an int but goes to next interger if x.5 if obtain or more(eg 7/2 become 3.5 and gives 4.0). also that makes ugly values in tables reported. using int() the the nearest integer down is obtain and a nice values if displayed
							trainingset_size_the_biggest = count - 1 # because we are doing LOO on OL2 ##!!! isolate in a function for any CV defined
							num_all_features = len(list(dframe)[5:])

							# >>>>>>>>>>>>>>>>>>>>>>>>>METRICS OF CLASSIFICATION STEP ESTIMATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
							if ("CAB" in our_args.Classif_studies) | ("Both" in our_args.Classif_studies):
								print(">>>>>>>>>>>>>>>>>>>>>>>>ME in developpement<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
								# >>>>>>>>>>>>>>>>CREATING THE DF TO RECEIVE THE RESULTS OF METRICS ESTIMATIONS
								df_of_results_omc_mdl = df_of_results_for_metrics_one_mdl_creator()
								df_of_results_allfts_mdl = df_of_results_for_metrics_one_mdl_creator()
								df_of_results_omcmdl_vs_allftsmdl_omc_info = df_of_results_for_metrics_all_mdls_creator()
								df_of_results_omcmdl_vs_allftsmdl_allfts_info = df_of_results_for_metrics_all_mdls_creator()
								df_of_results_omcmdl_vs_allftsmdl = df_of_results_for_metrics_all_mdls_creator() # is built during computations if no class sparsity but is initialised anyway because need to fill in case of class sparsity error where computations dont happen
								print(bcolors.BOLD + "Result file created for this subcase analysis, each line is a seed and last is all sedds averaging. Contains : 0 lines" + bcolors.ENDC)
								#>>>>>>>>>>CHECKING DATA SPARSITY
								if (dataBin.iloc[:, 0].value_counts()[encoded_classes[0]] >= classif_msn_by_class) & (dataBin.iloc[:, 0].value_counts()[encoded_classes[1]] >= classif_msn_by_class) :
									print(bcolors.WARNING + "Data sparsity on classes checked. No data issue met. Analysis ongoing..." + bcolors.ENDC)
									#>>>>>>>>>>>> PREAPRING FOR OUTER LOOP 1
									print(" PREPARING FOR OUTER LOOP 1 : creation of metrics collectors by seed")
									# create the vectors to collect the metrics of performance either for one seed and for being the one to be tranformed to give the collector of a metric for all seeds
									# (this final reason is the reason why we init it here b4 the loop on seeds)
									# collectors of the tables to keep track of the predictions probabilities and the class called each time from those probabitilities
									all_seeds_col_of_df_from_print_pred_col = []
									all_seeds_col_of_df_from_print_pred_all_col = []
									# collectors related to omc
									ol1_col_med_ol2_col_omc = []  # P1I a list of the median of the best features at each seed; transmit a median of it; to self
									ol1_col_persistentin_ol2_col_of_topfeats_corresponding_to_omc_il1 = [] # collectors for at each seed, a list of the persistent fts...
									ol1_col_nonpersistentin_ol2_col_of_topfeats_corresponding_to_omc_il1 = [] #...and a list of non persistent fts
									# colllectors for the metrics of each seed to make stats for all seeds ran (omc mdl)
									ol1_acc_col_omc_for_1seed = [] # a list to collect the acc of the omc mdl for each seed
									ol1_MCC_col_omc_for_1seed = []  # P2I a list of the mcc values of each mc; transmit a med of it; to med
									ol1_PREC_col_omc_for_1seed = [] # P6I transmit a median of it; to self
									ol1_RECALL_col_omc_for_1seed = []  # P7I transmit a median of it; to self
									ol1_fpr_col_omc_for_1seed = [] # a list to collect the fpr of the omc mdl for each seed
									ol1_TP_col_omc_for_1seed = [] # P9I transmit a median of it; to self
									ol1_FN_col_omc_for_1seed = [] # a list to collect the support of the omc mdl for each seed
									ol1_TN_col_omc_for_1seed = [] # a list to collect "the num_tests_w_outcome_as_class_pos" of the omc mdl for each seed
									ol1_FP_col_omc_for_1seed = []  # a list to collect "the num_tests_w_outcome_as_class_pos" of the omc mdl for each seed
									# colllectors for the metrics of each seed to make stats for all seeds ran (omc mdl)
									ol1_acc_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_MCC_col_allfts_for_1seed = []  # P4I a list of the mcc values of each all_model; transmit a med of it; to med
									ol1_PREC_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_RECALL_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_fpr_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_TP_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_FN_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_TN_col_allfts_for_1seed = [] # a list to collect the acc of the allfts mdl for each seed
									ol1_FP_col_allfts_for_1seed = []  # a list to collect the acc of the allfts mdl for each seed
									# collectors for the auc and the roc curve computations for each seed (for the omc mdl)
									tprs_col_by_seed_omc_mdl = []  # intialise a list to stock the arrays of tpr values for each iteration
									aucs_col_by_seed_omc_mdl = []  # intialise a list to stock the arrays of auc values for each iteration
									mean_fpr_by_seed_omc_mdl = np.linspace(0, 1, 100)  # a gallery of values to base the interpolating on ##!! change the 100 to number of samples that is equals to num of fpr and tpr that will be at each iteration computed
									figure_omc_mdl, ax1 = plt.subplots()
									# collectors for the auc and the roc curve computations for each seed (for the allfts mdl)
									tprs_col_by_seed_allfts_mdl = []  # intialise a list to stock the arrays of tpr values for each iteration
									aucs_col_by_seed_allfts_mdl = []  # intialise a list to stock the arrays of auc values for each iteration
									mean_fpr_by_seed_allfts_mdl = np.linspace(0, 1, 100)  # a gallery of values to base the interpolating on ##!! change the 100 to number of samples that is equals to num of fpr and tpr that will be at each iteration computed
									figure_allfts_mdl, ax2 = plt.subplots()
									# collectors for a figure reuniting both models averages roc curves
									figure_mdl_vs_mdl, ax3 = plt.subplots()
									average_roc_curve_init(ax3)

									# ol1_SPEC_col_omc_for_1seed = [] # P8I transmit a median of it; to self
									# ol1_col_num_pred_call_as_cPos_w_omc_by_1seed = []  # P10I transmit a sum of it; to self
									#===============> START OF OUTER LOOP 1
									# if an analysis use a ML alg exploiting a random seed, multiples runs can be made to see how changing the seed affects the robustness of the analysis
									# that seed has to be initialised before the analysis
									# this is a loop on the seed values (10 seeds with the zero being 5678 and the rest is 1 to 9 values)
									#old place of seeds lists to chosen one from
									all_classif_seeds_start = timer_started()  # start a clock to get the time after alls seeds have been used
									print(bcolors.OKGREEN + " > START OF OUTER LOOP 1 (OL1): loop on all",len(classif_seeds),"seeds..." + bcolors.ENDC)
									for aseed in classif_seeds: # loop on the seeds each is n #outer loop 1 # for testing aseed = classif_seeds[0]
										a_classif_seeds_start = timer_started()  # start a clock to get the time after one seed use
										print(bcolors.OKGREEN + " > - OL1's seed",aseed,"has started being used..." + bcolors.ENDC)
										np.random.seed(aseed)  # fixate the actual seed
										separate_value_for_short_random_actions = classif_seeds[0]
										# init 3 :
										# 1--# a prelim for the best fts at each fold (LOO so many folds) # list of # of fts in each seed analysis to compute the median over it bcuz contains values in of samples in the LOO
										# best.feats.list = c() #P1_prelim_I ## test if really in use or needed ???
										# 2---one collector by model, to stock the classes predicited for each fold of the val set of the (all model) (one class because its after prob thres is used)
										ol2_col_pred_call_1by_seed_w_allfts = []  # (for all model)
										ol2_col_pred_call_1by_seed_w_omc = []  # (for omc)
										# 3---#a dataframe by model, to keep track of predictions probs of classes in folds with only the header  : fold, pdxindex, res, sen (the two class in the same order they are given to the classwt)
										# print_pred = pd.DataFrame(pd.np.empty((0,4))*pd.np.nan) ##!!! maybe not needed
										# print_pred.columns = ["Fold", "PDXindex","Res","Sen"] # (for the OMC) ##!!! maybe not needed
										print_pred_col = []  # collect here the df resulting before concatenating them into print_pred df of content
										# print_pred_all = pd.DataFrame(pd.np.empty((0, 4)) * pd.np.nan) ##!!! maybe not needed
										# print_pred_all.columns = ["Fold", "PDXindex", "Res", "Sen"] # (for the all version) ##!!! maybe not needed
										print_pred_all_col = [] # collect here the df resulting before concatenating them into print_pred_all df of content

										ol2_col_of_omc_il1 = []  # each il1 loop will get 1 OMC and its mcc; you have to catch it in each iteration of ol2 and get an average on the whole lot (eg median) later
										ol2_col_of_omc_il1_mcc = []
										ol2_col_of_topfeats_corresponding_to_omc_il1 = []  # each il1 loop will get 1 OMC and a list of its topfeats; you have to catch it in each iteration of ol2 and get an average (the persistent fts) on the whole lot (eg median) later
										# folds creation
										ol2_folds = stratKfolds_making(classif_CV_folds_number,aseed,dframe,index_starting_fts_cols,Resp_col_name,True)
										# ol2_folds = dframe.index #(gives a list of numbers from 1 to num of folds to iterates on) # uses response col and # of folds wanted, here LOO so nrow of all_data
										# j = 0 #trial 1
										# ol2_folds = [0,1] #testing
										print(bcolors.WARNING + " >> START OF OUTER LOOP 2 (OL2): LOOCV on all data ie on",len(ol2_folds),"folds" + bcolors.ENDC)
										print(bcolors.WARNING + " >> Objective : The models (2 to n/2, allfts) complexities and their predictions will be searched for comparison to find OMC" + bcolors.ENDC)
										for one_ol2_fold_tools in ol2_folds: #outer loop 2 (each fold for LOO on the training set # for testing one_ol2_fold_tools = ol2_folds[0]
											print(bcolors.WARNING + " >> - OL2's fold",one_ol2_fold_tools[0], "has started being used..." + bcolors.ENDC)
											print(bcolors.WARNING + " >> -- Training and prediction of an Allfts model" + bcolors.ENDC)
											# mk training and testing data for each fold of LOOCV on all the data (use j to def sets and get the predictions )

											# ============================BUILD TRAINING SET AND TEST SET OF THE FOLD============================================================
											ol2_train_data = dframe.loc[one_ol2_fold_tools[1],:] # Set the training set (for a row j of data, select all rows of data without the row of index 1 (here the frame start index at 1)
											ol2_test_data = dframe.loc[one_ol2_fold_tools[2],:]  # Set the validation set (for a row j of data, select only the row j)
											# ....train_y, test_y of trainset_fold
											ol2_train_y = ol2_train_data.loc[:,[Resp_col_name]]  # keep the training set response column values (37 values) (for the row j of data analysed, the 1st value)
											ol2_test_y = ol2_test_data.loc[:,[Resp_col_name]]  # same (1 values)
											# .....train_x, test_x of trainset_fold
											ol2_train_x = ol2_train_data[list(ol2_train_data)[5:]]    # keep only the features (index 1 to ncol) #here are dropped the model and the response #odd code tho because _c(i,j) drops cols i to j
											ol2_test_x = ol2_test_data[list(ol2_test_data)[5:]]

											# ============================================== ALL FTS MODEL : ONE OF THE MODEL COMPARED (TRAINING AND STAASHING OF PREDICTION==============================
											#---training the allfts model
											# *******trying env specific loads
											if tag_alg == "DNN":
												# env_specific_loads_for_DNN(aseed)
												ol2_pred_2probs_mdl_allfts = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, ol2_train_x, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x, "prob", aseed)
											elif tag_alg == "SVM":
												# env_specifics
												ol2_pred_2probs_mdl_allfts = classifier_as_SVM_intro_train_pred(tag_alg_mark, ol2_train_x, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x, "prob", aseed)
											else:
												# ---introduce the model
												mdl_allfts = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x, ol2_train_y, aseed,encoded_classes,max_cores_minus2)  ## class weighting with classwt #to be check if not inversed
												# ---train the model on train_x,train_y
												mdl_allfts_fitted = classifier_model_training(mdl_allfts, ol2_train_x, ol2_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
												# -- mk test for each fold
												ol2_pred_2probs_mdl_allfts = classifier_model_prediction(mdl_allfts_fitted, ol2_test_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
											# *******
											ol2_pred_call_mdl_allfts = []  # last_stop_end_we
											ol2_pred_call_mdl_allfts = prediction_calling(ol2_pred_2probs_mdl_allfts, ol2_pred_call_mdl_allfts, classif_thres, encoded_classes, separate_value_for_short_random_actions)
											# ---adding raw predictions to a keep tracker of predictions in folds
											print_pred_all_col = raw_predictions_pusher(ol2_pred_2probs_mdl_allfts, one_ol2_fold_tools[0], one_ol2_fold_tools[2], encoded_classes, print_pred_all_col,ol2_pred_call_mdl_allfts,aseed)
											# ---adding the call to a collector for stats or metrics
											ol2_col_pred_call_1by_seed_w_allfts = called_predictions_pusher(ol2_pred_call_mdl_allfts, one_ol2_fold_tools[2], ol2_col_pred_call_1by_seed_w_allfts)
											# ----------------------end of all_model predictions making and stashing

											# ============================================== OMC TUNING 1/2 : OMC MODEL RESEARCH OVER MULTIPLES FOLDS =======================================
											# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of tested portion
											### IL1 will be multiproccessed, each fold a process using Process
											# Step 1 : initialise the needed parameters
											OMC_quest_allfts_vs_MCs_wide_col = []  # intialise a big list to put in it all the allfts anc omc predictions called to compute mcc on it and compare the (MCs,allfts) models
											# >>>>>>>>>>>MAKE # OF FOLDS FOR CV
											il1_folds = stratKfolds_making(classif_CV_folds_number, aseed, ol2_train_data, index_starting_fts_cols,Resp_col_name, True)
											# il1_folds = ol2_train_data.index  # create the folds for the LOO for the omc model
											len_il1_folds = len(il1_folds)
											# l = 1 # testing
											# il1_folds = [1,2,3] #testing
											# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
											print(bcolors.OKBLUE + " >>> START OF INNER LOOP 1 (IL1): LOOCV on OL2 present fold", one_ol2_fold_tools[0], "training data. Going for", len_il1_folds, "folds" + bcolors.ENDC)
											print(bcolors.OKBLUE + " >>> With each fold, training and prediction with many MCs and thus determining later OMC on one IL1 loop" + bcolors.ENDC)
											# for l in il1_folds: # this was the start of  # inner loop 1 #parallelise 37 jobs on each of on 37 folds/samples # the packages necessary are given # the results are combined in row after row fashion to form a matrix #and that in order of the samples

											# Step 2 : describe the function that each process will launch when it will accept a job in a queue
											# bench_of_results = [] # for testing # a collector for the result of each fold carried out to sort later and make an ordered df (a shared list is used in worker function of xproc)

											# fixate a seed for all this xproc part so that variactions dont happen when the env is copied for each process
											np.random.seed(separate_value_for_short_random_actions)
											def il1_job(respective_fold_tools): # respective_fold_id = 2 for testing
												print(bcolors.OKBLUE + " >>> - Task running IL1's fold", respective_fold_tools[0], "has started being carried out..." + bcolors.ENDC)
												print(bcolors.OKBLUE + " >>> - Ranking the features in the fold to get the FS for every MCs to estimate next" + bcolors.ENDC)
												# >>>>>>>>>>>>PRE-BUILDING THE SETS NEEDED
												# ---1-making of trainframe and valframe
												il1_train_data = ol2_train_data.loc[respective_fold_tools[1],:]  # Set the training set of 36 (by dropping one row/sample)
												il1_test_data = ol2_train_data.loc[respective_fold_tools[2],:]  # Set the validation set of 1 (by keeping only that previously dropped sample)
												# trainframe_y & valframe_y
												il1_train_y = il1_train_data.loc[:, [Resp_col_name]]  # train_y (needed to train)
												# il1_test_y = il1_test_data.loc[:, [Resp_col_name]]  # val_y (not needed because prediction estimation uses the resp_col_nme of the dataset directly restricted to the indexes implicated)
												# trainframe_x & valframe_x
												il1_train_x = il1_train_data[list(il1_train_data)[5:]]  # train_x (needed to train) (## should not transmit to self but to new name)
												il1_test_x = il1_test_data[list(il1_test_data)[5:]]  # val_x (needed to predict)
												# <<<<<<<<<<<<<<END OF PRE-BUILDING THE SETS NEEDED
												if classif_omc_search_type == "OMC":
													# ~~~~~~~~~~~~~~~~~~~~~ optional part (depending on omc search chosen)
													# >>>>>>>>>>>>>>START OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
													# ---2-estimate the fts with two-sided fisher exact test p-values or two sided unpaired t-test p-values
													# we are about to train a model so lets re set the seed at its rightful place in case a function from a module changed it
													np.random.seed(aseed)
													#*******trying env specific loads
													if tag_alg == "DNN":
														# env_specific_loads_for_DNN(aseed)
														il1_pred_2probs_w_allfts = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, il1_train_x, il1_train_y, feature_val_type, binary_classes_le, il1_test_x, "prob", aseed)
													elif tag_alg == "SVM":
														# env_specific_loads_for_DNN(aseed)
														il1_pred_2probs_w_allfts = classifier_as_SVM_intro_train_pred(tag_alg_mark, il1_train_x, il1_train_y, feature_val_type, binary_classes_le, il1_test_x, "prob", aseed)
													else:
														# ---introduce the model
														rf1 = classifier_introduction(tag_alg, tag_alg_mark, il1_train_x, il1_train_y,aseed,encoded_classes,max_cores_minus2)  ## class weighting with classwt #to be check if not inversed
														# ---train the model on train_x,train_y
														rf1_fitted = classifier_model_training(rf1, il1_train_x, il1_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
														# -- mk test for each fold
														il1_pred_2probs_w_allfts = classifier_model_prediction(rf1_fitted, il1_test_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
													# *******
													il1_pred_call_w_allfts = []
													il1_pred_call_w_allfts = prediction_calling(il1_pred_2probs_w_allfts, il1_pred_call_w_allfts, classif_thres, encoded_classes, separate_value_for_short_random_actions)
													##maybe add this!!! ---adding raw predictions to a keep tracker of predictions in folds
													##maybe add this!!! ---adding the call to a collector for stats or metrics
													# <<<<<<<<<<<<<<END OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
													# ~~~~~~~~~~~~~~~~~~~~~ end of optional part (depending on omc search chosen)
												# >>>>>>>>>>>>>>ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
												# ---process of limiting training data to only fts who shows variation (at least one variation in a sample)
												# il1_train_x = eliminate_non_variable_fts(il1_train_x) # no need bbcuz we manage the exception
												# ---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (a vector of fts's p values in the order of the fts)
												il1_fts_ranking = ranker_by_pval_v2(il1_train_x, il1_train_y, feature_val_type, Resp_col_name,encoded_classes)
												# <<<<<<<<<<<END OF ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
												# >>>>>>>>>>>MCs PREDICTIONS MAKING
												# ---3-training a model for each MC and test it to compare them and take the best OMC
												il2_col_il2_pred_call_mc = []  # a collector of prediction called at each loop on complxities
												# mc = 2 # trial
												# print "- Estimating the MCs"
												print(bcolors.FAIL + " >>>> START OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_tools[0], "training data. Going for", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
												for mc in list(range(2, (max_complexity_of_tested_MCs + 1))):  # inner loop 2 # loop on the MCs
													print(bcolors.FAIL + " >>>> - In Fold", respective_fold_tools[0], ", MC of", mc, "features has started being made and estimated..." + bcolors.ENDC)
													topfeats = il1_fts_ranking[1][0:mc]
													## isolate as ranking function for p_value
													# rank it, keep this rank from original, get the names, get the top MC or those names) ## try to gives the rank at end of estimating fts to select in it directly here
													# set up a training of the built with elected fts MC model
													il2_train_mc_x = il1_train_x.loc[:, topfeats]  # train_x restricted to MC fts
													il2_test_mc_x = il1_test_x.loc[:, topfeats]  # test_x restricted to MC fts
													np.random.seed(aseed)
													# *******trying env specific loads
													if tag_alg == "DNN":
														# env_specific_loads_for_DNN(aseed)
														il2_pred_2probs_w_mc = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, il2_train_mc_x, il1_train_y, feature_val_type, binary_classes_le, il2_test_mc_x, "prob", aseed)
													elif tag_alg == "SVM":
														# env_specific_loads_for_DNN(aseed)
														il2_pred_2probs_w_mc = classifier_as_SVM_intro_train_pred(tag_alg_mark, il2_train_mc_x, il1_train_y, feature_val_type, binary_classes_le, il2_test_mc_x, "prob", aseed)
													else:
														# ---introduce the model
														rf2 = classifier_introduction(tag_alg,tag_alg_mark, il2_train_mc_x, il1_train_y,aseed,encoded_classes,max_cores_minus2)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
														# ---train the model on train_x,train_y
														rf2_fitted = classifier_model_training(rf2, il2_train_mc_x, il1_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
														# -- mk test for each fold
														il2_pred_2probs_w_mc = classifier_model_prediction(rf2_fitted, il2_test_mc_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
													# *******
													il2_pred_call_mc = []
													il2_pred_call_mc = prediction_calling(il2_pred_2probs_w_mc, il2_pred_call_mc, classif_thres, encoded_classes, separate_value_for_short_random_actions)
													il2_col_il2_pred_call_mc.append(il2_pred_call_mc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
												# ------end of inner loop2
												print(bcolors.FAIL + " <<<< END OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_tools[0], "training data. Gone through", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
												# make a df that shows all found for this fold of il1 including all in il2 mcs
												# OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame()  # the part of the OMC quest wide col for a fold
												the_fold_column_content = [respective_fold_tools[0]] * len(il1_test_data.index.tolist())  # col fold
												the_il1_test_data_indexes_column_content = il1_test_data.index.tolist()  # col indexes in il1_test_data
												# il1_pred_call_w_allfts # is the [preds called for allfts on il1 to compare with the mcs models prediction] column content
												# il2_pred_call_mc # is the [each mc preds called] column content and a lot are in store into il2_col_il2_pred_call_mc so use a * in front of it in the zip to extract its content at the same level
												if classif_omc_search_type == "OMC":
													list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, il1_pred_call_w_allfts, *il2_col_il2_pred_call_mc))  # old zipping coming with including allfts model in estimations of mcc for omc mdls # zip col1,col2,col3, and extrcted col4
												else:
													list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, *il2_col_il2_pred_call_mc)) # zip col1,col2, and extrcted col4

												OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame(list_of_all_cols_for_a_fold_zipped)  # init the part of the OMC quest wide col for a fold # make a df of it
												tuple_result = (tuple([respective_fold_tools[0], OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold]))  # keep a tuple of the fold id and the mini df produced (later this is used to put in the right order the results and make a full df of them)
												print(bcolors.OKBLUE + " <<<  - Task running Fold", respective_fold_tools[0], ",among", len_il1_folds, "folds, is done. Carried out by " + current_process().name + bcolors.ENDC)
												return tuple_result
											#~~~~modes of batch runs (parallel or sequential)
											if alg_pw == "par": # launching IL1 in a multiprocessing way
												il1_multiprocessing_handler(il1_folds, tag_num_xproc, OMC_quest_allfts_vs_MCs_wide_col, il1_job)
											else: # launching IL1 in a sequential way
												il1_sequential_processing_handler(il1_folds, OMC_quest_allfts_vs_MCs_wide_col, il1_job)
											# the list OMC_quest_allfts_vs_MCs_wide_col is updated with values add it to concatenate later as a pd
											# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
											#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of tested portion
											# end of inner loop 1 (official end)
											print(bcolors.OKBLUE + " <<< END OF INNER LOOP 1 (IL1): LOOCV on OL2 present fold", one_ol2_fold_tools[0], ". IL1 counted", len_il1_folds, "folds" + bcolors.ENDC)
											#lets concatenate all dfs, one from each fold of inner loop 1 into one (parse on it later, restrict to fold, get mcc ie many mcc for many folds of a models s med.mcc)
											OMC_quest_allfts_vs_MCs_wide = pd.concat(OMC_quest_allfts_vs_MCs_wide_col)
											# lets rename the columns properly to what they correspond
											if classif_omc_search_type == "OMC":
												edited_colnames = ["Fold"]+["Sample_index"]+[num_all_features]+list(range(2, (max_complexity_of_tested_MCs+1))) # old naming cols coming with including allfts model in estimations of mcc for omc mdls
											else:
												edited_colnames = ["Fold"] + ["Sample_index"] + list(range(2, (max_complexity_of_tested_MCs + 1)))
											OMC_quest_allfts_vs_MCs_wide.columns = edited_colnames
											# <<<<<<<<<<<<END OF PREDICTIONS MAKING
											# >>>>>>>>>>>ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
											print(bcolors.WARNING + " << - OL2 is still at fold",one_ol2_fold_tools[0],": Choosing OMC among MCs of IL1" + bcolors.ENDC)
											# ------------Median MCCs calculations as the criteria to elect omc on
											dict_of_mcc_values_by_mdl_to_update = {}
											# to calculate MCCs, restrict the OMC_quest_wide to the mdl_column for the corresponding preds...
											frame_of_all_mdls_called_predictions = OMC_quest_allfts_vs_MCs_wide.iloc[:, 2:len(OMC_quest_allfts_vs_MCs_wide.columns)]
											# index_start_space_of_cols_as_mdls = 2 and endplus1_space_of_cols_as_mdls = len(endplus1_space_of_cols_as_mdls.columns)
											# space_of_cols_indexes_to_consider_as_mdls = 2: or if native = :
											# ...and mock up a same length response column from ol2_train_y with each row of col of indexes having the corresponding response
											frame_of_all_mdls_implicated_samples_observations = ol2_train_y.loc[OMC_quest_allfts_vs_MCs_wide["Sample_index"].tolist(), :] ##!!! last_stop_check if the 2 last frames works after the columns renaming
											dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, frame_of_all_mdls_called_predictions, frame_of_all_mdls_implicated_samples_observations, Resp_col_name, encoded_classes)
											# ~~~~~~~~preceed with this restriction to use singular mcc calculation
											# for l in il1_folds:
											# 	restricted_to_samples_in_test = restriction_of_MCCs_wide(OMC_quest_allfts_vs_MCs_wide, l, il1_test_data.index.tolist())
											#   dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, restricted_to_samples_in_test, 2, len(restricted_to_samples_in_test.columns), il1_test_y, Resp_col_name, encoded_classes)
											# ~~~~~~~~
											# ---choosing the best mc....
											omc_il1,omc_il1_mcc = OMC_founder_in_dict_MCs_MCCs(dict_of_mcc_values_by_mdl_to_update)
											# and...keeping it away...
											ol2_col_of_omc_il1.append(omc_il1) # 2 lists are used to also take the mcc with us alongside the omc until the end
											ol2_col_of_omc_il1_mcc.append(omc_il1_mcc)
											# ...contains the best_fts_mc at each fold of LOO of the outer loop2 ie 38 values
											# <<<<<<<<<<<<<<<END OF ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
											# >>>>>>>>>>>>>START OF OMC PREDICTIONS MAKING
											print(bcolors.WARNING + " << - OL2 is still at fold", one_ol2_fold_tools[0], ": Getting for OMC of IL1 its predictions on OL2 training data" + bcolors.ENDC)
											# ---process of estimating fts for fts ranking
											# ---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (z dict of feat: p_val )
											ol2_fts_ranking = ranker_by_pval_v2(ol2_train_x, ol2_train_y, feature_val_type, Resp_col_name,encoded_classes)
											#---get on ol2_data the prediction of the omc
											if classif_omc_search_type == "OMC":
												model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + ", " + str(num_all_features) + "]"
												if omc_il1 == num_all_features: # ------ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS # basically its if nfts = all_var gives pred_prob of all_model otherwise use the function for ranking fts with p_values
													# topfeats_in_ol2_fts_ranking = ol2_train_x.columns.tolist() #line not needed for the prediction here because its the all fts model so just taking the whole ol2trainset
													# *******trying env specific loads
													if tag_alg == "DNN":
														# env_specific_loads_for_DNN(aseed)
														ol2_pred_2probs_w_omc = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, ol2_train_x, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x, "prob", aseed)
													elif tag_alg == "SVM":
														# env_specific_loads_for_DNN(aseed)
														ol2_pred_2probs_w_omc = classifier_as_SVM_intro_train_pred(tag_alg_mark, ol2_train_x, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x, "prob", aseed)
													else:
														# ---introduce the model
														mdl_allfts = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x, ol2_train_y, aseed,encoded_classes,max_cores_minus2)  ## class weighting with classwt #to be check if not inversed
														# ---train the model on train_x,train_y
														mdl_allfts_fitted = classifier_model_training(mdl_allfts, ol2_train_x, ol2_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
														# -- mk test for each fold
														ol2_pred_2probs_w_omc = classifier_model_prediction(mdl_allfts_fitted, ol2_test_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
													# *******
													ol2_pred_call_w_omc = []  # last_stop_end_we
													ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, encoded_classes, separate_value_for_short_random_actions)
													# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
													# ---adding raw predictions to a keep tracker of predictions in folds
													print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, one_ol2_fold_tools[0], one_ol2_fold_tools[2], encoded_classes, print_pred_col,ol2_pred_call_w_omc,aseed)
													# ---adding the call to a collector for stats or metrics
													ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, one_ol2_fold_tools[2], ol2_col_pred_call_1by_seed_w_omc)
												else : # ------A MODEL SMALLER IN SIZE THAN ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS
													# restrict train and test to fts of omc
													topfeats_in_ol2_fts_ranking = ol2_fts_ranking[1][0:omc_il1]
													ol2_train_x_w_omc_il1 = ol2_train_x.loc[:, topfeats_in_ol2_fts_ranking]  # train_x restricted to MC fts
													ol2_test_x_w_omc_il1 = ol2_test_x.loc[:, topfeats_in_ol2_fts_ranking]  # test_x restricted to MC fts
													np.random.seed(aseed)
													# *******trying env specific loads
													if tag_alg == "DNN":
														# env_specific_loads_for_DNN(aseed)
														ol2_pred_2probs_w_omc = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x_w_omc_il1, "prob", aseed)
													elif tag_alg == "SVM":
														# env_specific_loads_for_DNN(aseed)
														ol2_pred_2probs_w_omc = classifier_as_SVM_intro_train_pred(tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x_w_omc_il1, "prob", aseed)
													else:
														# ---introduce the model
														rf3 = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y,aseed,encoded_classes,max_cores_minus2)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
														# ---train the model on train_x,train_y
														rf3_fitted = classifier_model_training(rf3, ol2_train_x_w_omc_il1, ol2_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
														# -- mk test for each fold
														ol2_pred_2probs_w_omc = classifier_model_prediction(rf3_fitted, ol2_test_x_w_omc_il1, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
													# *******
													ol2_pred_call_w_omc = []
													ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, encoded_classes, separate_value_for_short_random_actions)
													# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
													# ---adding raw predictions to a keep tracker of predictions in folds
													print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, one_ol2_fold_tools[0], one_ol2_fold_tools[2], encoded_classes, print_pred_col,ol2_pred_call_w_omc,aseed)
													# ---adding the call to a collector for stats or metrics
													ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, one_ol2_fold_tools[2], ol2_col_pred_call_1by_seed_w_omc)
													##*******deal with this line later
													# il2_col_il2_pred_call_mc.append(ol2_pred_call_w_omc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
													# *******deal with this line later
											else: # ------A MODEL SMALLER IN SIZE THAN ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS
												model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + "]"
												# restrict train and test to fts of omc
												topfeats_in_ol2_fts_ranking = ol2_fts_ranking[1][0:omc_il1]
												ol2_train_x_w_omc_il1 = ol2_train_x.loc[:, topfeats_in_ol2_fts_ranking]  # train_x restricted to MC fts
												ol2_test_x_w_omc_il1 = ol2_test_x.loc[:, topfeats_in_ol2_fts_ranking]  # test_x restricted to MC fts
												np.random.seed(aseed)
												# *******trying env specific loads
												if tag_alg == "DNN":
													# env_specific_loads_for_DNN(aseed)
													ol2_pred_2probs_w_omc = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x_w_omc_il1, "prob", aseed)
												elif tag_alg == "SVM":
													# env_specific_loads_for_DNN(aseed)
													ol2_pred_2probs_w_omc = classifier_as_SVM_intro_train_pred(tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, feature_val_type, binary_classes_le, ol2_test_x_w_omc_il1, "prob", aseed)
												else:
													# ---introduce the model
													rf3 = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, aseed,encoded_classes,max_cores_minus2)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
													# ---train the model on train_x,train_y
													rf3_fitted = classifier_model_training(rf3, ol2_train_x_w_omc_il1, ol2_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
													# -- mk test for each fold
													ol2_pred_2probs_w_omc = classifier_model_prediction(rf3_fitted, ol2_test_x_w_omc_il1, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
												# *******
												ol2_pred_call_w_omc = []
												ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, encoded_classes, separate_value_for_short_random_actions)
												# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
												# ---adding raw predictions to a keep tracker of predictions in folds
												print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, one_ol2_fold_tools[0], one_ol2_fold_tools[2], encoded_classes, print_pred_col,ol2_pred_call_w_omc,aseed)
												# ---adding the call to a collector for stats or metrics
												ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, one_ol2_fold_tools[2], ol2_col_pred_call_1by_seed_w_omc)
												##*******deal with this line later
												# il2_col_il2_pred_call_mc.append(ol2_pred_call_w_omc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
												# *******deal with this line later
											# end of judging the feature ranking of the omc in all folds of the training data
											# >>>>>>>>>> END OF OMC PREDICTIONS MAKING
										# end of outer loop 2 (on each fold for LOOCV on all training set, predictions are reaped)
										print(bcolors.WARNING + " << END OF OUTER LOOP 2 (OL2): LOOCV on all data ie on", len(ol2_folds), "folds" + bcolors.ENDC)
										# >>>>>>>>>>>>METRICS computations FOR EACH SEED
										# ~~~~~~>(FOR THE OMC MODEL)
										print(bcolors.OKGREEN + " > - # OL1 is still at seed", aseed, ": Metrics of the OMC model on this one seed alone..." + bcolors.ENDC)
										# 1: Create the requirements...
										# ~~~~~(finishing touches for outer loop 1 collectors)-A
										# lets add a median of all the OMCs from OL2, one from each fold on the training set (getting one OMC estimation by seed) # P1 (P1U)
										med_ol2_col_omc = np.median(ol2_col_of_omc_il1)
										ol1_col_med_ol2_col_omc.append(med_ol2_col_omc)  # P1U # a list that collect the # of fts found in at least 50% at each seed analysis
										# ~~~~~(finishing touches for outer loop 1 collectors)-B-omc model
										#### lets make for the 2 models (omc and allfts) the df of the preds called collected to ready it for their mcc calculations
										df_from_ol2_col_pred_call_1by_seed_w_omc = pd.DataFrame(ol2_col_pred_call_1by_seed_w_omc)
										# lets rename the columns properly to what they correspond
										edited_colnames_for_ol2_col_pred_call_1by_seed_w_omc = ["Sample_index"] + ["OMC_called_preds"]
										df_from_ol2_col_pred_call_1by_seed_w_omc.columns = edited_colnames_for_ol2_col_pred_call_1by_seed_w_omc
										# on the full dataset LOO folds, to calculate the MCC and others metrics for the OMC, make a df of the mdl corresponding preds and the responses...
										frame_of_omc_mdl_called_predictions_on_all_data = df_from_ol2_col_pred_call_1by_seed_w_omc.iloc[:,[1]]
										frame_of_omc_mdl_implicated_samples_observations_on_all_data = dataBin.loc[df_from_ol2_col_pred_call_1by_seed_w_omc["Sample_index"].tolist(), :]
										# ~~~~~(finishing touches for outer loop 1 collectors)-C-omc model
										# lets report the predictions with each seed in a file from the temp dataframe ## no need to write, juste make df of it # used for the auc computations and roc curve plot
										df_from_print_pred_col = pd.concat(print_pred_col)
										all_seeds_col_of_df_from_print_pred_col.append(df_from_print_pred_col)  # stock it from all seeds version creation
										# on the full dataset LOO folds, to calculate the AUC and make the roc curve, make 2 arrays of the mdl corresponding probs of preds and the responses...
										array_of_omc_mdl_predictions_2probs_on_all_data = np.array(df_from_print_pred_col.loc[:,[df_from_print_pred_col.columns[4]]]) # df_from_print_pred_col.columns[4] is the name of the col containing the probabilities of the pos class. its colname is derived from RespClasses[1] but has been modified so we can't catch it with it
										array_of_omc_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data = np.array(dataBin.loc[df_from_print_pred_col[df_from_print_pred_col.columns[2]].tolist(), :]) # df_from_print_pred_col.columns[2] is the same as the older "Test_sample_index"
										#~~~~~~~~uncomment this for indivudal computation of the mcc
										# mcc_w_omc_as_dict_of_one_entry = {}
										# mcc_w_omc_as_dict_of_one_entry = calculate_mcc_wo_storing(frame_of_omc_mdl_called_predictions_on_all_data, frame_of_omc_mdl_implicated_samples_observations_on_all_data, Resp_col_name, encoded_classes).values()[0]
										# ~~~~~~~~
										# 2 : ...calculate the 8 metrics...
										metrics_mdl_omc = pd_ml_classif_report_on_cm_binary(frame_of_omc_mdl_implicated_samples_observations_on_all_data, frame_of_omc_mdl_called_predictions_on_all_data,binary_classes_le)
										acc_w_omc = metrics_mdl_omc[0]
										mcc_w_omc = metrics_mdl_omc[1]
										precision_w_omc = metrics_mdl_omc[2]
										recall_w_omc = metrics_mdl_omc[3]
										fpr_w_omc = metrics_mdl_omc[4]
										tp_w_omc = metrics_mdl_omc[5]
										fn_w_omc = metrics_mdl_omc[6]
										tn_w_omc = metrics_mdl_omc[7]
										fp_w_omc = metrics_mdl_omc[8]
										# 3 : add them to a collector to get later stats on all seeds that have been ran
										ol1_acc_col_omc_for_1seed.append(acc_w_omc)
										ol1_MCC_col_omc_for_1seed.append(mcc_w_omc)	      # P2U
										ol1_PREC_col_omc_for_1seed.append(precision_w_omc)	  # P6U
										ol1_RECALL_col_omc_for_1seed.append(recall_w_omc) # P7U
										ol1_fpr_col_omc_for_1seed.append(fpr_w_omc) #-----------------
										ol1_TP_col_omc_for_1seed.append(tp_w_omc) # P9U #--------ol1_F1_col_omc_for_1seed --> ol1_TP_col_omc_for_1seed
										ol1_FN_col_omc_for_1seed.append(fn_w_omc) #--------ol1_support_col_omc_for_1seed --> ol1_FN_col_omc_for_1seed
										ol1_TN_col_omc_for_1seed.append(tn_w_omc) # P10U #--------ol1_num_tests_w_outcome_as_class_pos_col_omc_for_1seed --> ol1_TN_col_omc_for_1seed
										ol1_FP_col_omc_for_1seed.append(fp_w_omc) #--------create --> ol1_FP_col_omc_for_1seed
										# 4 : make the calculations and plots needed for auc and roc curves (the AUC value for the OMC mdl for this seed is catched at the same time)
										roc_auc_w_omc = roc_curve_updater_after_one_iteration_of_the_mdl(array_of_omc_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data, array_of_omc_mdl_predictions_2probs_on_all_data, encoded_classes, mean_fpr_by_seed_omc_mdl, tprs_col_by_seed_omc_mdl, aucs_col_by_seed_omc_mdl, aseed, ax1)
										#...add this value later in the report after remoling the columns

										# ~~~~~~>(FOR THE ALLFTS MODEL)
										print(bcolors.OKGREEN + " > - ## OL1 is still at seed", aseed, ": Metrics of the Allfts model on this one seed alone..." + bcolors.ENDC)
										# 1: Create the requirements...
										# ~~~~~(finishing touches for outer loop 1 collectors)-A : it is about stocking the complexity of this model and it  is not an output because evident (" of all fts) !
										# ~~~~~(finishing touches for outer loop 1 collectors)-B-allfts model
										#### lets make for the 2 models (omc and allfts) the df of the preds called collected to ready it for their mcc calculations
										df_from_ol2_col_pred_call_1by_seed_w_allfts = pd.DataFrame(ol2_col_pred_call_1by_seed_w_allfts)
										# lets rename the columns properly to what they correspond
										edited_colnames_for_ol2_col_pred_call_1by_seed_w_allfts = ["Sample_index"] + ["Allfts_called_preds"]
										df_from_ol2_col_pred_call_1by_seed_w_allfts.columns = edited_colnames_for_ol2_col_pred_call_1by_seed_w_allfts
										# on the full dataset LOO folds, to calculate the MCC and others metrics for the OMC, make a df of the mdl corresponding preds and the responses...
										frame_of_allfts_mdl_called_predictions_on_all_data = df_from_ol2_col_pred_call_1by_seed_w_allfts.iloc[:, [1]]
										frame_of_allfts_mdl_implicated_samples_observations_on_all_data = dataBin.loc[df_from_ol2_col_pred_call_1by_seed_w_allfts["Sample_index"].tolist(), :]
										# ~~~~~(finishing touches for outer loop 1 collectors)-C-allfts models
										# lets report the predictions with each seed in a file from the temp dataframe ## no need to write, juste make df of it # used for the auc computations and roc curve plot
										df_from_print_pred_all_col = pd.concat(print_pred_all_col)
										all_seeds_col_of_df_from_print_pred_all_col.append(df_from_print_pred_all_col) # stock it from all seeds version creation
										# on the full dataset LOO folds, to calculate the AUC and make the roc curve, make 2 arrays of the mdl corresponding probs of preds and the responses...
										array_of_allfts_mdl_predictions_2probs_on_all_data = np.array(df_from_print_pred_all_col.loc[:, [df_from_print_pred_all_col.columns[4]]])
										array_of_allfts_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data = np.array(dataBin.loc[df_from_print_pred_all_col[df_from_print_pred_all_col.columns[2]].tolist(), :])
										# 2 : ...calculate the 8 metrics...
										metrics_mdl_allfts = pd_ml_classif_report_on_cm_binary(frame_of_allfts_mdl_implicated_samples_observations_on_all_data, frame_of_allfts_mdl_called_predictions_on_all_data,binary_classes_le)
										acc_w_allfts = metrics_mdl_allfts[0]
										mcc_w_allfts = metrics_mdl_allfts[1]
										precision_w_allfts = metrics_mdl_allfts[2]
										recall_w_allfts = metrics_mdl_allfts[3]
										fpr_w_allfts = metrics_mdl_allfts[4]
										tp_w_allfts = metrics_mdl_allfts[5]
										fn_w_allfts = metrics_mdl_allfts[6]
										tn_w_allfts = metrics_mdl_allfts[7]
										fp_w_allfts = metrics_mdl_allfts[8]
										# 3 : add them to a collector to get later stats on all seeds that have been ran
										ol1_acc_col_allfts_for_1seed.append(acc_w_allfts)
										ol1_MCC_col_allfts_for_1seed.append(mcc_w_allfts)  # P2U
										ol1_PREC_col_allfts_for_1seed.append(precision_w_allfts)  # P6U
										ol1_RECALL_col_allfts_for_1seed.append(recall_w_allfts)  # P7U
										ol1_fpr_col_allfts_for_1seed.append(fpr_w_allfts)
										ol1_TP_col_allfts_for_1seed.append(tp_w_allfts)  # P9U # ol1_F1_col_allfts_for_1seed ---> ol1_TP_col_allfts_for_1seed
										ol1_FN_col_allfts_for_1seed.append(fn_w_allfts) # ol1_F1_col_allfts_for_1seed ---> ol1_FN_col_allfts_for_1seed
										ol1_TN_col_allfts_for_1seed.append(tn_w_allfts)  # P10U # ol1_F1_col_allfts_for_1seed ---> ol1_TN_col_allfts_for_1seed
										ol1_FP_col_allfts_for_1seed.append(fp_w_allfts)  # P10U # create ---> ol1_FP_col_allfts_for_1seed
										# 4 : make the calculations and plots needed for auc and roc curves (the AUC value for the Allfts mdl for this seed is catched at the same time)
										roc_auc_w_allfts = roc_curve_updater_after_one_iteration_of_the_mdl(array_of_allfts_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data, array_of_allfts_mdl_predictions_2probs_on_all_data, encoded_classes, mean_fpr_by_seed_allfts_mdl, tprs_col_by_seed_allfts_mdl, aucs_col_by_seed_allfts_mdl, aseed, ax2)
										# ...add this value later in the report after remoling the columns
										# all metrics for a seed are computed
										# ...lets check the time this one seed analysis took as it will be part of the report of each
										runtime_case_for_one_seed = duration_from(a_classif_seeds_start)
										# ...lets report them for the omc model...
										print(bcolors.OKGREEN + " > - ### OL1 is still at seed", aseed, ": Reporting metrics of OMC mode in respective dataframe of results, for this one seed alone..." + bcolors.ENDC)
										index_line_to_write_in_for_omc_mdl_one_seed_1 = len(df_of_results_omc_mdl)
										df_of_results_omc_mdl.loc[index_line_to_write_in_for_omc_mdl_one_seed_1] = [ctype,tag_drugname,tag_profilename,
											len(dframe),trainingset_size_the_biggest,
											model_complexities_explored_report,
											aseed,str(runtime_case_for_one_seed),
											int(ol1_col_med_ol2_col_omc[classif_seeds.index(aseed)]),
											acc_w_omc,
											mcc_w_omc,
											roc_auc_w_omc,
											precision_w_omc,
											recall_w_omc,
											fpr_w_omc,
											tp_w_omc,
											fn_w_omc,
											tn_w_omc,
											fp_w_omc]
										#add this line also to a prebuilt of the df that will compare models
										index_line_to_write_in_for_omc_mdl_one_seed_2 = len(df_of_results_omcmdl_vs_allftsmdl_omc_info)
										df_of_results_omcmdl_vs_allftsmdl_omc_info.loc[index_line_to_write_in_for_omc_mdl_one_seed_2] = [ctype, tag_drugname, tag_profilename,
											len(dframe), trainingset_size_the_biggest,
											models_compared[0],
											model_complexities_explored_report,
											aseed, str(runtime_case_for_one_seed),
											int(ol1_col_med_ol2_col_omc[classif_seeds.index(aseed)]),
											acc_w_omc,
											mcc_w_omc,
											roc_auc_w_omc,
											precision_w_omc,
											recall_w_omc,
											fpr_w_omc,
											tp_w_omc,
											fn_w_omc,
											tn_w_omc,
											fp_w_omc]
										# ...lets report them for the allfts model...
										print(bcolors.OKGREEN + " > - ### OL1 is still at seed", aseed, ": Reporting metrics of Allfts model in respective dataframe of results, for this one seed alone..." + bcolors.ENDC)
										index_line_to_write_in_for_allfts_mdl_one_seed_1 = len(df_of_results_allfts_mdl)
										df_of_results_allfts_mdl.loc[index_line_to_write_in_for_allfts_mdl_one_seed_1] = [ctype,tag_drugname,tag_profilename,
											len(dframe), trainingset_size_the_biggest,
											num_all_features,
											aseed, str(runtime_case_for_one_seed),
											"NA",
											acc_w_allfts,
											mcc_w_allfts,
											roc_auc_w_allfts,
											precision_w_allfts,
											recall_w_allfts,
											fpr_w_allfts,
											tp_w_allfts,
											fn_w_allfts,
											tn_w_allfts,
											fp_w_allfts]
										# add this line also to a prebuilt of the df that will compare models
										index_line_to_write_in_for_allfts_mdl_one_seed_2 = len(df_of_results_omcmdl_vs_allftsmdl_allfts_info)
										df_of_results_omcmdl_vs_allftsmdl_allfts_info.loc[index_line_to_write_in_for_allfts_mdl_one_seed_2] = [ctype,tag_drugname,tag_profilename,
											len(dframe), trainingset_size_the_biggest,
											models_compared[1],
											num_all_features,
											aseed, str(runtime_case_for_one_seed),
											"NA",
											acc_w_allfts,
											mcc_w_allfts,
											roc_auc_w_allfts,
											precision_w_allfts,
											recall_w_allfts,
											fpr_w_allfts,
											tp_w_allfts,
											fn_w_allfts,
											tn_w_allfts,
											fp_w_allfts]
										# ...all metrics for a seed are reported
										# >>>>>>>>>>>>>>>>>END OF METRICS FOR EACH SEED
									# end of outer loop 1 (on the seeds)
									print(bcolors.OKGREEN + " < END OF OUTER LOOP 1 (OL1): loop on all", len(classif_seeds), "seeds..." + bcolors.ENDC)
									# >>>>>>>>METRICS BASED ON ALL SEEDS
									# for the omc mdl
									print(bcolors.OKGREEN + " <~~~ METRICS OF THE OMC MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									ol1_col_med_ol2_col_omc_made_med_omc_allseeds = int(np.nanmedian(ol1_col_med_ol2_col_omc)) # a list of the OMC median at each seed; lets make a median of it for a value for all seeds # put in int() to give it more sense
									acc_med_omc_for_allseeds = np.nanmedian(ol1_acc_col_omc_for_1seed)
									MCC_med_omc_for_allseeds = np.nanmedian(ol1_MCC_col_omc_for_1seed)  # P3I,P3U,P3F (for each seed there is one ol1_MCC_col_omc_for_1seed and its for the omc)
									PREC_med_omc_for_allseeds = np.nanmedian(ol1_PREC_col_omc_for_1seed)  # P6F (for each seed, the next 5 metrics only exist for the OMC model. ## do it the all_fts also : no need, its the random but still ask)
									RECALL_med_omc_for_allseeds = np.nanmedian(ol1_RECALL_col_omc_for_1seed)  # P7F
									fpr_med_omc_for_allseeds = np.nanmedian(ol1_fpr_col_omc_for_1seed)  # P8F
									TP_med_omc_for_allseeds = int(np.nanmedian(ol1_TP_col_omc_for_1seed))  # P9F
									FN_med_omc_for_allseeds = int(np.nanmedian(ol1_FN_col_omc_for_1seed)) # put in it() to give it more sense
									TN_med_omc_for_allseeds = int(np.nanmedian(ol1_TN_col_omc_for_1seed)) # put in it() to give it more sense
									FP_med_omc_for_allseeds = int(np.nanmedian(ol1_FP_col_omc_for_1seed))  # put in it() to give it more sense
									#...and finish the roc curve and while doing that computations done will give us the mean auc (catched at the same time)
									mean_auc_from_cols_of_omc_mdl, std_auc_from_aucs_col_of_omc_mdl = roc_curve_finisher_after_all_iterations_of_the_mdl(figure_omc_mdl, ax1, ax3, tprs_col_by_seed_omc_mdl, mean_fpr_by_seed_omc_mdl, aucs_col_by_seed_omc_mdl, basedir, tag_task_type, tag_alg, models_compared[0], tag_ctype, tag_drugname, tag_profilename, tag_num_trial)
									Mean_ROC_AUC_omc_for_allseeds = ufloat(mean_auc_from_cols_of_omc_mdl, std_auc_from_aucs_col_of_omc_mdl)
									# ...add this value (and it s std) later in the report after remoling the columns
									# for the allfts mdl
									print(bcolors.OKGREEN + " <~~~ METRICS OF THE ALLFTS MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									acc_med_allfts_for_allseeds = np.nanmedian(ol1_acc_col_allfts_for_1seed)
									MCC_med_allfts_for_allseeds = np.nanmedian(ol1_MCC_col_allfts_for_1seed)  # P3I,P3U,P3F (for each seed there is one ol1_MCC_col_omc_for_1seed and its for the omc)
									PREC_med_allfts_for_allseeds = np.nanmedian(ol1_PREC_col_allfts_for_1seed)  # P6F (for each seed, the next 5 metrics only exist for the OMC model. ## do it the all_fts also : no need, its the random but still ask)
									RECALL_med_allfts_for_allseeds = np.nanmedian(ol1_RECALL_col_allfts_for_1seed)  # P7F
									fpr_med_allfts_for_allseeds = np.nanmedian(ol1_fpr_col_allfts_for_1seed)  # P8F
									TP_med_allfts_for_allseeds = int(np.nanmedian(ol1_TP_col_allfts_for_1seed))  # P9F
									FN_med_allfts_for_allseeds = int(np.nanmedian(ol1_FN_col_allfts_for_1seed)) # put in it() to give it more sense
									TN_med_allfts_for_allseeds = int(np.nanmedian(ol1_TN_col_allfts_for_1seed)) # put in it() to give it more sense
									FP_med_allfts_for_allseeds = int(np.nanmedian(ol1_FP_col_allfts_for_1seed))  # put in it() to give it more sense
									# ...and finish the roc curve and while doing that computations done will give us the mean auc (catched at the same time)
									mean_auc_from_cols_of_allfts_mdl, std_auc_from_aucs_col_of_allfts_mdl = roc_curve_finisher_after_all_iterations_of_the_mdl(figure_allfts_mdl, ax2, ax3, tprs_col_by_seed_allfts_mdl, mean_fpr_by_seed_allfts_mdl, aucs_col_by_seed_allfts_mdl, basedir, tag_task_type, tag_alg, models_compared[1], tag_ctype, tag_drugname, tag_profilename, tag_num_trial)
									Mean_ROC_AUC_allfts_for_allseeds = ufloat(mean_auc_from_cols_of_allfts_mdl, std_auc_from_aucs_col_of_allfts_mdl)
									# ...add this value (and it s std) later in the report after remoling the columns
									# for both models, after the last average roc curve is plotted, we can finish the separate plot to compare only the average roc curve of both models
									average_roc_curve_finisher(ax3,figure_mdl_vs_mdl,models_compared,tag_task_type,tag_alg,tag_ctype,tag_drugname,tag_profilename,tag_num_trial,basedir)
									# lets check the time all seeds analysis took as it will be part of the report
									runtime_case_for_all_seeds = duration_from(all_classif_seeds_start)
									# lets report the metrics values averaged in any fashion for all seeeds (for the omc mdl)
									print(bcolors.OKGREEN + " <~~~ REPORTING METRICS OF OMC MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									index_line_to_write_in_for_omc_mdl_all_seeds_1 = len(df_of_results_omc_mdl)
									df_of_results_omc_mdl.loc[index_line_to_write_in_for_omc_mdl_all_seeds_1] = [ctype,tag_drugname,tag_profilename,
										len(dframe), trainingset_size_the_biggest,
										model_complexities_explored_report,
										str(len(classif_seeds))+" seeds", str(runtime_case_for_all_seeds),
										ol1_col_med_ol2_col_omc_made_med_omc_allseeds,
										acc_med_omc_for_allseeds,
										MCC_med_omc_for_allseeds,
										Mean_ROC_AUC_omc_for_allseeds,
										PREC_med_omc_for_allseeds,
										RECALL_med_omc_for_allseeds,
										fpr_med_omc_for_allseeds,
										TP_med_omc_for_allseeds,
										FN_med_omc_for_allseeds,
										TN_med_omc_for_allseeds,
										FP_med_omc_for_allseeds]
									# lets report the metrics values averaged in any fashion for all seeeds (for the allfts mdl)
									print(bcolors.OKGREEN + " <~~~ REPORTING METRICS OF ALLFTS MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									index_line_to_write_in_for_omc_mdl_all_seeds_1 = len(df_of_results_allfts_mdl)
									df_of_results_allfts_mdl.loc[index_line_to_write_in_for_omc_mdl_all_seeds_1] = [ctype, tag_drugname, tag_profilename,
										len(dframe), trainingset_size_the_biggest,
										num_all_features,
										str(len(classif_seeds))+" seeds", str(runtime_case_for_all_seeds),
										"NA",
										acc_med_allfts_for_allseeds,
										MCC_med_allfts_for_allseeds,
										Mean_ROC_AUC_allfts_for_allseeds,
										PREC_med_allfts_for_allseeds,
										RECALL_med_allfts_for_allseeds,
										fpr_med_allfts_for_allseeds,
										TP_med_allfts_for_allseeds,
										FN_med_allfts_for_allseeds,
										TN_med_allfts_for_allseeds,
										FP_med_allfts_for_allseeds]
									##--finish building df comparing both mdls from 2 pre-builts dfs with info on each model
									df_of_results_omcmdl_vs_allftsmdl = pd.concat([df_of_results_omcmdl_vs_allftsmdl_omc_info,df_of_results_omcmdl_vs_allftsmdl_allfts_info])
									# add a line for all seeds metrics obtained for omc model
									index_line_to_write_in_for_omc_mdl_all_seeds_2 = len(df_of_results_omcmdl_vs_allftsmdl)
									df_of_results_omcmdl_vs_allftsmdl.loc[index_line_to_write_in_for_omc_mdl_all_seeds_2] = [ctype, tag_drugname, tag_profilename,
										len(dframe), trainingset_size_the_biggest,
										models_compared[0],
										model_complexities_explored_report,
										str(len(classif_seeds))+" seeds", str(runtime_case_for_all_seeds),
										ol1_col_med_ol2_col_omc_made_med_omc_allseeds,
										acc_med_omc_for_allseeds,
										MCC_med_omc_for_allseeds,
										Mean_ROC_AUC_omc_for_allseeds,
										PREC_med_omc_for_allseeds,
										RECALL_med_omc_for_allseeds,
										fpr_med_omc_for_allseeds,
										TP_med_omc_for_allseeds,
										FN_med_omc_for_allseeds,
										TN_med_omc_for_allseeds,
										FP_med_omc_for_allseeds]
									# add a line for all seeds metrics obtained for allfts model
									index_line_to_write_in_for_omc_mdl_all_seeds_2 = len(df_of_results_omcmdl_vs_allftsmdl)
									df_of_results_omcmdl_vs_allftsmdl.loc[index_line_to_write_in_for_omc_mdl_all_seeds_2] = [ctype, tag_drugname, tag_profilename,
										len(dframe), trainingset_size_the_biggest,
										models_compared[1],
										num_all_features,
										str(len(classif_seeds))+" seeds", str(runtime_case_for_all_seeds),
										"NA",
										acc_med_allfts_for_allseeds,
										MCC_med_allfts_for_allseeds,
										Mean_ROC_AUC_allfts_for_allseeds,
										PREC_med_allfts_for_allseeds,
										RECALL_med_allfts_for_allseeds,
										fpr_med_allfts_for_allseeds,
										TP_med_allfts_for_allseeds,
										FN_med_allfts_for_allseeds,
										TN_med_allfts_for_allseeds,
										FP_med_allfts_for_allseeds]
									# ...all metrics for all seeds are reported
									# lets concatenate for each model the tables  that helped us to keep track of predictions probabilities and the class called each time
									df_to_keep_track_of_preds_probs_and_class_called_omc_mdl = pd.concat(all_seeds_col_of_df_from_print_pred_col)
									df_to_keep_track_of_preds_probs_and_class_called_allfts_mdl = pd.concat(all_seeds_col_of_df_from_print_pred_all_col)
									# all infos for all seeds are reported
								else : ## exception of data sparsity
									# put here the last for adding only NA in place of the values
									# closing condittion on class sparsity in data defined so no analysis and empty results (NAs)
									print(bcolors.HEADER + " W : Class sparsity declared with : ",dataBin.iloc[:, 0].value_counts()[encoded_classes[0]],encoded_classes[0],"and",dataBin.iloc[:, 0].value_counts()[encoded_classes[1]],encoded_classes[1] + bcolors.ENDC)
									print(bcolors.HEADER + " W : To analyse, augment your unsuffisant classes population size or lower your limit of declaring class sparsity." + bcolors.ENDC)
									index_line_to_write_in_df_of_results_omc_mdl = len(df_of_results_omc_mdl)
									df_of_results_omc_mdl.loc[index_line_to_write_in_df_of_results_omc_mdl] = [tag_ctype,tag_drugname,tag_profilename,len(dframe)]+["Class_sparsity!!!"] * 15
									index_line_to_write_in_df_of_results_allfts_mdl = len(df_of_results_allfts_mdl)
									df_of_results_allfts_mdl.loc[index_line_to_write_in_df_of_results_allfts_mdl] = [tag_ctype, tag_drugname, tag_profilename, len(dframe)] + ["Class_sparsity!!!"] * 15
									index_line_to_write_in_df_of_results_omcmdl_vs_allftsmdl = len(df_of_results_omcmdl_vs_allftsmdl)
									df_of_results_omcmdl_vs_allftsmdl.loc[index_line_to_write_in_df_of_results_omcmdl_vs_allftsmdl] = [tag_ctype, tag_drugname, tag_profilename, len(dframe)] + ["Class_sparsity!!!"] * 16
									# runtime_case_for_one_seed not needed here because nothing analysed
									# df_of_results.loc[index_line_to_write_in] = [str(runtime_case_for_one_seed), tag_alg, tag_alg_mark, tag_num_trial, ctype, drug, tag_drugname, featuretype, feature_val_type, len(dframe)] + ["NA"] * 23
									print(bcolors.HEADER + "W : Empty line added to result files" + bcolors.ENDC)
								#### lets get in a csv copy of the df_of_results. the output filename should identify the exact trial
								# lets make up a filename and then use it to create the .csv files
								## for the metrics values
								output_filename_for_omc_mdl = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_MetricsValues.csv"
								df_of_results_omc_mdl.to_csv(output_filename_for_omc_mdl, index=None, header=True)
								output_filename_for_allfts_mdl = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[1] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_MetricsValues.csv"
								df_of_results_allfts_mdl.to_csv(output_filename_for_allfts_mdl, index=None, header=True)
								output_filename_for_comparing_mdls = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0]+"vs"+models_compared[1] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_MetricsValues.csv"
								df_of_results_omcmdl_vs_allftsmdl.to_csv(output_filename_for_comparing_mdls, index=None, header=True)
								## for the table that helped us keep track of the predictions probabilities and the class called
								output_filename_for_omc_mdl_preds_probs_and_class_called = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_Predictions.csv"
								df_to_keep_track_of_preds_probs_and_class_called_omc_mdl.to_csv(output_filename_for_omc_mdl_preds_probs_and_class_called, index=None, header=True)
								output_filename_for_allfts_mdl_preds_probs_and_class_called = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[1] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_Predictions.csv"
								df_to_keep_track_of_preds_probs_and_class_called_allfts_mdl.to_csv(output_filename_for_allfts_mdl_preds_probs_and_class_called, index=None, header=True)
							else :
								print(bcolors.HEADER + " W : In classification, no Metrics Estimations study request has been detected. To do so, please involve the proper arguments in command line." + bcolors.ENDC)
							# end of metrics of classification step estimations<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

							# >>>>>>>>>>>>>>>>>>>>>>>>>FEATURE SELECTION ESTIMATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
							if ("C-FSB" in our_args.Classif_studies) | ("Both" in our_args.Classif_studies):
								print(">>>>>>>>>>>>>>>>>>>>>>>>FS in developpement<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
								# >>>>>>>>>>>>>>>>CREATING THE DF TO RECEIVE THE RESULTS OF METRICS ESTIMATIONS
								df_of_results_FS_omc_mdl = df_of_results_for_FS_one_mdl_creator()
								# is built during computations if no class sparsity but is initialised anyway because need to fill in case of class sparsity error where computations dont happen
								print(bcolors.BOLD + "Result file created for this subcase analysis, each line is a seed and last is all seeds averaging. Contains : 0 lines" + bcolors.ENDC)
								# >>>>>>>>>>CHECKING DATA SPARSITY
								if (dataBin.iloc[:, 0].value_counts()[encoded_classes[0]] >= classif_msn_by_class) & (dataBin.iloc[:, 0].value_counts()[encoded_classes[1]] >= classif_msn_by_class):
									print(bcolors.WARNING + "Data sparsity on classes checked. No data issue met. Analysis ongoing..." + bcolors.ENDC)
									# >>>>>>>>>>>> PREPAPRING FOR OUTER LOOP 1
									print(" PREPARING FOR OUTER LOOP 1 : creation of metrics collectors by seed")
									#---- create the vectors to collect the metrics of performance for all seeds
									# collectors of the tables to keep track of the features ranked with their pval of the univariate test
									all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed = []
									all_seeds_col_of_list_fts_correlated_enough_to_response = []
									# collectors related to omc
									ol2_col_of_omc_il1 = []  # each il1 loop will get 1 OMC and its mcc; you have to catch it in each iteration of ol2 and get an average on the whole lot (eg median) later
									ol2_col_of_omc_il1_mcc = []
									ol2_col_of_topfeats_corresponding_to_omc_il1 = []  # each il1 loop will get 1 OMC and a list of its topfeats; you have to catch it in each iteration of ol2 and get an average (the persistent fts) on the whole lot (eg median) later
									# ===============> START OF OUTER LOOP 1
									# if an analysis use a ML alg exploiting a random seed, multiples runs can be made to see how changing the seed affects the robustness of the analysis
									# that seed has to be initialised before the analysis
									# this is a loop on the seed values (10 seeds with the zero being 5678 and the rest is 1 to 9 values)
									# old place of seeds lists to chosen one from
									all_classif_seeds_start = timer_started()  # start a clock to get the time after alls seeds have been used
									print(bcolors.OKGREEN + " > START OF OUTER LOOP 2 (OL2): loop on all", len(classif_seeds), "seeds..." + bcolors.ENDC)
									for aseed in classif_seeds:  # loop on the seeds each is n #outer loop 1 # aseed = 0 for testing
										a_classif_seeds_start = timer_started()  # start a clock to get the time after one seed use
										print(bcolors.OKGREEN + " > - OL1's seed", aseed, "has started being used..." + bcolors.ENDC)
										np.random.seed(aseed)  # fixate the actual seed for any random actions that have to vary
										separate_value_for_short_random_actions = classif_seeds[0]
										# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>C-FSB START
										# ==============================================TOWARDS AN OMC MODEL=======================================										#
										### IL1 will be multiproccessed, each fold a process using Process
										# Step 1 : initialise the needed parameters
										OMC_quest_allfts_vs_MCs_wide_col = []  # intialise a big list to put in it all the allfts anc omc predictions called to compute mcc on it and compare the (MCs,allfts) models

										## get the data divided in fts and response first for the need on later use ranking and FS by a metric of choice
										ol2_train_x = dframe[list(dframe)[5:]]  # keep only the features (index 1 to ncol) #here are dropped the model and the response #odd code tho because _c(i,j) drops cols i to j
										ol2_train_y = dframe.loc[:, [Resp_col_name]]  # keep the training set response column values (37 values) (for the row j of data analysed, the 1st value)

										# >>>>>>>>>>>GET LIST OF FOLDS FOR CV
										il1_folds = stratKfolds_making(classif_CV_folds_number, aseed, dframe, index_starting_fts_cols,Resp_col_name, True)
										# il1_folds = dframe.index  # create the folds for the LOO for the omc model
										len_il1_folds = len(il1_folds)
										# l = 1 # testing
										# il1_folds = [1,2,3] #testing
										# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
										print(bcolors.OKBLUE + " >>> START OF INNER LOOP 1 (IL1): LOOCV on OL2 all training data. Going for", len_il1_folds, "folds" + bcolors.ENDC)
										print(bcolors.OKBLUE + " >>> With each fold, training and prediction with many MCs and thus determining later OMC on one IL1 loop" + bcolors.ENDC)
										# for l in il1_folds: # this was the start of  # inner loop 1 #parallelise 37 jobs on each of on 37 folds/samples # the packages necessary are given # the results are combined in row after row fashion to form a matrix #and that in order of the samples

										# Step 2 : describe the function that each process will launch when it will accept a job in a queue
										# bench_of_results = [] # for testing # a collector for the result of each fold carried out to sort later and make an ordered df

										# fixate a seed for all this xproc part so that variactions dont happen when the env is copied for each process
										np.random.seed(separate_value_for_short_random_actions)
										def il1_job(respective_fold_tools): # respective_fold_id = 3 for testing
											print(bcolors.OKBLUE + " >>> - Task running IL1's fold", respective_fold_tools[0], "has started being carried out..." + bcolors.ENDC)
											print(bcolors.OKBLUE + " >>> - Ranking the features in the fold to get the FS for every MCs to estimate next" + bcolors.ENDC)
											# >>>>>>>>>>>>PRE-BUILDING THE SETS NEEDED
											# ---1-making of trainframe and valframe
											il1_train_data = dframe.loc[respective_fold_tools[1],:]  # Set the training set of 36 (by dropping one row/sample)
											il1_test_data = dframe.loc[respective_fold_tools[2],:]  # Set the validation set of 1 (by keeping only that previously dropped sample)
											# trainframe_y & valframe_y
											il1_train_y = il1_train_data.loc[:, [Resp_col_name]]  # train_y # (needed to train)
											# il1_test_y = il1_test_data.loc[:, [Resp_col_name]]  # val_y # val_y (not needed because prediction estimation uses the resp_col_nme of the dataset directly restricted to the indexes implicated)
											# trainframe_x & valframe_x
											il1_train_x = il1_train_data[list(il1_train_data)[5:]]  # train_x # (needed to train) (## should not transmit to self but to new name)
											il1_test_x = il1_test_data[list(il1_test_data)[5:]]  # val_x # (needed to predict)
											# <<<<<<<<<<<<<<END OF PRE-BUILDING THE SETS NEEDED
											if classif_omc_search_type == "OMC":
												# ~~~~~~~~~~~~~~~~~~~~~ optional part (depending on omc search chosen)
												# >>>>>>>>>>>>>>START OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
												# ---2-estimate the fts with two-sided fisher exact test p-values or two sided unpaired t-test p-values
												# we are about to train a model so lets re set the seed at its rightful place in case a function from a module changed it
												np.random.seed(aseed)
												# *******trying env specific loads
												if tag_alg == "DNN":
													# env_specific_loads_for_DNN(aseed)
													il1_pred_2probs_w_allfts = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, il1_train_x, il1_train_y, feature_val_type, binary_classes_le, il1_test_x, "prob", aseed)
												elif tag_alg == "SVM":
													# env_specific_loads_for_DNN(aseed)
													il1_pred_2probs_w_allfts = classifier_as_SVM_intro_train_pred(tag_alg_mark, il1_train_x, il1_train_y, feature_val_type, binary_classes_le, il1_test_x, "prob", aseed)
												else:
													# ---introduce the model
													rf1 = classifier_introduction(tag_alg,tag_alg_mark, il1_train_x, il1_train_y,aseed,encoded_classes,max_cores_minus2)  ## class weighting with classwt #to be check if not inversed
													# ---train the model on train_x,train_y
													rf1_fitted = classifier_model_training(rf1, il1_train_x, il1_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
													# -- mk test for each fold
													il1_pred_2probs_w_allfts = classifier_model_prediction(rf1_fitted, il1_test_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
												# *******
												il1_pred_call_w_allfts = []
												il1_pred_call_w_allfts = prediction_calling(il1_pred_2probs_w_allfts, il1_pred_call_w_allfts, classif_thres, encoded_classes, separate_value_for_short_random_actions)
												##maybe add this!!! ---adding raw predictions to a keep tracker of predictions in folds
												##maybe add this!!! ---adding the call to a collector for stats or metrics
												# <<<<<<<<<<<<<<END OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
												# ~~~~~~~~~~~~~~~~~~~~~ end of optional part (depending on omc search chosen)
											# >>>>>>>>>>>>>>ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
											# ---process of limiting training data to only fts who shows variation (at least one variation in a sample)
											# il1_train_x = eliminate_non_variable_fts(il1_train_x) # no need bbcuz we manage the exception
											# ---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (a vector of fts's p values in the order of the fts)
											il1_fts_ranking = ranker_by_pval_v2(il1_train_x, il1_train_y, feature_val_type, Resp_col_name,encoded_classes)
											# <<<<<<<<<<<END OF ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
											# >>>>>>>>>>>MCs PREDICTIONS MAKING
											# ---3-training a model for each MC and test it to compare them and take the best OMC
											il2_col_il2_pred_call_mc = []  # a collector of prediction called at each loop on complxities
											# mc = 2 # trial
											# print "- Estimating the MCs"
											print(bcolors.FAIL + " >>>> START OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_tools[0], "training data. Going for", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
											for mc in list(range(2, (max_complexity_of_tested_MCs + 1))):  # inner loop 2 # loop on the MCs
												print(bcolors.FAIL + " >>>> - In Fold", respective_fold_tools[0], ", MC of", mc, "features has started being made and estimated..." + bcolors.ENDC)
												topfeats = il1_fts_ranking[1][0:mc]
												## isolate as ranking function for p_value
												# rank it, keep this rank from original, get the names, get the top MC or those names) ## try to gives the rank at end of estimating fts to select in it directly here
												# set up a training of the built with elected fts MC model
												il2_train_mc_x = il1_train_x.loc[:, topfeats]  # train_x restricted to MC fts
												il2_test_mc_x = il1_test_x.loc[:, topfeats]  # test_x restricted to MC fts
												np.random.seed(aseed)
												# *******trying env specific loads
												if tag_alg == "DNN":
													# env_specific_loads_for_DNN(aseed)
													il2_pred_2probs_w_mc = classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark, il2_train_mc_x, il1_train_y, feature_val_type, binary_classes_le, il2_test_mc_x, "prob", aseed)
												elif tag_alg == "SVM":
													# env_specific_loads_for_DNN(aseed)
													il2_pred_2probs_w_mc = classifier_as_SVM_intro_train_pred(tag_alg_mark, il2_train_mc_x, il1_train_y, feature_val_type, binary_classes_le, il2_test_mc_x, "prob", aseed)
												else:
													# ---introduce the model
													rf2 = classifier_introduction(tag_alg,tag_alg_mark, il2_train_mc_x, il1_train_y,aseed,encoded_classes,max_cores_minus2)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
													# ---train the model on train_x,train_y
													rf2_fitted = classifier_model_training(rf2, il2_train_mc_x, il1_train_y) ## tag_alg,tag_alg_mark,featuretype,classif_list_cat_fts,aseed,binary_classes_le
													# -- mk test for each fold
													il2_pred_2probs_w_mc = classifier_model_prediction(rf2_fitted, il2_test_mc_x, "prob") ## tag_alg,aseed,binary_classes_le # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
												# *******
												il2_pred_call_mc = []
												il2_pred_call_mc = prediction_calling(il2_pred_2probs_w_mc, il2_pred_call_mc, classif_thres, encoded_classes, separate_value_for_short_random_actions)
												il2_col_il2_pred_call_mc.append(il2_pred_call_mc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
											# ------end of inner loop2
											print(bcolors.FAIL + " <<<< END OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_tools[0], "training data. Gone through", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
											# make a df that shows all found for this fold of il1 including all in il2 mcs
											# OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame()  # the part of the OMC quest wide col for a fold
											the_fold_column_content = [respective_fold_tools[0]] * len(il1_test_data.index.tolist())  # col fold
											the_il1_test_data_indexes_column_content = il1_test_data.index.tolist()  # col indexes in il1_test_data
											# il1_pred_call_w_allfts # is the [preds called for allfts on il1 to compare with the mcs models prediction] column content
											# il2_pred_call_mc # is the [each mc preds called] column content and a lot are in store into il2_col_il2_pred_call_mc so use a * in front of it in the zip to extract its content at the same level
											if classif_omc_search_type == "OMC":
												list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, il1_pred_call_w_allfts, *il2_col_il2_pred_call_mc))  # old zipping coming with including allfts model in estimations of mcc for omc mdls # zip col1,col2,col3, and extrcted col4
											else:
												list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, *il2_col_il2_pred_call_mc))  # zip col1,col2, and extrcted col4
											OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame(list_of_all_cols_for_a_fold_zipped)  # init the part of the OMC quest wide col for a fold # make a df of it
											tuple_result = (tuple([respective_fold_tools[0], OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold]))  # keep a tuple of the fold id and the mini df produced (later this is used to put in the right order the results and make a full df of them)
											print(bcolors.OKBLUE + " <<<  - Task running Fold", respective_fold_tools[0], ",among", len_il1_folds, "folds, is done. Carried out by " + current_process().name + bcolors.ENDC)
											return tuple_result
										# ~~~~modes of batch runs (parallel or sequential)
										if alg_pw == "par":  # launching IL1 in a multiprocessing way
											il1_multiprocessing_handler(il1_folds, tag_num_xproc, OMC_quest_allfts_vs_MCs_wide_col, il1_job)
										else:  # launching IL1 in a sequential way
											il1_sequential_processing_handler(il1_folds, OMC_quest_allfts_vs_MCs_wide_col, il1_job)
										# the list OMC_quest_allfts_vs_MCs_wide_col is updated with values add it to concatenate later as a pd
										# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
										# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of tested portion
										# end of inner loop 1 (official end)
										print(bcolors.OKBLUE + " <<< END OF INNER LOOP 1 (IL1): LOOCV on OL2 present seed", aseed, ". IL1 counted", len_il1_folds, "folds" + bcolors.ENDC)
										# lets concatenate all dfs, one from each fold of inner loop 1 into one (parse on it later, restrict to fold, get mcc ie many mcc for many folds of a models s med.mcc)
										OMC_quest_allfts_vs_MCs_wide = pd.concat(OMC_quest_allfts_vs_MCs_wide_col)
										# lets rename the columns properly to what they correspond
										if classif_omc_search_type == "OMC":
											edited_colnames = ["Fold"] + ["Sample_index"] + [num_all_features] + list(range(2, (max_complexity_of_tested_MCs + 1))) # old naming cols coming with including allfts model in estimations of mcc for omc mdls
										else:
											edited_colnames = ["Fold"] + ["Sample_index"] + list(range(2, (max_complexity_of_tested_MCs + 1)))
										OMC_quest_allfts_vs_MCs_wide.columns = edited_colnames
										# <<<<<<<<<<<<END OF PREDICTIONS MAKING
										# >>>>>>>>>>>>METRICS computations FOR EACH SEED# ~~~~~~>(FOR THE OMC MODEL)
										print(bcolors.OKGREEN + " > - # OL2 is still at seed", aseed, ": Metrics of the OMC model on this one seed alone..." + bcolors.ENDC)
										# >>>>>>>>>>>ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
										print(bcolors.WARNING + " << - Choosing OMC among MCs of IL1" + bcolors.ENDC)
										# ------------Median MCCs calculations as the criteria to elect omc on
										dict_of_mcc_values_by_mdl_to_update = {}
										# to calculate MCCs, restrict the OMC_quest_wide to the mdl_column for the corresponding preds...
										frame_of_all_mdls_called_predictions = OMC_quest_allfts_vs_MCs_wide.iloc[:, 2:len(OMC_quest_allfts_vs_MCs_wide.columns)]
										# index_start_space_of_cols_as_mdls = 2 and endplus1_space_of_cols_as_mdls = len(endplus1_space_of_cols_as_mdls.columns)
										# space_of_cols_indexes_to_consider_as_mdls = 2: or if native = :
										# ...and mock up a same length response column from ol2_train_y with each row of col of indexes having the corresponding response
										frame_of_all_mdls_implicated_samples_observations = ol2_train_y.loc[OMC_quest_allfts_vs_MCs_wide["Sample_index"].tolist(), :]  ##!!! last_stop_check if the 2 last frames works after the columns renaming
										dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, frame_of_all_mdls_called_predictions, frame_of_all_mdls_implicated_samples_observations, Resp_col_name, encoded_classes)
										# ~~~~~~~~preceed with this restriction to use singular mcc calculation
										# for l in il1_folds:
										# 	restricted_to_samples_in_test = restriction_of_MCCs_wide(OMC_quest_allfts_vs_MCs_wide, l, il1_test_data.index.tolist())
										#   dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, restricted_to_samples_in_test, 2, len(restricted_to_samples_in_test.columns), il1_test_y, Resp_col_name, encoded_classes)
										# ~~~~~~~~
										# ---choosing the best mc....
										omc_il1, omc_il1_mcc = OMC_founder_in_dict_MCs_MCCs(dict_of_mcc_values_by_mdl_to_update)
										# and...keeping it away...
										ol2_col_of_omc_il1.append(omc_il1)  # 2 lists are used to also take the mcc with us alongside the omc until the end
										ol2_col_of_omc_il1_mcc.append(omc_il1_mcc)
										# ...contains the best_fts_mc at each seed of the outer loop2 ie 10 values
										# <<<<<<<<<<<<<<<END OF ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
										# >>>>>>>>>>>>>START OF FINDING OMC CORRESPONDING TOP FEATURES ON ALL DATA
										print(bcolors.WARNING + " << - Getting for OMC of IL1 its corresponding top feats on all data " + bcolors.ENDC)
										# ---process of estimating fts for fts ranking
										# feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (z dict of feat: p_val )
										ol2_fts_ranking = ranker_by_pval_v2(ol2_train_x, ol2_train_y, feature_val_type, Resp_col_name,encoded_classes)
										# ---get on all the data the prediction of the omc (the features that are corresponding to the omc following a ranking metric )
										if classif_omc_search_type == "OMC":
											model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + ", " + str(num_all_features) + "]"
											# restrict the material to build the metric not train and test here but the list of selected fts for the omc in the ranking
											topfeats_in_ol2_fts_ranking = "All " + str(num_all_features) + " features of the dataset"
											ol2_col_of_topfeats_corresponding_to_omc_il1.append(topfeats_in_ol2_fts_ranking)  # ...same as the omc, keeping topfts away for metrics reports
										else:
											model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + "]"
											# restrict the material to build the metric not train and test here but the list of selected fts for the omc in the ranking
											topfeats_in_ol2_fts_ranking = ol2_fts_ranking[1][0:omc_il1]
											ol2_col_of_topfeats_corresponding_to_omc_il1.append(topfeats_in_ol2_fts_ranking)  # ...same as the omc, keeping topfts away for metrics reports
											# -----lets try to document the fts that had good enough pvalue to be part of the FS but did not make the cut
											pval_last_feat_of_omc = ol2_fts_ranking[2][(omc_il1-1)]
											list_pvals_for_fts_correlated_enough_to_response = [a_pval for a_pval in ol2_fts_ranking[2] if a_pval <= pval_last_feat_of_omc]
											list_fts_correlated_enough_to_response = ol2_fts_ranking[1][:len(list_pvals_for_fts_correlated_enough_to_response)]
											# add the list of fts ranked to a collector to later compute the persistent accross seeds and the non persistent
											all_seeds_col_of_list_fts_correlated_enough_to_response.append(list_fts_correlated_enough_to_response)
											# - creating the df to receive ["Seed","pval","feat ranked"] 2nd col content before the 1st col because the 1st col uses it to be created
											df_of_fts_correlated_enough_to_response_by_seed = pd.DataFrame()
											content_Seed_column = np.repeat(aseed, len(list_pvals_for_fts_correlated_enough_to_response))  # the longest column is the indexes in the fold column so start from it
											content_Pvals_column = list_pvals_for_fts_correlated_enough_to_response # pretty straight forward as it is just the pvals
											content_FeatsRanked_column = list_fts_correlated_enough_to_response # it is the list of the feats but if it did not make the FS cut mark it as such
											for feat in content_FeatsRanked_column:
												if feat not in topfeats_in_ol2_fts_ranking:
													content_FeatsRanked_column[content_FeatsRanked_column.index(feat)] = ' '.join([feat, '(CutFromFS)'])
											# supplying the frame...
											df_of_fts_correlated_enough_to_response_by_seed["Seed"] = content_Seed_column
											df_of_fts_correlated_enough_to_response_by_seed["Features_ranked"] = content_FeatsRanked_column
											df_of_fts_correlated_enough_to_response_by_seed["Pval_of_univ_stat"] = content_Pvals_column
											all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed.append(df_of_fts_correlated_enough_to_response_by_seed)

										# from here on out, you cannot train and because you already used all the data
										# end of judging the feature ranking of the omc in all folds of the training data
										# >>>>>>>>>> END OF FINDING OMC CORRESPONDING TOP FEATURES ON ALL DATA
										print(bcolors.WARNING + " << END OF A SEED INTERATION OF OUTER LOOP 2 (OL2): LOOCV on all data ie on", len(il1_folds), "folds" + bcolors.ENDC)
										# all metrics for a seed are computed
										# ...lets check the time this one seed analysis took as it will be part of the report of each
										runtime_case_for_one_seed = duration_from(a_classif_seeds_start)
										# ...lets report them for the omc model...
										print(bcolors.OKGREEN + " > - ### OL2 is still at seed", aseed, ": Reporting corresponding features of OMC mode in respective dataframe of results, for this one seed alone..." + bcolors.ENDC)
										index_line_to_write_in_for_omc_mdl_one_seed_1 = len(df_of_results_FS_omc_mdl)
										df_of_results_FS_omc_mdl.loc[index_line_to_write_in_for_omc_mdl_one_seed_1] = [ctype, tag_drugname, tag_profilename,
											len(dframe), trainingset_size_the_biggest,
											model_complexities_explored_report,
											aseed, str(runtime_case_for_one_seed),
											ol2_col_of_omc_il1[classif_seeds.index(aseed)], ol2_col_of_omc_il1_mcc[classif_seeds.index(aseed)],
											ol2_col_of_topfeats_corresponding_to_omc_il1[classif_seeds.index(aseed)], "NA"]
									# ...all metrics for a seed are reported
									# >>>>>>>>>>>>>>>>>END OF METRICS FOR EACH SEED
									# end of outer loop 2 (on the seeds)
									print(bcolors.OKGREEN + " < END OF OUTER LOOP 2 (OL2): loop on all", len(classif_seeds), "seeds..." + bcolors.ENDC)
									# >>>>>>>>METRICS BASED ON ALL SEEDS
									# for the omc mdl
									print(bcolors.OKGREEN + " <~~~ TOP FEATURES OF THE OMC MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									# 1: Create the requirements...
									# ~~~~~(finishing touches for outer loop 1 collectors)-A
									# lets add a median of all the OMCs from OL2, one from each IL1
									med_ol2_col_omc = int(np.median(ol2_col_of_omc_il1))
									# lets get for the median mcc
									med_ol2_col_of_omc_il1_mcc = np.nanmedian(ol2_col_of_omc_il1_mcc)
									# next step is to dispatch the topfts lists collected as being the proper list corresponding because their omc correspond to omc median value...
									# ...between a list holding lists of persistent topfts by seed (s seeds ie s lists of persitent topfts) # we sort to have ordered list (for a more easier check of present and absent fts in the output)
									persistentin_ol2_col_of_topfeats_corresponding_to_omc_il1 = sorted(set.intersection(*[set(list_of_topfeats) for list_of_topfeats in ol2_col_of_topfeats_corresponding_to_omc_il1]))
									# ...and a list holding one list of non persistent topfts by seed (s seeds ie s lists of non persitent topfts) # also if repeats happens to exits in the flat list of all feats setdiff1d eliminates them
									flat_list_from_all_lists_of_topfeats_to_keep_from_ol2_col_of_topfeats_corresponding_to_omc_il1 = [feat for list_of_topfeats in ol2_col_of_topfeats_corresponding_to_omc_il1 for feat in list_of_topfeats]
									nonpersistentin_ol2_col_of_topfeats_corresponding_to_omc_il1 = sorted(np.setdiff1d(flat_list_from_all_lists_of_topfeats_to_keep_from_ol2_col_of_topfeats_corresponding_to_omc_il1, persistentin_ol2_col_of_topfeats_corresponding_to_omc_il1).tolist())
									# we sort to have ordered list (for a more easier check of present and absent fts in the output)
									# lets check the time all seeds analysis took as it will be part of the report
									runtime_case_for_all_seeds = duration_from(all_classif_seeds_start)
									# lets report the metrics values averaged in any fashion for all seeds (for the omc mdl)
									print(bcolors.OKGREEN + " <~~~ REPORTING TOP FEATURES OF THE OMC MODEL FOR ALL SEEDS..." + bcolors.ENDC)
									index_line_to_write_in_for_omc_mdl_all_seeds_1 = len(df_of_results_FS_omc_mdl)
									df_of_results_FS_omc_mdl.loc[index_line_to_write_in_for_omc_mdl_all_seeds_1] = [ctype, tag_drugname, tag_profilename,
										len(dframe), trainingset_size_the_biggest,
										model_complexities_explored_report,
										str(len(classif_seeds)) + " seeds", str(runtime_case_for_all_seeds),
										med_ol2_col_omc, med_ol2_col_of_omc_il1_mcc,
										persistentin_ol2_col_of_topfeats_corresponding_to_omc_il1, nonpersistentin_ol2_col_of_topfeats_corresponding_to_omc_il1]
									# -----------lets also report the table keeping track of the features ranking : pval included and the fts (marked as cut if it is the case)

									#---lets add to it a new df but containing only the persistent or nonpersistent
									# these are the cols names to respect to be able to concatenate into the already existing dataframe : ["Seed"]["Pval_of_univ_stat"]["Features_as_in_rank"]
									# next step is to dispatch the topfts lists collected...
									# ...between a list holding lists of persistent topfts by seed (s seeds ie s lists of persitent topfts) # we sort to have ordered list (for a more easier check of present and absent fts in the output)
									persistent_feats_in_lists_of_topfeats_extended = sorted(set.intersection(*[set(list_of_topfeats_extended) for list_of_topfeats_extended in all_seeds_col_of_list_fts_correlated_enough_to_response]))
									# ...and a list holding one list of non persistent topfts by seed (s seeds ie s lists of non persitent topfts) # also if repeats happens to exits in the flat list of all feats setdiff1d eliminates them
									flat_list_from_all_seeds_col_of_list_fts_correlated_enough_to_response = [feat for list_of_topfeats_extended in all_seeds_col_of_list_fts_correlated_enough_to_response for feat in list_of_topfeats_extended]
									non_persistent_feats_in_lists_of_topfeats_extended = sorted(np.setdiff1d(flat_list_from_all_seeds_col_of_list_fts_correlated_enough_to_response, persistent_feats_in_lists_of_topfeats_extended).tolist())
									# we sort to have ordered list (for a more easier check of present and absent fts in the output)
									#---lets supply the frame created for across seeds info on fts in fashion ['title, list of what we want it the column]
									accross_seeds_frame_content_persistent_column = ["Persistent_in_extFS"] + persistent_feats_in_lists_of_topfeats_extended
									accross_seeds_frame_content_nonpersistent_column = ["NON_Persistent_in_extFS"] + non_persistent_feats_in_lists_of_topfeats_extended
									if len(accross_seeds_frame_content_persistent_column) > len(accross_seeds_frame_content_nonpersistent_column) :
										accross_seeds_frame_content_allsseeds_column = ["All seeds"] + np.repeat(str(len(classif_seeds)) + " seeds", len(accross_seeds_frame_content_persistent_column)).tolist()
									else : # if len(accross_seeds_frame_content_persistent_column) < len(accross_seeds_frame_content_nonpersistent_column) or if they are equals
										accross_seeds_frame_content_allsseeds_column = ["All seeds"] + np.repeat(str(len(classif_seeds)) + " seeds", len(accross_seeds_frame_content_nonpersistent_column)).tolist()
									#...building the last dataframe (across all seeds) and appending it to the col of the dfs for across the seeds extended lists of feats selected
									df_of_persistent_or_not_feats_accross_seeds = pd.DataFrame({'Seed': pd.Series(accross_seeds_frame_content_allsseeds_column), 'Features_ranked': pd.Series(accross_seeds_frame_content_persistent_column), 'Pval_of_univ_stat': pd.Series(accross_seeds_frame_content_nonpersistent_column)})
									all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed.append(df_of_persistent_or_not_feats_accross_seeds)
									# the df for all extended feats lists across the seeds
									df_of_fts_correlated_enough_to_response_across_all_seeds = pd.concat(all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed)
								# ...all metrics for all seeds are reported
								else:  ## exception of data sparsity
									# put here the last for adding only NA in place of the values
									# closing condittion on class sparsity in data defined so no analysis and empty results (NAs)
									print(bcolors.HEADER + " W : Classification part, class sparsity declared with : ", dataBin.iloc[:, 0].value_counts()[encoded_classes[0]], encoded_classes[0], "and", dataBin.iloc[:, 0].value_counts()[encoded_classes[1]], encoded_classes[1] + bcolors.ENDC)
									print(bcolors.HEADER + " W : To analyse, augment your unsuffisant classes population size or lower your limit of declaring class sparsity." + bcolors.ENDC)
									index_line_to_write_in_df_of_results_FS_omc_mdl = len(df_of_results_FS_omc_mdl)
									df_of_results_FS_omc_mdl.loc[index_line_to_write_in_df_of_results_FS_omc_mdl] = [tag_ctype, tag_drugname, tag_profilename, len(dframe)] + ["Class_sparsity!!!"] * 8

									# runtime_case_for_one_seed not needed here because nothing analysed
									# df_of_results.loc[index_line_to_write_in] = [str(runtime_case_for_one_seed), tag_alg, tag_alg_mark, tag_num_trial, ctype, drug, tag_drugname, featuretype, feature_val_type, len(dframe)] + ["NA"] * 23
									print(bcolors.HEADER + "W : Empty line added to result files" + bcolors.ENDC)
								#### lets get in a csv copy of the df_of_results. the output filename should identify the exact trial
								# lets make up a filename for the FS and then use it to create the .csv file
								output_filename_for_FS_omc_mdl = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_FS.csv"
								df_of_results_FS_omc_mdl.to_csv(output_filename_for_FS_omc_mdl, index=None, header=True)
								# lets make up a filename for the extended FS and then use it to create the .csv file
								output_filename_for_ExtFS_omc_mdl = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_ExtFS.csv"
								df_of_fts_correlated_enough_to_response_across_all_seeds.to_csv(output_filename_for_ExtFS_omc_mdl, index=None, header=True)

							else :
								print(bcolors.HEADER + " W : In Classification, no Feature Selection study request has been detected. To do so, please involve the proper arguments in command line." + bcolors.ENDC)
							# end of feature selection estimations<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
						else :
							print(bcolors.HEADER + " W : Classification part, analysis of this sub-case (", ctype, "-", drug, "-", featuretype, ") has not been carried out. Reason : number of samples inferior to literature based minimal samples number (",classif_msn,")" + bcolors.ENDC)
						# end of a profile in a case (subcase)
						# get the runtime for this profile
						runtime_one_profile_in_a_case = duration_from(before_one_profile_in_a_case_timer_start)
						print(bcolors.OKGREEN + " Classification part, analysis of the subcase (", ctype, "-", drug, "-", featuretype, ") is finished. Time taken : ",runtime_one_profile_in_a_case , bcolors.ENDC)
					# ---- all featypes availables explored
					# get the runtime for allprofiles in this case
					runtime_all_profiles_in_a_case = duration_from(before_any_profile_in_a_case_timer_start)
					print(bcolors.OKGREEN + " Classification part, all profiles available in the data of the case (",ctype,"-",drug,") have been explored in",runtime_all_profiles_in_a_case, bcolors.ENDC)
				else :
					print(bcolors.HEADER + " W : Classification part, analysis of this case (",ctype,"-",drug,") has not been carried out. Reason : number of samples inferior to literature based limit (35 samples)" + bcolors.ENDC)
					pass
			# ---- all drugs have been explored.
			# get the runtime for all drugs in this cancer type
			runtime_all_drugs_tested_on_a_ctype = duration_from(before_any_drugs_tested_on_a_ctype_timer_start)
			print(bcolors.OKGREEN + " Classification part, all drugs tested on the cancer type (",ctype,") have been explored in ",runtime_all_drugs_tested_on_a_ctype, bcolors.ENDC)
			# dispose of the frame with drugs restriction of the last loop iteration (del drugframe)
			# del drugframe
		# --- all ctypes have been explored.
		# get the runtime for all ctypes in this analysis
		runtime_all_ctypes = duration_from(before_any_ctype_timer_start)
		print(bcolors.OKGREEN + " Classification part, all cancer types have been explored in",runtime_all_ctypes, bcolors.ENDC)
		# dispose of the frame with ctype restriction of the last loop iteration (del cframe)
		# del cframe
	# end of all classifiers use
	runtime_all_classifiers = duration_from(before_all_classifiers_timer_start)
	print(bcolors.OKGREEN + " Classification part, all classifiers done : Time taken by all the classifiers runs is : ", runtime_all_classifiers, bcolors.ENDC)
# end of all classification operations #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
runtime_classif_part = duration_from(before_all_classif_ops_timer_start)
print(bcolors.OKGREEN + " Classification part of (" + tag_num_trial + ") done: Time taken : ", runtime_classif_part, bcolors.ENDC)

# uncomment to access the regression part ##!! just imitate the classification part and change the metrics used
# #>>>>>>>>>>>>>>>>>>>>>>>>>>> REGRESSION OPERATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print(bcolors.OKGREEN + "Engaging Regression task of the analysis named "+tag_num_trial+"..." + bcolors.ENDC)
# if ("Regr" in our_args.SL_tasks_to_perform) | ("Both" in our_args.SL_tasks_to_perform) :
#===================paste classification code here and modify================================================
# # end of all regression operations
# #wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

# all the analysis is done : get the runtime
runtime_analysis = duration_from(globalstart)
print(bcolors.OKGREEN + " Analysis (" + tag_num_trial + ") done : Time taken is : ", runtime_analysis, bcolors.ENDC)
# concatenate end with "," instead of "+" to avoid TypeError: unsupported operand type(s) for +: 'datetime.timedelta' and 'str'
# stop redirection of stdout if it was being done #!! update this condittion with regression log redirection when regression implemented
if classif_decision_make_log in ["yes", "y"]:
	sys.stdout = original_out
# this is the dev file