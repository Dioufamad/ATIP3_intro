# place interpreter path here as #!path_to_python installation
# put encoding here
# ========================script for XGBoost========================================
# -------------------------DEPENDENCIES--------------------------------
# import datetime
import locale
import numpy as np
import os
# import pandas as pd
import random
import scipy
import warnings
# from sklearn.preprocessing import LabelEncoder #to change the Response values from string to classes 0 and 1
# import xgboost as xgb
# from enginesV3.fs_engine import feat_selection1
# from enginesV3.Stratify import stratify
from enginesV3.data_engine1_mgmt import data_mgmt_1,data_mgmt_2,data_mgmt_3,data_mgmt_4,unifier_creator,data_loadout_right_corresponder,data_mgmt_5
from enginesV3.data_engine2_allocation import testset_indexes_spacer,testset_indexes_selector, trainingset_indexes_selector, set_creator_w_rows_index,add_entry_in_dict
from enginesV3.fs_engine import length_features_list,maximal_complexity_as_half_tr,list_of_complexities_ext,list_of_complexities_ltd,stratification1,feat_selection1
from enginesV3.Classif_algs_engine import classifier_introduction,classifier_model_training,classifier_model_prediction
from enginesV3.metrics_engine import r2,rmse,spearmanr_test,spearmanr_test_dec,r2_dec,rmse_dec
from enginesV3.watcher_engine import timer_started,duration_from
from random import shuffle
# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # for setting the characters format

msn = 35 # minimal_samples_number for classification following litterature

cv_standard = 5  #for K-fold cross-validation

num_cores = 38 #for the Number of cores used

basedir = os.getcwd() #for setting the working directory

warnings.filterwarnings("ignore") #for the behaviour of the warnings alerts

# add here a section to fixated shebang style

testset_size = 10 # size of the test set
min_complexity = 2 # minimal complexity to test
# uncomment one of these two
task_type = "Classif"
# task_type = "Regr"
Resp_col_name = "Resp_Class"
#uncomment on of these to use an ml alg :
classifier_version = "XGBoost_C_1" # the xgboost alg with default parameters
# minimum conserved correlation ie for the prediction to be conserved # usaually 0.25
min_cons_corr = -2000

# add here a list_of_binary_profiles
list_of_binary_profiles = ["SNV", "CNA"]
## the output filename can have appointed tags to identify easier the analysis it reports on :
# task_type = "Classif" #already initialised
tag_ctype = "brca"
tag_num_drug = "t17"
tag_num_profiles = "gex"
tag_num_copy = "trial1"
#-------------------------Paths to data sources---------------------------------------------------------------

### Paths needed to extract data (uncomment to activate a data_path1 for data matrix and a data_path2 for names of the drugs tested
# data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_1C_1T"
# data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_1C_3T"
# data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_1C_2T_var"
# data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_2C_1T"
# data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_1C_1T_2P_issue_dframe" # problematic shortest test
data_path1 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataTest_1C_1T_1P" # okay shortest test
data_path2 = "/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/inp"
# NB : this part of the links change from machine to machine : /home/khamasiga/ClassHD_work/actual_repo_home/homework_side/xgb_basic27_2/PDX__data
#-------------------------Data extraction step 1 / 2 ------------------------------------------------------------

### data_extraction strategy : data_management_1 (NIBR-PDXE data)
extracted_data1 = data_mgmt_1(data_path1, data_path2,Resp_col_name)
data_loadout_left_all = extracted_data1[0]
data_loadout_right_gex = extracted_data1[1]
data_loadout_right_cn = extracted_data1[2]
data_loadout_right_snv = extracted_data1[3]
data_loadout_right_cna = extracted_data1[4]
data_loadouts_origin_files_for_gex = extracted_data1[5]
data_loadouts_origin_files_for_cn = extracted_data1[6]
data_loadouts_origin_files_for_snv = extracted_data1[7]
data_loadouts_origin_files_for_cna = extracted_data1[8]
mydrugs = extracted_data1[9]
ctypes = extracted_data1[10]
drugs = extracted_data1[11]
mylabels = extracted_data1[12]

#-------------------------Results file step 1/2---------------------------------------------------------------

### lets create the output file for the results with only the columns to fill for now and at the end of analysis we fill them
## please see tags at start of script in variable to initialise
# lets create the output file and the titles of its columns # + task_type + tag_ctype +
with open(basedir + "/" + "outputs" + "/" + "out_" + task_type + "_" + tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy + ".txt", "w") as output_file :
	output_file.write("Seed" + "\t" + "Prof_type" + "\t" + "Ctype" + "\t" + "DrugID" + "\t" + "Drug_Name" + "\t"
			  + "Number_CellLines" + "\t" + "Train_set_size" + "\t" + "N/2" + "\t" "allFeat" + "\t" + "nOpt" + "\t"
			  + "ValidationSet_Correlation" + "\t" + "Rs_test_OMC" + "\t" + "Rs_test_all" + "\t"
			  + "Rs_test_OMC_Controlled" + "\t" + "RMSE_test_OMC" + "\t" + "RMSE_test_all" + "\t" + "R2_test_OMC"
			  + "\t" + "R2_test_all" + "\t" + "Selected_Features" + "\t" + "Predictions_OMC" + "\t"
			  + "Controlled_Preds_OMC" + "\t" + "Predictions_allFeat" + "\t" + "Predictions_observed" + "\n")

#------------------------Analysis--------------------------------------------------------------
globalstart = timer_started() # start a clock to get the time after all the analysis

print "Starting data analysis"
# if an analysis use a ML alg excploiting a random seed, that seed has to be initialised before the analysis
random.seed(1)
random_seeds = [1]
# multiples runs can be made to see how changing the seed affects the robustness of the analysis

#### Step 1 : Loop on the ctype, drug and featuretype to find the proper info, and unifying the left and right tables for that info by using a unifier commonly found in both tables

####loop1 : on cancertypes (# loop in the ctypes and define each time a drug_response dataframe (cframe) by restricting to only actual considered ctype)
for ctype in ctypes:
	cframe = data_loadout_left_all.loc[data_loadout_left_all["Cancer_type"] == ctype] #take only rows of drug_response that are that ctype
	####loop2 : on treatments ( # loop in the drugs (numbers here) and define each time a specific_cancer-drug_response dataframe (drugframe) by restricting to only actual considered ctype's and drug used)
	for drug in drugs:
		drugframe = cframe.loc[cframe["Treatment_id"] == drug]  # take only rows of drug_response that are that drug
		starttime = timer_started()  # start a clock to get the time after each feature in a profile
		if len(drugframe) >= msn:  # verify if at least n=35 samples are available as the smallest value encountered for a classification task in oncology
			# intialisation of a dict to keep track of the profiles found for a case (ctype-drug)
			profile_data_found_for_case = {}
			####loop3 : on profile_types (# loop in the features and each time do the appending of the proper feature values to the specific_cancertype-drug_specific dataframe. in each case, the appending is based on both dtaframes containing the unique cosmic ids)
			for featuretype in mylabels:
				featureframe = drugframe.loc[drugframe["Profile_type"] == featuretype]
				unifier = unifier_creator(ctype, drug, featuretype)
				corresponding_data_loadout_right = data_loadout_right_corresponder(featuretype, data_loadout_right_snv, data_loadout_right_cna, data_loadout_right_gex, data_loadout_right_cn)
				# lets create dframe ie the table of the full data of a subcase (ctype-drug-profile(s)) and that is to divide in training and testing
				dframe = data_mgmt_2(unifier,featureframe,corresponding_data_loadout_right)[0]
				profile_data_archived = data_mgmt_2(unifier,featureframe,corresponding_data_loadout_right)[1] # for keeping track
				add_entry_in_dict(profile_data_found_for_case,featuretype,profile_data_archived) # for keeping track
				print "Entire data source for computations is obtained for : ", ctype, drug, featuretype
				## formatting the dframe
				print "Entire data source formatting - for features - for response - dropping non informative instances/samples: "
				dframe = data_mgmt_5(dframe, 5, featuretype, list_of_binary_profiles)
				# lets change values of response in the appropriate type (floats for regression and categorial for classification)
				dframe = data_mgmt_3(dframe,task_type,Resp_col_name) # choose 2nd agr as task_type (defined at start of script as one of "Classif" or "Regr") ###testing
				# not a tuple returned but just one df so no indexing into a tuple
				# lets eliminate features with only zeros as values, sort the dataframe entries following repsonse col values and remake a new index
				dframe = data_mgmt_4(dframe, Resp_col_name)
				# subcase_frame = data_mgmt_4(subcase_frame,Resp_col_name)

				#-------------reflections
				# data_mgmt_4 can also be at the start of the task 1 and was initialy. moved before the loop on each seed to make it just one time
				# also lets extract the name of the response column because it will be used very often

				# after having obtained dframe and stylised it as required for classfication or regression,lets compute on it
				# dframe = subcase_frame  # dframe is a copy that will be used for division into training and test set
				try:
					# dframe = subcase_frame #dframe is a copy that will be used for division into training and test set
					if len(dframe) >= msn: # verify if at least minimal samples number (msn) responses are still available after data management
						for aseed in random_seeds:
							np.random.seed(aseed) # the value from the list of seeds is made a seed (used later as stocharsitic parameter of the classifier algorithm used to create the model) # each seed used is a run, and many runs can be joined for a mean value of prediciton

							##########################################
							#          Task 1 : Train & test split   #
							##########################################
							# streatgy to select the indexes of the x samples in the test set and y samples in the training set
							# testset_size = # use this one to set a proportion of length dframe as size of the testset
							tsi_spacer = testset_indexes_spacer(dframe,testset_size)
							# using the spacer to elect the test set indexes
							testset_indexes = testset_indexes_selector(dframe, testset_size, tsi_spacer)
							# deduce it from the whole indexes of the dataframe to elect the training set indexes
							trainingset_indexes = trainingset_indexes_selector(dframe, testset_indexes)
							# using created indexes to elect the train and test set
							trainset = set_creator_w_rows_index(dframe, trainingset_indexes)
							testset = set_creator_w_rows_index(dframe, testset_indexes)
							#---------------------------------------------

							##########################################
							#  Task 2 : Feature selection using OMC  #
							##########################################
							# The plan : loop 1 on the CV folds, loop 2 (on each complexity), train and test then keep the estimation of the prediction (coor between preds and obs) and the p-value.
							# Later those value will be used to select the the best OMC and its two values. that selected OMC will be trained using full training set and tested, along with random model to get classifier estimations
							#------Task 2-1 : Define complexities to explore
							# we capped complexities at the top n/2 with n = samples number
							max_complexity = maximal_complexity_as_half_tr(trainset)
							additional_complexity_value = length_features_list(trainset, 5)
							complexities_list = list_of_complexities_ext(min_complexity,max_complexity,additional_complexity_value)

							#------Task 2-2 : Define the indexes to use (in the full training set cross validation) as training or test set
							stratified_folds_as_train_test = stratification1(cv_standard, Resp_col_name, trainset)

							# ------Task 2-3 : Caracterisation of each complexity model prediction ability in a cross validation setting
							# initialisation of collectors of the complexities caracteristics in a prediction task
							MCs_rel_estimations = {} # a place to collect for each prediction, complexity-goodness of prediction (MCs_rel_estimations) # rel=relative
							MCs_rel_pvalues = {} # a place to collect for each prediction, complexity-cloness to random model (put these after the store and preds collectors) (MCs_rel_pvalues) #rel=relative

							# for prediction accuracy computation, we need a compariosn of preds to obs : here we create the container of those values for all the folds:
							store_preds = {} # a dict key-list_of_arrays_as value for the preds (because we can catch when looping on the complexities, many arrays, bacause one complexity end up being caracterised for each fold; the dict will be based on "one complexity value-all its predictions,one in each fold)
							store_obs = [] # a list of x arrays for the obs (because we can catch a list of x arrays of resp values at each turn of looping on the x folds)

							# that makes 4 milestones to carry out in the next loop
							for train, test in stratified_folds_as_train_test: # loop on folds # for a unit in a list form as a,b (perfectly corresponds to a=indexes of training set and b=indexes of valdation set
								# train, test = stratfolds[0] # for testing
								# create each time the CV training set using one of the actual training indexes and # remaking the order or indexes
								trainframe = set_creator_w_rows_index(trainset, train)
								# create each time the CV validation set using the actual test indexes and # remaking the order or indexes
								valframe = set_creator_w_rows_index(trainset, test)

								# milestone 1/4 : # store values of resp in test part of the fold in an array # for prediction accuracy computation, we need a compariosn of preds to obs : here we store all obs in a fold
								store_obs.append(valframe[Resp_col_name].values)
								# carry out feature selection : build a dict of {one feature : its test p-value} and return at index 0 the dict and index 1 the sorted list on the p-value
								FS = feat_selection1(featuretype, trainframe[list(trainframe)[5:]], trainframe[Resp_col_name])
								fs_ledger = FS[0] # to keep track of the p-value obtained for each feature
								sorted_list_of_feats = FS[1] # a list of the keys if the values (p-value of test) are sorted # make a dict with the right order to see the order clearly later

								for complexity in complexities_list: # loop on complexities (2-n/2, all fts)
									# complexity = complexities[0] # uncomment to test following
									ctrain_x = trainframe[sorted_list_of_feats[:complexity]] # select the top x (x = presently considered complexity) in sorted_list of fts with test p-values, and reduce trainframe in this fold to those columns (training fts in the fold obtained)
									cval_x = valframe[sorted_list_of_feats[:complexity]] # do the same for the validation set in the fold (training fts in the fold obtained)
									ctrain_y = trainframe[Resp_col_name] # get the response column from the trainframe to train with (training resp in the fold obtained)
									cval_y = valframe[Resp_col_name] # # get the response column from the validation set to compare against in the validation step (validation resp-truth in the fold obtained)

									# introduction of a classifier to make a model
									model = classifier_introduction(classifier_version,num_cores,aseed) # stock the method
									# model fitting
									model_fitted = classifier_model_training(model,ctrain_x, ctrain_y) # fit the method to the training data to get a classifier and predict with it later
									# model prediction
									model_prediction = classifier_model_prediction(model_fitted,cval_x) # a prediction using validation set for considered features

									# milestone 2/4 : for prediction accuracy computation, we need a comparison of preds to obs : here we store some pred, in a fold, in one complexity
									add_entry_in_dict(store_preds,complexity,model_prediction)

									# acc = (model_prediction == cval_y).sum().astype(float) / len(model_prediction)*100 # testing classification
									# print ("complexity is : ", complexity, "with acc of : ", acc)  # testing classification

									# model prediction estimations (goodness of prediction and closeness to random model)
									try:
										validation_OMC = spearmanr_test_dec(model_prediction, cval_y) # a metric of the prediction: a paired t-test for spearman correlation producing (correlation, p-value)
										# milestone 3/4 : goodness of prediction kept away
										add_entry_in_dict(MCs_rel_estimations,complexity,validation_OMC[0]) #stock the correlation in a dict (complexity-correlation) (to estimates the prediction made by this complexity)
										# milestone 4/4 : closeness to random model kept away
										add_entry_in_dict(MCs_rel_pvalues,complexity,validation_OMC[1]) #stock the p-value in a dict (complexity-p_value) (to judge the probability of this prediction made by this complexity to be further from the random model)
									except FloatingPointError: # action to do if the paired t-test of spearman correlation give a float the interpreter has trouble representing
										print ("a FPE has occured for complexity :", complexity)
										pass

								# all complexitites have been parsed for a fold
							# all folds have been parsed (so now the part about the complexities and 5xcv is over)

							# ------Task 2-4 : OMC selection
							# next part (the following for loop), is to allocate these : the best OMC value, the mean estimation (spearman) for that best OMC, and the mean p-value for that best OMC
							# the for loop on the finished MCs_rel_estimations.
							# understood finally : the OMC in MCs_rel_estimations are looped on and these 3 following values are updated each time if a better OMC is found. in the end, only one OMC stays and we compute the predictions on the test set with it :
							# updating algorithm : has to have each corr > 0.25 and 5 values to be studied (if better corr than previous, keep as actual; if equal corr as actual, keep only if better p_value)
							# the 3 collectors are initialised with the worst values
							OMC = 0 # not even possible here but should be the worst selection ie means no feat selected
							OMC_spearman = -1 # oppositely correlated is the worst correlation between preds and obs
							OMC_pvalue = 1 # this is the value of probability meaning selection surely being obtained at random ie that is the worst possible selection

							for acomplexity in MCs_rel_estimations:
								if np.min(MCs_rel_estimations[acomplexity]) > min_cons_corr: # verify if complexity is potentially predictive (> 0.25)
									if len(MCs_rel_estimations[acomplexity]) != 5: # we did a 5xcv so for computation, we discard collection others than that of 5 values (happen in case of a pass event due to an error, e.g. floating point error
										pass
									else:
										if np.median(MCs_rel_estimations[acomplexity]) > OMC_spearman: # restrict to value likely to be correlated prediction ie non anti-correlated
											OMC = acomplexity # keep complexity
											OMC_spearman = np.median(MCs_rel_estimations[acomplexity]) # keep medians values of estimations of the prediction
											OMC_pvalue = np.median(MCs_rel_pvalues[acomplexity])  # Disable when R2
										else:
											if np.median(MCs_rel_estimations[acomplexity]) == OMC_spearman: # in case of the complexity giving anti-correlation,
												if np.median(MCs_rel_pvalues[acomplexity]) < OMC_pvalue: # if on top of that, it shows that the complexity median correspond to the probability of random (1),
													OMC = acomplexity # keep complexity # seems like the same is done for correlation values  in >=-1, so why not join them in one condittion???
													OMC_spearman = np.median(MCs_rel_estimations[acomplexity]) # keep medians values of estimations of the prediction
													OMC_pvalue = np.median(MCs_rel_pvalues[acomplexity])
											else:
												pass
								else:
									pass
							# ---------------------------------------------

							##################################################
							# 3- Best OMC model and Random model - training and prediction #
							##################################################
							# the three precedent values (omc+2 medians were kept for the best OMC). Computations are made for that OMC here

							if OMC == 0: # no need to bother doing next treatment if pre-initialised value of OMC is the always there (possible if only we have errors or case of non acceted values)
								pass
							else:
								# We respect this order :
								# Training : OMCmodel-random model; Test : OMCmodel-random model;

								# =======================>PRELIMINARY COMPUTATIONS
								# the complexity correspondintg features are researched again (again a complexity is a # of feats not the feats themselves)
								FS_omc = feat_selection1(featuretype, trainset[list(trainset)[5:]], trainset[Resp_col_name])  # use FS metrics to rank features
								fs_ledger_omc = FS_omc[0]
								sorted_list_of_feats_omc = FS_omc[1]  # produce the list of rank

								# =======================>TRAINING : Train the models to compare on full dataset
								# -------->OMC model
								# trainingset building
								train_x = trainset[list(trainset)[5:]][sorted_list_of_feats_omc[:OMC]]  # df of all features in the trainset, restrited to the columns that are the top-x (x=OMC value)
								# model introduction
								OMCmodel = classifier_introduction(classifier_version,num_cores,aseed)
								# model training
								OMCmodel_fitted = classifier_model_training(OMCmodel,train_x,trainset[Resp_col_name].values) # fit the method to the training data to get a classifier and predict with it later
								# ---------->Random model : All features, no pre-ranking model training
								# trainingset building
								sorter_all = list(trainset)[5:]  # stock a list of columns of features
								shuffle(sorter_all)  # randomly rearrange the order of the columns
								train_x_noFS = trainset[list(trainset)[5:]][sorter_all]  # rebuild the trainset but with the randomized order of columns
								# model introduction
								allfeat_model = classifier_introduction(classifier_version,num_cores,aseed)
								# all fts model training
								allfeat_model_fitted = classifier_model_training(allfeat_model,train_x_noFS,trainset[Resp_col_name].values) # trainset resp values is unchanged is regards to columns randomising because only columns are moved but not rows ie nothing happened to resp column
								# =======================>TEST :
								# ---------Step 1/2 : Create the appropriate test set for each model on full dataset
								# # -------->OMC model
								test_x = testset[list(testset)[5:]][sorted_list_of_feats_omc[:OMC]]  # same as with trainset, lets extract in testset the columns that are in the OMC and in order (test_x)
								# ---------->Random model
								test_x_noFS = testset[list(trainset)[5:]][sorter_all]
								# NB : obs for test set : the testset_y for all fts is just the same as with the OMC model
								test_reg_y = testset[Resp_col_name]  # lets extract the response for the testset (test_y)
								# ---------Step 2/2 : Evaluation of the models (OMC, all fts models,+added exp) training and testing for using spearman
								# make functions to isolate for each computation a corr (0) and a p-value 1
								# Train, OMC
								spearman_train = spearmanr_test(OMCmodel.predict(train_x), trainset[Resp_col_name].values)
								# Train, all features-random order of fts
								spearman_train_allFeat = spearmanr_test(allfeat_model.predict(train_x_noFS), trainset[Resp_col_name].values)
								# Test, OMC
								spearman_test = spearmanr_test(OMCmodel.predict(test_x), test_reg_y.values)
								# Test, all features-random order of fts-
								spearman_test_allFeat = spearmanr_test(allfeat_model.predict(test_x_noFS), test_reg_y.values)

								# OMC model rectified predictions experience
								# it could be interesting to see the difference between the OMCmodel predictions and those same predictions but rectified with the relation of the model in the training when searching for the OMC
								# - on one hand, the model from the training  can be assimilated to the regression between preds and obs in the cross validation
								# on the other hand, the OMCmodel is available
								# instead of using predictions with the model from all the data, we seek out the correlation between the result of  one one hand and the obs in from the predscould predict with the OMCmodel but also re from the trainset from the cross-validation using a regression y=ax+b
								# could be interesting to see the difference with the usual model.predict--------1_bis

								# get the regr relation components between preds and obs for the OMC
								store_preds = store_preds[OMC]
								store_preds = np.array([item for sublist in store_preds for item in sublist])
								store_obs = np.array([item for sublist in store_obs for item in sublist])
								slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(store_preds, store_obs)  # arguments here are in the form (x,y) to form : y = slope*x + intercept
								# get rectified predictions using the regression relation from the training step during cross validation research of the OMC
								controlled_preds = [(slope * x + intercept) for x in OMCmodel.predict(test_x)]
								#estimate the corr of the prediction using those rectified predictions
								spearman_test_controlled = spearmanr_test(controlled_preds, test_reg_y.values)[0]
								# =======================> PERFORMANCES METRICS  : RMSE and R2
								# RMSE
								RMSE_OMC = rmse(OMCmodel.predict(test_x), test_reg_y.values)  # RMSE for OMC model----1
								RMSE_all = rmse(allfeat_model.predict(test_x_noFS), test_reg_y.values)  # RMSE for all fts model----2
								# R2
								R2_OMC = r2(test_reg_y.values, OMCmodel.predict(test_x))
								R2_all = r2(test_reg_y.values, allfeat_model.predict(test_x_noFS))

								##################################################
								# 4- Reporting the results                       # #----Results file step 2/2---------------
								##################################################
								if OMC < 100:
									with open(basedir + "/" + "outputs" + "/" + "out_" + task_type + "_" + tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy + ".txt", "a") as output_file:
										output_file.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(max_complexity) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(MCs_rel_estimations[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + " ".join(sorted_list_of_feats_omc[:OMC]) + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")
									# outfile.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(max_complexity) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(MCs_rel_estimations[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + " ".join(sorted_list_of_feats_omc[:OMC]) + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")
								else:
									with open(basedir + "/" + "outputs" + "/" + "out_" + task_type + "_" + tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy + ".txt", "a") as output_file:
										output_file.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(max_complexity) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(MCs_rel_estimations[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + ">100 features" + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")
									# outfile.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(max_complexity) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(MCs_rel_estimations[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + ">100 features" + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")
							# ---------end of tasks to do for a seed
						# ----------- all seed values have been explored
						# lets print he type it took for all seeds work to complete
						print featuretype, ctype, drug, "Done for all seeds", duration_from(starttime)
					else :
						print "Analysis of this sub-case (",ctype,"-",drug,"-",featuretype,") has not been carried out. Reason : number of samples inferior to literature based limit (35 samples)"
				except NameError:
					print ctype, drug, featuretype,"Passed due some calculations that was unconclusive ", duration_from(starttime)  # error message stating a problem with the existence of the given ctype-given drug frame
				# ------ the dframe have been explored using all its samples for model estimations. lets dispose of it
				# del subcase_frame # just an idea... didnt wwork
				# del dframe
			# ---- all featypes availables explored
		else :
			print "Analysis of this case (",ctype,"-",drug,") has not been carried out. Reason : number of samples inferior to literature based limit (35 samples)"
			pass
		# ---- all drugs have been explored. dispose of the frame with drugs restriction
		# del drugframe
	# --- all ctypes have been explored. dispose of the frame with ctype restriction
	# del cframe
# all loops on data are over. lets close the the result file
# outfile.close()
# get the time it took for all loops on the data until this point
runtime_case = duration_from(globalstart)
print "Time taken for the case [",tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy, "] : ", runtime_case


# okay fine testing all now
# explanation on reporting results : restart here
# each line in outfile contains these values. the code for each value is given. two value are separated by a separator.
# SEPARATOR between two values:  + "\t" +
# Seed : str(aseed)
# Prof_type : featuretype
# Ctype : ctype
# DrugID : str(drug)
# Drug_Name : "".join([str(x) for x in mydrugs[drug]])
# Number_CellLines : str(len(dframe))
# Train_set_size : str(len(trainset))
# N/2 : str(maxfeatures)
# allFeat : str(len(list(trainset[list(trainset)[5:]])))
# nOpt : str(OMC)
####### to modify
# ValidationSet_Correlation : str(np.median(MCs_rel_estimations[OMC]))
# Rs_test_OMC : str(spearman_test[0])
# Rs_test_all : str(spearman_test_allFeat[0])
# Rs_test_OMC_Controlled : str(spearman_test_controlled)
# RMSE_test_OMC : str(RMSE_OMC)
# RMSE_test_all : str(RMSE_all)
# R2_test_OMC : str(R2_OMC)
# R2_test_all : str(R2_all)
#####
# Selected_Features : " ".join(sorted_list_of_feats_omc[:OMC])
# Predictions_OMC : str(OMCmodel.predict(test_x)).replace("\n", "")
# Controlled_Preds_OMC : str(controlled_preds)
# Predictions_allFeat : str(allfeat_model.predict(test_x_noFS)).replace("\n", "")
# Predictions_observed : str(test_reg_y.values).replace("\n", "")
# at the end of each line to go to next line : "\n"
#