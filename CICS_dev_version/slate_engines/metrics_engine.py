###--------------------- This is the location of some functions caring for metrics calculations V3.1-----------------------

###---------------------IMPORTS FOR CLASSIFICATION
import numpy as np
import pandas as pd # for pushing dataframes functions in order to report or keep track
from math import sqrt
import locale
# ---for the metrics
# from sklearn.metrics import classification_report # for if classification_report() has to be used
from pandas_ml import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
###---------------------IMPORTS COMPLEMENTARY FOR REGRESSION
#--- used for computation in regression metrics
# import scipy
# from decimal import Decimal
#--- to test it or fixate the precision
# from decimal import getcontext
# getcontext().prec = 5 # to set the precision by default at 28 with round_at_even
# print(getcontext()) # show the precision
# Decimal(1)/Decimal(3)  # an example
#====================================================================

# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#  ---------------------------------------------------------------------------------------------------
# -------------------------------------- Function Definitions ---------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CLASSIFICATION FUNCTIONS (from Linh Nguyen work on RF-OMC with NIBR-PDXE data)
##### Helper functions

##### Classification metrics
# compute mcc and add to a collector
def calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, frame_of_all_mdls_called_predictions, frame_of_all_mdls_implicated_samples_observations, resp_col, RespClasses):
	for one_mdl_called_predictions_as_a_column in list(frame_of_all_mdls_called_predictions):
		the_feat_col = pd.Categorical(frame_of_all_mdls_called_predictions[one_mdl_called_predictions_as_a_column], categories=[RespClasses[0], RespClasses[1]])
		the_resp_col = pd.Categorical(frame_of_all_mdls_implicated_samples_observations[resp_col], categories=[RespClasses[0], RespClasses[1]])
		contingency_table_filled = pd.crosstab(the_feat_col, the_resp_col,dropna=False)
		TP = contingency_table_filled.loc[RespClasses[1]][RespClasses[1]]
		TN = contingency_table_filled.loc[RespClasses[0]][RespClasses[0]]
		FP = contingency_table_filled.loc[RespClasses[1]][RespClasses[0]]
		FN = contingency_table_filled.loc[RespClasses[0]][RespClasses[1]]
		# calculation of a mcc
		mcc_numerator = (TP*TN)-(FP*FN)
		candidate_for_mcc_denominateur = sqrt((TP+FP)*(FP+TN)*(TN+FN)*(FN+TP))
		if candidate_for_mcc_denominateur == 0: # condittion of existence of mcc mathematically
			mcc_denominateur = 1
		else:
			mcc_denominateur = candidate_for_mcc_denominateur
		mdl_all_called_preds_mcc = mcc_numerator / mcc_denominateur
		# dict_of_mcc_values_by_mdl_to_update has been created before all the workings here
		dict_of_mcc_values_by_mdl_to_update[one_mdl_called_predictions_as_a_column] = mdl_all_called_preds_mcc # update the dict
	return dict_of_mcc_values_by_mdl_to_update #think about keeping the contingency matrices

###---compute mcc and just deliver the value #unused because old version is not stocking in col. ##!! could still be used if the end of function is to repaired
# def calculate_mcc_wo_storing(frame_of_all_mdls_called_predictions,frame_of_all_mdls_implicated_samples_observations,resp_col,RespClasses):
# 	dict_of_mcc_values_by_mdl_to_update = {}
# 	for one_mdl_called_predictions_as_a_column in list(frame_of_all_mdls_called_predictions):
# 		the_feat_col = pd.Categorical(frame_of_all_mdls_called_predictions[one_mdl_called_predictions_as_a_column], categories=[RespClasses[0], RespClasses[1]])
# 		the_resp_col = pd.Categorical(frame_of_all_mdls_implicated_samples_observations[resp_col], categories=[RespClasses[0], RespClasses[1]])
# 		contingency_table_filled = pd.crosstab(the_feat_col, the_resp_col,dropna=False)
# 		TP = contingency_table_filled.loc[RespClasses[1]][RespClasses[1]]
# 		TN = contingency_table_filled.loc[RespClasses[0]][RespClasses[0]]
# 		FP = contingency_table_filled.loc[RespClasses[1]][RespClasses[0]]
# 		FN = contingency_table_filled.loc[RespClasses[0]][RespClasses[1]]
# 		# calculation of a mcc
# 		mcc_numerator = (TP*TN)-(FP*FN)
# 		candidate_for_mcc_denominateur = sqrt((TP+FP)*(FP+TN)*(TN+FN)*(FN+TP))
# 		if candidate_for_mcc_denominateur == 0: # condittion of existence of mcc mathematically
# 			mcc_denominateur = 1
# 		else:
# 			mcc_denominateur = candidate_for_mcc_denominateur
# 		mdl_all_called_preds_mcc = mcc_numerator / mcc_denominateur
# 		# dict_of_mcc_values_by_mdl_to_update has been created before all the workings here
# 	dict_of_mcc_values_by_mdl_to_update[one_mdl_called_predictions_as_a_column] = mdl_all_called_preds_mcc # update the dict
# 	return dict_of_mcc_values_by_mdl_to_update #think about keeping the contingency matrices

#---the 8 metrics of the prediction of the positive class in a binary classif
def pd_ml_classif_report_on_cm_binary(frame_of_y_true, frame_of_y_pred,binary_classes_le):
	# ConfusionMatrix() uses arrays of encoded value reflecting the existence of two classes (even if one class does not exist the values must gives an intuition of another class that exist
	# Step 1 : encode the classes to memorize (done already after data_management) # only using now the encoder
	# le.classes_ gives an array of the classes
	# Step 2 : get arrays from frames of thruths ans predictions with values encoded (using encoding done after data management)
	y_true_as_array_of_encoded_values = np.array(binary_classes_le.transform(frame_of_y_true.iloc[:,0].tolist()))
	y_pred_as_array_of_encoded_values = np.array(binary_classes_le.transform(frame_of_y_pred.iloc[:,0].tolist()))
	# Step 3: build a confusion matrix
	confusion_matrix = ConfusionMatrix(y_true_as_array_of_encoded_values, y_pred_as_array_of_encoded_values)  # new line working w python3.7
	# sorted_list_of_the_classes is not needed here as the classes found are used and they are ordered and last is pos class
	# Also it can exist only one class in predictions so manage that with following
	# Step 4 : getting the metrics out of the classification report from the confusion matrix
	try:
		#~~~~~> for the class1 (version 1 deprecated working)
		# a confusion matrix of only two class gives a slighly different dict
		# diff1 : only one column as treating only the positive class
		# diff 2 : the keys in the dict "confusion_matrix.stats()" are different : these are to be considered.
		# [u'population', u'P', u'N', u'PositiveTest', u'NegativeTest', u'TP', u'TN', u'FP', u'FN', u'TPR', u'TNR', u'PPV', u'NPV', u'FPR', u'FDR', u'FNR', u'ACC', u'F1_score', u'MCC', u'informedness', u'markedness', u'prevalence', u'LRP', u'LRN', u'DOR', u'FOR']
		# for the class1 (present scheme)
		class1_acc = confusion_matrix.stats()["ACC"] # return [0]
		class1_MCC = confusion_matrix.stats()["MCC"] # return [1]
		class1_precision = confusion_matrix.stats()["TNR"] # aka "TNR=SPC: (Specificity)" # return [2]
		class1_recall = confusion_matrix.stats()["TPR"] # aka "TPR: (Sensitivity, hit rate, recall)" # return [3]
		class1_fpr = confusion_matrix.stats()["FPR"] # aka "FPR: False-out" # return [4]
		class1_TP = confusion_matrix.stats()["TP"] # return [5]
		class1_FN = confusion_matrix.stats()["FN"] # return [6]
		class1_TN = confusion_matrix.stats()["TN"] # return [7]
		class1_FP = confusion_matrix.stats()["FP"] # return [8]
		print("pandas_ml has confusion_matrix.stats() as a dict with [metric name as key, value of metric as value] (for positive class) : 9 metrics computed.")
		## not used
		# class1_f1score = confusion_matrix.stats()["F1_score"] # return [9]
		# class1_num_tests_w_outcome_as_class_pos = confusion_matrix.stats()["PositiveTest"]  # aka "Test outcome positive" # return [10]
		# class1_support = confusion_matrix.stats()["P"] # aka "P: Condition positive" # return [11]
		# <~~~~~~~~~~~~~(deprecated version 1 but working)
	except KeyError:
		# ~~~~~~> Version 2: deprecated (confusion_matrix.stats() is a dict of 3 entries(key-values))
		# 3rd entry has key "class" and is a dataframe obtained with (confusion_matrix.stats()["class"]) with all metrics as rows and each col is a class
		# reporting datframe format :
		# confusion_matrix.stats()["class"].columns
		# Index(['Res', 'Sen'], dtype='object', name='Classes')
		# confusion_matrix.stats()["class"].index
		# Index(['Population', 'P: Condition positive', 'N: Condition negative',
		#        'Test outcome positive', 'Test outcome negative', 'TP: True Positive',
		#        'TN: True Negative', 'FP: False Positive', 'FN: False Negative',
		#        'TPR: (Sensitivity, hit rate, recall)', 'TNR=SPC: (Specificity)',
		#        'PPV: Pos Pred Value (Precision)', 'NPV: Neg Pred Value',
		#        'FPR: False-out', 'FDR: False Discovery Rate', 'FNR: Miss Rate',
		#        'ACC: Accuracy', 'F1 score', 'MCC: Matthews correlation coefficient',
		#        'Informedness', 'Markedness', 'Prevalence',
		#        'LR+: Positive likelihood ratio', 'LR-: Negative likelihood ratio',
		#        'DOR: Diagnostic odds ratio', 'FOR: False omission rate'],
		#       dtype='object')
		# Accessions rules :
		# a row is accessed with confusion_matrix.stats()["class"].loc["Full metric name"]
		# and the value for that row (ie a metric) when considering a class x is confusion_matrix.stats()["class"].loc["Full metric name"]["class x"]
		class1_acc = confusion_matrix.stats()["class"].loc["ACC: Accuracy"][1]  # return [0]
		class1_MCC = confusion_matrix.stats()["class"].loc["MCC: Matthews correlation coefficient"][1]  # return [1]
		class1_precision = confusion_matrix.stats()["class"].loc["TNR=SPC: (Specificity)"][1]  # aka "TNR=SPC: (Specificity)" # return [2]
		class1_recall = confusion_matrix.stats()["class"].loc["TPR: (Sensitivity, hit rate, recall)"][1]  # aka "TPR: (Sensitivity, hit rate, recall)" # return [3]
		class1_fpr = confusion_matrix.stats()["class"].loc["FPR: False-out"][1]  # aka "FPR: False-out" # return [4]
		class1_TP = confusion_matrix.stats()["class"].loc["TP: True Positive"][1]  # return [5]
		class1_FN = confusion_matrix.stats()["class"].loc["FN: False Negative"][1]  # return [6]
		class1_TN = confusion_matrix.stats()["class"].loc["TN: True Negative"][1]  # return [7]
		class1_FP = confusion_matrix.stats()["class"].loc["FP: False Positive"][1]  # return [8]
		print("pandas_ml has confusion_matrix.stats() as a dict of 3 keys. Last key is class, a dataframe with [metric name as row, each class is defined as a column] : for the positive class, 9 metrics computed.")
		# <~~~~~~~~~~version 2 deprecated
	return class1_acc,class1_MCC,class1_precision,class1_recall,class1_fpr,class1_TP,class1_FN,class1_TN,class1_FP
	# class1_f1score,class1_num_tests_w_outcome_as_class_pos,class1_support, # handy outputs

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>REGRESSSION FUNCTIONS
##### Helper functions
# def restriction_of_MCCs_wide(wide_frame,fold,list_of_test_indexes):
# 	folds_col_name = wide_frame.columns[0]
# 	restricted_to_fold = wide_frame.loc[wide_frame[folds_col_name] == fold]
# 	samples_in_test_col_name = wide_frame.columns[1]
# 	restricted_to_samples_in_test = restricted_to_fold.loc[restricted_to_fold[samples_in_test_col_name].isin(list_of_test_indexes)] # list_of_test indexes = list_of_il1_test_data_indexes
# 	# restricted_to_mdls_tried = restricted_to_samples_in_test.iloc[:,index_start_space_of_cols_as_mdls:endplus1_space_of_cols_as_mdls]
# 	return restricted_to_samples_in_test

# def df_multiplier_in_rows(frame_to_multiply,times_to_multiply_into):
# 	frame_from_the_multiplication = pd.concat([frame_to_multiply] * times_to_multiply_into)
# 	return frame_from_the_multiplication

############ Regression metrics(from Stephan Nalauerts work on xgboost with GDSC data)
# #R2
# def r2(a, b):
# 	try :
# 		r2_calc = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)
# 	except FloatingPointError:
# 		r2_calc = "FTP" # FTP = FloatingPointError
# 	return r2_calc
#
# #R2 a version taking floating point error into account and using decimal
# def r2_dec(a, b):
# 	r2_calc_dec = Decimal(1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2))
# 	return r2_calc_dec
#
# # RMSE
# def rmse(a, b):
# 	try:
# 		rmse_calc = np.sqrt(np.sum((a - b) ** 2) / len(a))
# 	except FloatingPointError:
# 		rmse_calc = "FTP"  # FTP = FloatingPointError
# 	return rmse_calc
#
# # RMSE a version taking floating point error into account and using decimal
# def rmse_dec(a, b):
# 	rmse_calc_dec = Decimal(np.sqrt(np.sum((a - b) ** 2) / len(a)))
# 	return rmse_calc_dec
#
# # spearman_test
# def spearmanr_test(predicted_col, observed_col):
# 	try:
# 		spearman_test = scipy.stats.spearmanr(predicted_col, observed_col)
# 	except FloatingPointError:
# 		spearman_test = "FTP" # FTP = FloatingPointError
# 	return spearman_test
#
# # spearman_test a version taking floating point error into account and using decimal
# def spearmanr_test_dec(predicted_col, observed_col):
# 	# spearman_test = scipy.stats.spearmanr(predicted_col, observed_col,nan_policy='omit')
# 	spearman_test_corr = Decimal(scipy.stats.spearmanr(predicted_col, observed_col,nan_policy='omit')[0])
# 	spearman_test_pval = Decimal(scipy.stats.spearmanr(predicted_col, observed_col,nan_policy='omit')[1])
# 	return spearman_test_corr,spearman_test_pval
