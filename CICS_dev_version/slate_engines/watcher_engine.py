###--------------------- This is the location of some functions caring for whatever-----------------------

### IMPORTS
from datetime import datetime # for time functions (2)
#---
import locale
#---for drawing roc curves
import numpy as np
from scipy import interp  # used to interpolate data points between data instances # for roc curve
from sklearn.metrics import roc_curve, auc # to compute fpr,tpr based by looping on a gallery of thresholds # compute auc of roc curve # for roc curve
from textwrap import wrap # to wrap plot titles
#---for reporting results on files
import pandas as pd
# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#---

#------------------ timers functions
def timer_started():   # start a clock to get the time before a step
	start = datetime.now()
	return start

def duration_from(time_at_start):  # start a clock to get the time after a step and remove it from an initial time to get the duration of the step
	time_at_end = datetime.now()
	elapsed_time = time_at_end - time_at_start
	return elapsed_time


#-------------formatting functions
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#-----------drawing ROC curves functions
def roc_curve_updater_after_one_iteration_of_the_mdl(array_of_the_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data,array_of_the_mdl_predictions_2probs_on_all_data,RespClasses,mean_fpr_of_mdl,tprs_col_of_mdl,aucs_col_of_mdl,id_iter,fig_subplot_as_axi):
	# fpr and tpr using a set of thresholds
	fpr_mdl_one_iter, tpr_mdl_one_iter, thresholds_mdl_one_iter = roc_curve(array_of_the_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data, array_of_the_mdl_predictions_2probs_on_all_data, pos_label=RespClasses[1])
	# first arg is col truth classes, 2nd arg is col of class pos in classic array of probabilities, 3rd arg indicate the classes to take as pos among the two seen if the classes are not {0,1} or {-1,1}
	# a galerie of thresholds is made and each value of it is used to make calls of predictions and calculate with truth classes col, the fpr and tpr
	# Nb : one value in thresholds will seems out of the classic range 0-1 of probabilites in the positives (eg: 1,888). it is to ancher the extreme value of (0,0) and by calling alls samples as neg
	# to make a situation of TP = 0 ie tpr = 0 ie all curves of each iteration have a starting point
	# that pose the question of why not and when to fixated also the extreme point of (1,1) : it not needed for each iteration curve, we let them end wherever. though, that extreme point will be
	# defined at the time of building the mean curve ie the real roc curve
	tprs_col_of_mdl.append(interp(mean_fpr_of_mdl, fpr_mdl_one_iter, tpr_mdl_one_iter))  # stock the array of values of tpr that will be in y axis
	tprs_col_of_mdl[-1][0] = 0.0  # force the first value of the first array in the content of tpr collector to be 0.0 (see previous explanation)
	roc_auc_of_mdl_one_iter = auc(fpr_mdl_one_iter, tpr_mdl_one_iter)  # compute the auc value for this iteration
	aucs_col_of_mdl.append(roc_auc_of_mdl_one_iter)  # ...and add it to the auc collector
	fig_subplot_as_axi.plot(fpr_mdl_one_iter, tpr_mdl_one_iter, lw=1, alpha=0.3, label='ROC seed %d (AUC = %0.2f)' % (id_iter, roc_auc_of_mdl_one_iter))  # plot this iteration roc curve and what will be in the legend marked for it (roc interation id and auc value)
	return roc_auc_of_mdl_one_iter
def roc_curve_finisher_after_all_iterations_of_the_mdl(fig,fig_subplot_as_axi,mdls_comp_fig_subplot_as_axi,tprs_col_of_mdl,mean_fpr_of_mdl,aucs_col_of_mdl,basedir,task_type,tag_alg,the_model_compared,tag_ctype,tag_drugname,tag_profilename,tag_num_trial):
	##--for the ROC curve of omc mdl
	# add to the plot the line for the random prediction
	fig_subplot_as_axi.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
	# compute mean of tpr collector of tpr array : gives an array of tpr use for mean roc curve
	mean_tpr_from_tprs_col_of_mdl = np.mean(tprs_col_of_mdl, axis=0)
	# fixate the etrame values of tpr for the mean curve to 1
	mean_tpr_from_tprs_col_of_mdl[-1] = 1.0
	# use mean fpr and mean tpr to get mean auc (same as using auc function for one fpr,one tpr and get one auc value)
	mean_auc_from_cols_of_mdl = auc(mean_fpr_of_mdl, mean_tpr_from_tprs_col_of_mdl)
	# compute standard deviation on the flattened array given
	std_auc_from_aucs_col_of_mdl = np.std(aucs_col_of_mdl)
	# plot the mean roc curve and what will be in the legend marked for it (mean roc and auc value)
	fig_subplot_as_axi.plot(mean_fpr_of_mdl, mean_tpr_from_tprs_col_of_mdl, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_from_cols_of_mdl, std_auc_from_aucs_col_of_mdl), lw=2, alpha=.8)
	#----*added : this mean roc curve to the average plot
	if the_model_compared.startswith("OMC"):
		color_of_line = 'c' # 'c' for cyan for OMC
	else :
		color_of_line = 'm'  # 'm' for magenta for Allfts
	# plot the mean roc curve and what will be in the legend marked for it (mean roc and auc value)
	mdls_comp_fig_subplot_as_axi.plot(mean_fpr_of_mdl, mean_tpr_from_tprs_col_of_mdl, color=color_of_line, label=r'Mean ROC %s model (AUC = %0.2f $\pm$ %0.2f)' % (the_model_compared,mean_auc_from_cols_of_mdl, std_auc_from_aucs_col_of_mdl), lw=2, alpha=.8)
	#----*
	# # -----*substracted : this to color between the highest and lowest of a mean TPR+-std dev at a fixated FPR value
	# # compute standard deviation on the arrays given, give a list of values, one for each column
	# std_tpr_from_tprs_col_of_mdl = np.std(tprs_col_of_mdl, axis=0)
	#
	# # to fill up with grayed-out color the area between the highest and lowest tpr values, we get the highest tpr value without going past 1
	# tprs_upper_from_tprs_col_of_mdl = np.minimum(mean_tpr_from_tprs_col_of_mdl + std_tpr_from_tprs_col_of_mdl, 1)
	# # ...same but wo going under 0
	# tprs_lower_from_tprs_col_of_mdl = np.maximum(mean_tpr_from_tprs_col_of_mdl - std_tpr_from_tprs_col_of_mdl, 0)
	# # make the grayed-out indicate area
	# fig_subplot_as_axi.fill_between(mean_fpr_of_mdl, tprs_lower_from_tprs_col_of_mdl, tprs_upper_from_tprs_col_of_mdl, color='grey', alpha=.2, label=r'$\pm$ std. dev. of mean TPR at fixated FPR ') # old version had 1 std. dev.
	# # -----*
	# limits of the x axis and y axis
	fig_subplot_as_axi.set_xlim([-0.05, 1.05])
	fig_subplot_as_axi.set_ylim([-0.05, 1.05])
	# labels of the x axis and y axis
	fig_subplot_as_axi.set_xlabel('False Positive Rate')
	fig_subplot_as_axi.set_ylabel('True Positive Rate')
	# title of the plot ##!! to modify using tags
	fig_subplot_as_axi.set_title("\n".join(wrap('ROC curve of %(Task)s using %(Alg)s-%(Model)s model, on case %(Ctype)s-%(Drug)s, %(Profile)s profile,  %(Trial)s' %
								 {"Task": task_type, "Alg": tag_alg, "Model": the_model_compared, "Ctype": tag_ctype, "Drug": tag_drugname, "Profile": tag_profilename, "Trial": tag_num_trial})))
	# position of legend
	fig_subplot_as_axi.legend(loc="lower right")
	## display the plot ##!! replace it with a plot saving ##!! check it our way of creating files creates also folders
	fig.savefig(basedir + '/outputs/Output_' + task_type + "_" + tag_alg + "-" + the_model_compared + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + '_ROCcurve.png', bbox_inches='tight')
	return mean_auc_from_cols_of_mdl, std_auc_from_aucs_col_of_mdl


def average_roc_curve_init(mdls_comp_fig_subplot_as_axi):
	mdls_comp_fig_subplot_as_axi.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)


def average_roc_curve_finisher(mdls_comp_fig_subplot_as_axi, mdls_comp_fig, models_compared, task_type, tag_alg, tag_ctype, tag_drugname, tag_profilename, tag_num_trial, basedir):
	# limits of the x axis and y axis
	mdls_comp_fig_subplot_as_axi.set_xlim([-0.05, 1.05])
	mdls_comp_fig_subplot_as_axi.set_ylim([-0.05, 1.05])
	# labels of the x axis and y axis
	mdls_comp_fig_subplot_as_axi.set_xlabel('False Positive Rate')
	mdls_comp_fig_subplot_as_axi.set_ylabel('True Positive Rate')
	# title of the plot ##!! to modify using tags
	mdls_comp_fig_subplot_as_axi.set_title("\n".join(wrap('ROC curve comparing %(Model1)s and %(Model2)s models for %(Task)s using %(Alg)s, on case %(Ctype)s-%(Drug)s, %(Profile)s profile,  %(Trial)s' %
														  {"Model1": models_compared[0], "Model2": models_compared[1], "Task": task_type, "Alg": tag_alg, "Ctype": tag_ctype, "Drug": tag_drugname, "Profile": tag_profilename, "Trial": tag_num_trial})))
	# position of legend
	mdls_comp_fig_subplot_as_axi.legend(loc="lower right")
	## display the plot ##!! replace it with a plot saving ##!! check it our way of creating files creates also folders
	mdls_comp_fig.savefig(basedir + '/outputs/Output_' + task_type + "_" + tag_alg + "-" + models_compared[0] + "vs" + models_compared[1] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + '_ROCcurve.png', bbox_inches='tight')

#-------------tables filling functions
def df_of_results_for_metrics_one_mdl_creator():
	df_of_results_for_metrics_one_mdl = pd.DataFrame(columns=["Ctype", "DrugName", "Profile",
		"#Samples(=N)", "TrainingSetSize",
		"Model_complexities_explored (range or #) (fts)",
		"Seed(value or #)", "Duration",
		"Med_OMC",
		"Acc",
		"MCC",
		"AUC",
		"Prec",
		"Rec/TPR",
		"FPR",
		"TP",
		"FN",
		"TN",
		"FP"])
		# a df for results of 19 cols for each model metrics values
	return df_of_results_for_metrics_one_mdl

def df_of_results_for_metrics_all_mdls_creator():
	df_of_results_for_metrics_all_mdls = pd.DataFrame(columns=["Ctype", "DrugName", "Profile",
		"#Samples(=N)", "TrainingSetSize",
		"Model",
		"Model_complexities_explored (range or #) (fts)",
		"Seed(value or #)", "Duration",
		"Med_OMC",
		"Acc",
		"MCC",
		"AUC",
		"Prec",
		"Rec/TPR",
		"FPR",
		"TP",
		"FN",
		"TN",
		"FP"])
	# a df for results of 20 cols including all models metrics values
	return df_of_results_for_metrics_all_mdls

def df_of_results_for_FS_one_mdl_creator():
	df_of_results_for_FS_one_mdl = pd.DataFrame(columns=["Ctype", "DrugName", "Profile",
		"#Samples(=N)", "TrainingSetSize",
		"Model_complexities_explored (range or #) (fts)",
		"Seed(value or #)", "Duration",
		"Med_OMC",
		"MCC",
		"Selected fts list (for all seeds : fts always selected)","for all seeds : fts NOT always selected"])
		# a df for results of 12 cols for each model metrics values
	return df_of_results_for_FS_one_mdl
	### name of columns to keep for later :
	# "list_of_persitent_fts_in_OMC_FS","list_of_NON_persitent_fts_in_OMC_FS"


#------------------ writing in files results functions
# import sys
# import os
# basedir = os.getcwd()
# out1 = basedir + "/" + "testing1_writing.txt"
# out2 = basedir + "/" + "testing2_writing.txt"
# def writer(where,what):
# 	if where == sys.stdout:
# 		print(what)
# 	else :
# 		# where = open(basedir + "/" + "/complexity_testing_" + tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy + ".txt", "a")
# 		with open(where, "a") as outputfile1:
# 			outputfile1.write(what + "\n")




# outfile = open(basedir + "/" + "/complexity_testing_" + tag_ctype + "_" + tag_num_drug + "_" + tag_num_profiles + "_" + tag_num_copy + ".txt", "a")
# outfile.write("Seed" + "\t" + "Prof_type" + "\t" + "Ctype" + "\t" + "DrugID" + "\t" + "Drug_Name" + "\t"
# 			  + "Number_CellLines" + "\t" + "Train_set_size" + "\t" + "N/2" + "\t" "allFeat" + "\t" + "nOpt" + "\t"
# 			  + "ValidationSet_Correlation" + "\t" + "Rs_test_OMC" + "\t" + "Rs_test_all" + "\t"
# 			  + "Rs_test_OMC_Controlled" + "\t" + "RMSE_test_OMC" + "\t" + "RMSE_test_all" + "\t" + "R2_test_OMC"
# 			  + "\t" + "R2_test_all" + "\t" + "Selected_Features" + "\t" + "Predictions_OMC" + "\t"
# 			  + "Controlled_Preds_OMC" + "\t" + "Predictions_allFeat" + "\t" + "Predictions_observed" + "\n")

#-------------------------------
#-------------------------------
#-------------------------------
#-------------------------------
# globalstart = datetime.datetime.now()
# -----------------------------------------------------------------
# def data_mgmt_dummy(dframe, sl_type):
#     if sl_type == "Regr":
#         # change values of response into floats
#         dframe["Resp_class"] = dframe["Resp_class"].astype(float)
#         print ("Response values formated for a regression analysis")
#     elif sl_type == "Classif" :
#         # change values of response into classes (0 and 1)
#         le = LabelEncoder()
#         y_encoded = le.fit_transform(dframe["Resp_class"])
#         dframe["Resp_class"] = y_encoded
#         print ("Response values formated for a classification analysis")
#     return dframe
# -----------------------------------------------------------------