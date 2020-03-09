###--------------------- This is the location of some functions contributing to Feature Selection operations-----------------------

###---------------------IMPORTS FOR CLASSIFICATION
import numpy as np
import pandas as pd
import scipy
import locale
from operator import itemgetter # to sort lists of tuples using directly one position in the index elements
###---------------------IMPORTS COMPLEMENTARY FOR REGRESSION
# from rpy2 import robjects
# from rpy2.rinterface import RRuntimeError for python 2.7 solve later
# from rpy2.rinterface_lib.embedded import RRuntimeError # for python 3.7
# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#====================================================================

#====================================classification used functions
#-------get the number of feature in a df
def length_features_list(trainset, index_of_first_feat_in_cols_list): # take a frame as arg and index_of_first_feat_in_cols_list (5) as arg
	len_fts_list = len(list(trainset)[index_of_first_feat_in_cols_list:])
	return len_fts_list
##!! here the index of the first feature is to be given but in the future, make a function to detect in (hint : the index of the first column name to not contain sample id ctype etc.)

#---process of limiting training data to only fts who shows variation (at least one variation in a sample)
def eliminate_non_variable_fts(frame):
	frame_with_variable_fts_only = frame.drop(frame.columns[frame.apply(lambda col: col.nunique(dropna=True) == 1)], axis=1)
	return frame_with_variable_fts_only

#-------------------------------functions to define the value of the maximal complexity
# calcul de n/2 pour chaque train set
def maximal_complexity_as_half_tr(trainset):
	maxfeatures = int(np.ceil(float(len(trainset)) / 2))
	return maxfeatures
# add others functions for others values (eg : the 3/4, 70% etc.)
#-------------------------------functions that define the list of complexities to test seeking the best of them
def list_of_complexities_ext(minfeatures, maxfeatures,additional_complexity_value): # minfeatures = 2 for OMC
	complexities = list(range(minfeatures, maxfeatures)) + [additional_complexity_value]  # range de 2 a n/2 + number total of fts # sum of 2 lists is a bigger list with the two succesiively in it
	complexities = list(sorted(set(complexities)))  # make it a set, sort it and make it a list
	return complexities

def list_of_complexities_ltd(minfeatures, maxfeatures):
	complexities = list(range(minfeatures, maxfeatures))
	complexities = list(sorted(set(complexities)))
	return complexities

#---process of limiting training data to only fts who shows variation (at least one variation in a sample) ## isolate
# def ranker_by_pval_v1(frame_fts,frame_resp,feature_type,resp_col): ### deprecated
# 	dict_pvalues = {}
# 	if feature_type in ["SNV","CNA"]: # cases of discrete values, do a fisher exact test on contigency table of fts var with response and extract pvalues
# 		for feat in list(frame_fts):
# 			the_feat_col = pd.Categorical(frame_fts[feat],categories=[0,1])
# 			the_resp_col = pd.Categorical(frame_resp[resp_col],categories=["Res","Sen"]) ##!! externalise to the for loop to do it only one time ##!! use categories list instead of hard code
# 			contingency_table_filled = pd.crosstab(the_feat_col,the_resp_col,dropna=False) # dropna=False is used to tell to the system to still count classes that were not predicted
# 			oddsratio, p_value_f = scipy.stats.fisher_exact(contingency_table_filled, alternative="two-sided")
# 			dict_pvalues[feat] = p_value_f
# 	else : # case of reals values, find the two extreme situations of response segregation ability by the feature or do a t test
# 		res_indexes = frame_resp.index[frame_resp["Resp_Class"] == "Res"].tolist()
# 		sen_indexes = frame_resp.index[frame_resp["Resp_Class"] == "Sen"].tolist()
# 		for feat in list(frame_fts):
# 			the_resp_samples = frame_fts.loc[res_indexes,[feat]]
# 			the_sen_samples = frame_fts.loc[sen_indexes, [feat]]
# 			if (the_resp_samples[feat].nunique(dropna=True) == 1) & (the_sen_samples[feat].nunique(dropna=True) == 1) : # the 2 groups values are uniquely described
# 				if the_resp_samples[feat].unique()[0] != the_sen_samples[feat].unique()[0]: # by a different unique value
# 					p_val_top = -1 # the perfect differenciation then
# 					dict_pvalues[feat] = p_val_top
# 				else:
# 					p_val_worst = 1  # the worst differenciation then
# 					dict_pvalues[feat] = p_val_worst
# 			else: # not just one unique value exist in a group, the t-test is done then
# 				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples, the_sen_samples, equal_var=False) #old line 0
# 				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples.dropna()[feat], the_sen_samples.dropna()[feat], equal_var=False) #old line 1.5
# 				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples.iloc[:,0].tolist(),the_sen_samples.iloc[:,0].tolist(),equal_var = False,nan_policy='propagate') # old line 1
# 				t_stat, p_value_t = scipy.stats.ttest_ind(np.array(the_resp_samples.iloc[:, 0].tolist()), np.array(the_sen_samples.iloc[:, 0].tolist()), equal_var=False, nan_policy='omit')
# 				##!! transforming a column to an array is done by putting it as a list and then as a numpy array
# 				# ttest_ind suppose variance equals by default so underestimates p for unequal variances even if the t-statistic is the same (eg : case where size of samples are equals but on different scale :;
# 				# When n1 != n2, the equal variance t-statistic is no longer equal to the unequal variance t-statistic:
# 				# so we use  equal_var = False
# 				## !!! to try : make a condittion if variance and size are equals, use equal_var = True else use equal_var = False
# 				# nan_policy='propagate' to take into account the nan and 'omit' to not take them into account
# 				dict_pvalues[feat] = p_value_t
# 	# sorted_feats_by_pval = sorted(dict_pvalues, key=dict_pvalues.get, reverse=False) # deprecated as not working properly
# 	sorted_feats_by_pval = [key for key, value in sorted(dict_pvalues.items(), key=itemgetter(1), reverse=False)] # default is reverse = False so can omit it
# 	# sorted_pval = [value for key, value in sorted(dict_pvalues.items(), key=itemgetter(1), reverse=False)]  to report later on results # default is reverse = False so can omit it
# 	return dict_pvalues,sorted_feats_by_pval

# new versio of the ranker by pval function
def ranker_by_pval_v2(frame_fts,frame_resp,feature_val_type,resp_col,encoded_classes):
	dict_pvalues = {} # a col for all the pvalues computed
	res_indexes = frame_resp.index[frame_resp[resp_col] == encoded_classes[0]].tolist()  # isolate the uniques values for each group of the two response in the population
	sen_indexes = frame_resp.index[frame_resp[resp_col] == encoded_classes[1]].tolist()
	# considering one feature, the 2 groups values can be uniquely described by one unique value each : prfect p_value attributed if the 2 values are different; worst p_value attributed if the 2 values are identifical...
	p_val_top = -1  # ...the p_value value for the perfect differenciation then
	p_val_worst = 1  # ...the p_value value for the worst differenciation then
	if feature_val_type == "cat" : # cases of discrete values, do a fisher exact test on contigency table of fts var with response and extract pvalues
		for feat in list(frame_fts):
			the_resp_samples = frame_fts.loc[res_indexes, [feat]]
			the_sen_samples = frame_fts.loc[sen_indexes, [feat]]
			if (the_resp_samples[feat].nunique(dropna=True) == 1) & (the_sen_samples[feat].nunique(dropna=True) == 1) : # the 2 groups values are uniquely described
				if the_resp_samples[feat].unique()[0] != the_sen_samples[feat].unique()[0]: # by a different unique value
					dict_pvalues[feat] = p_val_top
				else:
					dict_pvalues[feat] = p_val_worst
			else : # not just one unique value exist in each group, the fisher exacttest is done then
				the_feat_col = pd.Categorical(frame_fts[feat],categories=[0,1])
				the_resp_col = pd.Categorical(frame_resp[resp_col],categories=encoded_classes) ## older ["Res","Sen"] is now encoded_classes (an array of shape (2,) ##!! externalise to the for loop to do it only one time ##!! use categories list instead of hard code
				contingency_table_filled = pd.crosstab(the_feat_col,the_resp_col,dropna=False) # dropna=False is used to tell to the system to still count classes that were not predicted
				oddsratio, p_value_f = scipy.stats.fisher_exact(contingency_table_filled, alternative="two-sided")
				dict_pvalues[feat] = p_value_f
	else : # case of reals values, find the two extreme situations of response segregation ability by the feature or do a t test
		for feat in list(frame_fts):
			the_resp_samples = frame_fts.loc[res_indexes,[feat]]
			the_sen_samples = frame_fts.loc[sen_indexes, [feat]]
			if (the_resp_samples[feat].nunique(dropna=True) == 1) & (the_sen_samples[feat].nunique(dropna=True) == 1) : # True is the 2 groups values are uniquely described by a different unique value each...
				if the_resp_samples[feat].unique()[0] != the_sen_samples[feat].unique()[0]: # ...True if those two values are the same
					dict_pvalues[feat] = p_val_top
				else:					# ....case where those two values are not the same
					dict_pvalues[feat] = p_val_worst
			else: # not just one unique value exist in each group, the t-test is done then
				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples, the_sen_samples, equal_var=False) #old line 0
				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples.dropna()[feat], the_sen_samples.dropna()[feat], equal_var=False) #old line 1.5
				# t_stat, p_value_t = scipy.stats.ttest_ind(the_resp_samples.iloc[:,0].tolist(),the_sen_samples.iloc[:,0].tolist(),equal_var = False,nan_policy='propagate') # old line 1
				t_stat, p_value_t = scipy.stats.ttest_ind(np.array(the_resp_samples.iloc[:, 0].tolist()), np.array(the_sen_samples.iloc[:, 0].tolist()), equal_var=False, nan_policy='omit')
				##!! transforming a column to an array is done by putting it as a list and then as a numpy array
				# ttest_ind suppose variance equals by default so underestimates p for unequal variances even if the t-statistic is the same (eg : case where size of samples are equals but on different scale :;
				# When n1 != n2, the equal variance t-statistic is no longer equal to the unequal variance t-statistic:
				# so we use  equal_var = False
				## !!! to try : make a condittion if variance and size are equals, use equal_var = True else use equal_var = False
				# nan_policy='propagate' to take into account the nan and 'omit' to not take them into account
				dict_pvalues[feat] = p_value_t
	# sorted_feats_by_pval = sorted(dict_pvalues, key=dict_pvalues.get, reverse=False) # deprecated as not working properly
	sorted_feats_by_pval = [key for key, value in sorted(dict_pvalues.items(), key=itemgetter(1), reverse=False)] # default is reverse = False so can omit it
	sorted_pvals = [value for key, value in sorted(dict_pvalues.items(), key=itemgetter(1), reverse=False)]  # to report later on results # default is reverse = False so can omit it
	return dict_pvalues,sorted_feats_by_pval,sorted_pvals

#----------go through the coupled value of mc and mcc and find the best one ie the OMC
def OMC_founder_in_dict_MCs_MCCs(dict_MCs_MCCs):
	OMC = list(dict_MCs_MCCs.keys())[0]  # allfts model complexity in fts by default taken as the OMC
	omc_mdl_mcc = list(dict_MCs_MCCs.values())[0]
	for mc_as_key in list(dict_MCs_MCCs.keys())[1:]:  #  loop on the rest of the mdl_complexities
		if dict_MCs_MCCs[mc_as_key] > omc_mdl_mcc:
			OMC = mc_as_key
			omc_mdl_mcc = dict_MCs_MCCs[mc_as_key]
		elif dict_MCs_MCCs[mc_as_key] == omc_mdl_mcc:
			if mc_as_key < OMC:
				OMC = mc_as_key
			# no need to attribute the correspondant value of mcc because an equal value is already there
			else:
				pass
		else:  # case of dict_of_mcc_values_by_mdl_to_update[mc_as_key] < omc_mdl_mcc
			pass
	# the OMC and its mcc variables is updated
	return OMC,omc_mdl_mcc

#-------function to get persistent features in list of lists
def make_list_of_persistents_fts_from_list_of_lists_of_fts(a_list_of_lists_of_fts):
	list_of_persistent_fts = []
	for list in a_list_of_lists_of_fts:
		for feat in list:
			if feat not in list_of_persistent_fts:
				list_of_persistent_fts.append(feat)
#==================== classification used functions  (uncomment to use)

#==================== regression used functions  (uncomment to use)
#
# #-------------------------------functions to select features (following a univariate test, a dict feat-p-value in test is created)
# # lets define a function taking as arg :
# # - presently computed featuretype (feature_type)
# # - the part of the training folds frame with only the features columns (feats_frame)
# # - the part of the training folds frame with only the resp column (frame_resp_col)
# def feat_selection1(feature_type, feats_frame, frame_resp_col):
# 	results_fs = {}  # Feature selection output as a dict {feat-pvalue}
#
# 	if feature_type == "SNV" or feature_type == "CNA":  # t-test (for discrte values feats use Wilcoxonrank-sum test)
# 		for somecolumn in list(feats_frame): # loop on the list of the feats (list of the columns names)
# 			mutindexes = feats_frame[somecolumn].loc[feats_frame[somecolumn] == True].index # give a list of indexes for the variant # strategy : for the whole frame get the lines having a given value; restrict that to the  frame but only cionsidering one column ; get indexes of the resulting lines (not needed to resptrict herte but it shows that in the spirit of the work only the resp column matter atthis point)
# 			wtindexes = feats_frame[somecolumn].loc[feats_frame[somecolumn] == False].index # give a list of indexes for the wt
# 			pval = 1
#
# 			try:  # NumPy wilcoxon not used due to it being nominal approximation  # to review and understand
# 				wilcoxon2 = robjects.r['wilcox.test']
# 				v12 = robjects.FloatVector(list(frame_resp_col[mutindexes].values)) # the resp values for the mutated indexes
# 				v22 = robjects.FloatVector(list(frame_resp_col[wtindexes].values)) # the resp values for the wt indexes
# 				wilcox_result2 = wilcoxon2(v12, v22, conf_int=True)
# 				pval = wilcox_result2.rx2("p.value")[0]
# 			except RRuntimeError:
# 				pass  # If always mutated or not mutated, it is not single-handedly correlated with response
# 			results_fs[somecolumn] = pval  # Add p-values to dictionary
# 	else:  # spearman test  (for real values feats use corr coef of the spearman test) (does a test of correlation between 2 variable a and b : a is the presently considered column (feat.) and b is the resp column
# 		for somecolumn in list(feats_frame):
# 			try:
# 				testresult = scipy.stats.spearmanr(list(feats_frame[somecolumn].values), frame_resp_col, axis=0, nan_policy='propagate')
# 				# in the var testresult, we have a tuple of [0] = correlation and [1] = pvalue.
# 				# lets extract the pvalue and add it as an entry of the dict resultfs
# 				results_fs[somecolumn] = testresult[1]
# 			except FloatingPointError:
# 				pass  # Same issue as for Wilcoxon, if always mutated or not, it alone is not correlated
#
# 	# sorted_fs = []
# 	sorted_fs = sorted(results_fs, key=results_fs.get, reverse=False) # une liste des cles du dict (les features) ordonne de maniere croissante de la valeur de p-value
# 	return results_fs, sorted_fs
# 	# return results_fs, sorted(results_fs, key=results_fs.get, reverse=False)
#====================for regression (uncomment to use)





