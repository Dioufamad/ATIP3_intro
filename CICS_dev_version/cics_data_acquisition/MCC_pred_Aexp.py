#----------get the mean limits
#---------------------get the HKGs
#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import pandas as pd
import numpy as np # linear algebra and exploit arrays faster and easier computations
from math import sqrt # need to compute mcc
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap # to wrap plot titles

#>>>>>>>>>>>>>>>>>>>Choosing the environnement values
# ----for the task to do
task_2do = "select_cols_based_on_medians"
# ----for the response strategy
resp_used = "RCH3HSall"
resp_used_in_full = "All the samples with -defined or not RCH and 3 hormonals status, are kept"
# resp_used = "RCH3HSdefined"
# resp_used_in_full = "Only defined -RCH and the 3 hormonals status- samples are kept"
# resp_used = "RCHdefined"
# resp_used_in_full = "Only defined RCH samples are kept"
# resp_used = "TNBCdefined"
# resp_used_in_full = "Only defined TNBC samples are kept"
# resp_used = "RCHandTNBCdefined"
# resp_used_in_full = "Only defined RCH and TNBC samples are kept"
print("For population restriction, the response strategy chosen is",resp_used_in_full,"(",resp_used,")...")
# ----for the cohorts to implicate
# cohort_used = "REMAGUS02"
# cohort_used = "REMAGUS04"
cohort_used = "MDAnderson"
# ----for the Ath to use to predict
# ath_origin = "Aexp_R02"
# ath_origin = "Aexp_R04"
ath_origin = "Aexp_MDA"
if ath_origin == "Aexp_R02":
	ath=0.339615210980143
elif ath_origin == "Aexp_R04":
	ath=0.487777016221156
else: # ath_origin == "Aexp_MDA"
	ath=0.54344429410706
#>>>>>>>>Setting up the environnement values
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
if command_center == "Gustave_Roussy":
	rest_of_abs_path_b4_content_root = "/home/amad/PycharmProjects/ATIP3_in_GR/"
else : # command_center = "Home"
	rest_of_abs_path_b4_content_root = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/"
print("command center used recognized...")
# ----making the dataset to use when the cohort(s) to manipulate is(are) known
#----paths to files of populations in cohorts
R02_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R02/"
R04_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R04/"
MDA_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/MDA/"
if resp_used == "RCH3HSall":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx226Sx54675Fx4RasRCH3HSall_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx142Sx22277Fx4RasRCH3HSall_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx4RasRCH3HSall_GEX.csv"
elif resp_used == "RCH3HSdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx218Sx54675Fx4RasRCH3HSdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx4RasRCH3HSdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx129Sx22283Fx4RasRCH3HSdefined_GEX.csv"
elif resp_used == "RCHdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx221Sx54675Fx1RasRCHdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx1RasRCHdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx1RasRCHdefined_GEX.csv"
elif resp_used == "TNBCdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx226Sx54675Fx1RasTNBCdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx142Sx22277Fx1RasTNBCdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx1RasTNBCdefined_GEX.csv"
else:  # resp_used == "RCHandTNBCdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx221Sx54675Fx2RasRCHandTNBCdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx2RasRCHandTNBCdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx2RasRCHandTNBCdefined_GEX.csv"
#---making the base dataset to use
sep_in_file = ","
if cohort_used == "REMAGUS02":
	df_file_R02 = pd.read_csv(file_path_R02, sep_in_file)
	df_file = df_file_R02
elif cohort_used == "REMAGUS04":
	df_file_R04 = pd.read_csv(file_path_R04, sep_in_file)
	df_file = df_file_R04
else : # cohort_used == "MDAnderson":
	df_file_MDA = pd.read_csv(file_path_MDA, sep_in_file)
	df_file = df_file_MDA

# #---add to the dataset a column having the cohort name for each sample
# if cohort_used == "REMAGUS02":
#     df_file.insert(len(list(df_file.columns)), 'Cohort', 'Remagus02')
# elif cohort_used == "REMAGUS04":
#     df_file.insert(len(list(df_file.columns)), 'Cohort', 'Remagus04')
# else:  # cohort_used == "MDAnderson":
#     df_file.insert(len(list(df_file.columns)), 'Cohort', 'MDAnderson')

#>>>>>>>>>>>>>>>>>>>restrict the dataset to the columns of interest
# df_file = df_file.set_index('Model')
# ----for the columns to keep in all 3 cohorts df
# we will keep all the cols except the responses (only fts and model remains)
if resp_used == "RCH3HSall":
	list_of_cols_to_remove = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2"]
elif resp_used == "RCH3HSdefined":
	list_of_cols_to_remove = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2"]
elif resp_used == "RCHdefined":
	list_of_cols_to_remove = ["BestResCat_as_RCH"]
elif resp_used == "TNBCdefined":
	list_of_cols_to_remove = ["BestResCat_as_TNBC"]
else : # resp_used == "RCHandTNBCdefined":
	list_of_cols_to_remove = ["BestResCat_as_RCH", "BestResCat_as_TNBC"]

# ---- making the df with full info to use
df_file_fts_model_only = df_file.drop(list_of_cols_to_remove, axis=1)

# the base df
df_base = df_file_fts_model_only
#---make df for the file from HM
sep_in_file = ","
if cohort_used == "REMAGUS02":
	file_path_clusters_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_clusters.txt"
elif cohort_used == "REMAGUS04":
	file_path_clusters_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_clusters.txt"
else : # cohort_used == "MDAnderson":
	file_path_clusters_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_clusters.txt"
df_clusters_cohort = pd.read_csv(file_path_clusters_cohort, sep_in_file)
df_clusters_cohort = df_clusters_cohort.rename_axis('Model_bis').reset_index()
df_clusters_cohort.rename(columns={"x": "Clusters"}, inplace=True)
# join the df to get big on on which to compute after restricted
df_joined = pd.merge(df_base, df_clusters_cohort, how="inner", left_on="Model", right_on="Model_bis")
df_joined.drop(labels=["Model_bis"], axis=1, inplace=True)

#---making the column of real clusters
# choose the cluster to call high (by examining the heatmap scale of color and the dendrogram
# the color of the biggest group at cut for 2 groups give it)
if cohort_used == "REMAGUS02":
	num_cluster2dropIEhigh = 1
	num_cluster2predictIElow = 2
elif cohort_used == "REMAGUS04":
	num_cluster2dropIEhigh = 2
	num_cluster2predictIElow = 1
else : # cohort_used == "MDAnderson":
	num_cluster2dropIEhigh = 1
	num_cluster2predictIElow = 2

# replace the numbers of clusters with strings
df_joined.rename(columns={"Clusters": "Real_Clusters"}, inplace=True)
df_joined["Real_Clusters"].replace([num_cluster2dropIEhigh,num_cluster2predictIElow], ["high", "low"], inplace=True)

#==========> area to correct later
# - copy the dataset, find the columns for the probes of ATIP3 and compute the mean expr for all row (samples) then rejoin with df we had before
# make a copy
df_Mg_start_low_end_only = df_joined
# they all have MTUS1 in the colname (they are 4 for R02 and 3 for R04 and MDA)
# a list of the column of the probes
full_list_of_cols = list(df_Mg_start_low_end_only.columns)
list_of_probes2keep_colnames = []
for colname in full_list_of_cols:
	if "MTUS1" in colname:
		list_of_probes2keep_colnames.append(colname)
list_of_cols2keep = list_of_probes2keep_colnames + ["Model"]
df_Mg_start_low_end_cols_restricted = df_Mg_start_low_end_only[list_of_cols2keep]
# put the col of samples content aside (into the index) and then bring it back later
df_Mg_start_low_end_cols_restricted = df_Mg_start_low_end_cols_restricted.set_index('Model')
# compute the mean and keep it in a col
df_Mg_start_low_end_cols_restricted["mean_probeset_MTUS1"] = df_Mg_start_low_end_cols_restricted.mean(axis=1) # default is skipna=True
# bring back the samples col
df_Mg_start_low_end_cols_restricted = df_Mg_start_low_end_cols_restricted.rename_axis('Model').reset_index()
# sort as descending on the col of the mean
df_Mg_start_low_end_cols_restricted= df_Mg_start_low_end_cols_restricted.sort_values("mean_probeset_MTUS1", axis=0, ascending=False, kind='mergesort')
# drop index in order to use new ordered index to adress the first line as index 0
df_Mg_start_low_end_cols_restricted_sortedOnMeanRow = df_Mg_start_low_end_cols_restricted.reset_index(drop=True)
# restrict on columns to keep for the joining
columns_need_to_have_when_joining = ["Model","mean_probeset_MTUS1"]
df_Mg_start_low_end_cols_restricted_sortedOnMeanRow = df_Mg_start_low_end_cols_restricted_sortedOnMeanRow[columns_need_to_have_when_joining]
df_Mg_start_low_end_cols_restricted_sortedOnMeanRow.rename(columns={"Model": "Model_bis2"}, inplace=True)
# now do the joining
df_joined = pd.merge(df_joined, df_Mg_start_low_end_cols_restricted_sortedOnMeanRow, how="inner", left_on="Model", right_on="Model_bis2")
df_joined.drop(labels=["Model_bis2"], axis=1, inplace=True)





# -----mean for top 20s HKGs (for each sample)
# make a copy
df_Std_start = df_joined
# - restrict the df
# drop unnecessary cols
df_Std_start_restricted = df_Std_start.drop(["Real_Clusters","mean_probeset_MTUS1"], axis=1)
# put the col of samples content aside (into the index) and then bring it back later
df_Std_start_restricted = df_Std_start_restricted.set_index('Model')
# ---get the cols to keep
#---make df for the file from HKGs selection
sep_in_file = ","
if cohort_used == "REMAGUS02":
	file_path_HKGs_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_Top20HKGsCVnoDupRanked_report.csv"
elif cohort_used == "REMAGUS04":
	file_path_HKGs_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_Top20HKGsCVnoDupRanked_report.csv"
else : # cohort_used == "MDAnderson":
	file_path_HKGs_cohort = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_Top20HKGsCVnoDupRanked_report.csv"
df_HKGs_cohort = pd.read_csv(file_path_HKGs_cohort, sep_in_file)
# a list of the colnames found in the df of HKGs and to use to restrict
list_colnames_asHKGs = df_HKGs_cohort['HKGs'].tolist()
# restrict on cols that are the HKGs
df_Std_start_restricted = df_Std_start_restricted[list_colnames_asHKGs]
# compute the mean and keep it in a col
df_Std_start_restricted["mean_top20_HKGs_sample_limit"] = df_Std_start_restricted.mean(axis=1) # default is skipna=True
# bring back the samples col
df_Std_start_restricted = df_Std_start_restricted.rename_axis('Model').reset_index()
# sort as descending on the col of the mean
df_Std_start_restricted= df_Std_start_restricted.sort_values("mean_top20_HKGs_sample_limit", axis=0, ascending=False, kind='mergesort')
# drop index in order to use new ordered index to adress the first line as index 0
df_Std_start_restricted_sortedOnMeanRow = df_Std_start_restricted.reset_index(drop=True)
# restrict on columns to keep for the joining
columns_need_to_have_when_joining2 = ["Model","mean_top20_HKGs_sample_limit"]
df_Std_start_restricted_sortedOnMeanRow = df_Std_start_restricted_sortedOnMeanRow[columns_need_to_have_when_joining2]
df_Std_start_restricted_sortedOnMeanRow.rename(columns={"Model": "Model_bis3"}, inplace=True)
# now do the joining
df_joined = pd.merge(df_joined, df_Std_start_restricted_sortedOnMeanRow, how="inner", left_on="Model", right_on="Model_bis3")
df_joined.drop(labels=["Model_bis3"], axis=1, inplace=True)

#--------restrict the cols for clarity before making the cols of predictions
list_of_to_keep_going_towards_predictions = ["Model","mean_probeset_MTUS1","mean_top20_HKGs_sample_limit","Real_Clusters"]
df_joined = df_joined[list_of_to_keep_going_towards_predictions]
# reming the ath used for the predictions
print("cohort being done predictions on is : ",cohort_used)
print("Ath being used is : ",ath_origin)
print("Value Ath being used is :",ath)
# produce a column mg
df_joined['mg_from_mean_top20_HKGs'] = ath * df_joined["mean_top20_HKGs_sample_limit"]
# produce the col of predictions
df_joined['Pred_Clusters'] = np.where(df_joined['mean_probeset_MTUS1'] < df_joined['mg_from_mean_top20_HKGs'], 'low', 'high')

#---------get the mcc for the total operation
TP = 0
TN = 0
FP = 0
FN = 0
list_of_the_samples = df_joined['Model'].tolist() # get a list of the samples to loop through them when they are an index
print("Predicting the cluster of this number of samples:",len(list_of_the_samples))
df_joined = df_joined.set_index('Model') # make an index out of the col of the samples
for a_sample in list_of_the_samples:
	real_cluster_sample = df_joined.at[a_sample,"Real_Clusters"]
	predicted_cluster_sample = df_joined.at[a_sample, "Pred_Clusters"]
	if predicted_cluster_sample=="high": # cases where the negative class  is predicted
		if predicted_cluster_sample==real_cluster_sample:
			TN+=1
		else : # not predicted_cluster_sample==real_cluster_sample
			FN+=1
	elif predicted_cluster_sample=="low": # cases where the positive class  is predicted
		if predicted_cluster_sample==real_cluster_sample:
			TP+=1
		else : # not predicted_cluster_sample==real_cluster_sample
			FP+=1

# computation of a mcc
mcc_numerator = (TP*TN)-(FP*FN)
candidate_for_mcc_denominateur = sqrt((TP+FP)*(FP+TN)*(TN+FN)*(FN+TP))
if candidate_for_mcc_denominateur == 0: # condittion of existence of mcc mathematically
	mcc_denominateur = 1
else:
	mcc_denominateur = candidate_for_mcc_denominateur
mcc_df_joined = mcc_numerator / mcc_denominateur
print(mcc_df_joined)








