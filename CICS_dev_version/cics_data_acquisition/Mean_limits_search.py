#----------get the mean limits
#---------------------get the HKGs
#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import pandas as pd
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

#-------mean for ATIP3 low end
df_Mg_start = df_joined
# restrict to samples of low end cluster
# choose the cluster to drop by examining the heatmap scale of color and the dendrogram
# (the color of the biggest group at cut for 2 groups give it)
if cohort_used == "REMAGUS02":
	num_cluster2dropIEhigh = 1
elif cohort_used == "REMAGUS04":
	num_cluster2dropIEhigh = 2
else : # cohort_used == "MDAnderson":
	num_cluster2dropIEhigh = 1

indexes2drop = df_Mg_start[df_Mg_start['Clusters'] == num_cluster2dropIEhigh].index
df_Mg_start_low_end_only = df_Mg_start.drop(indexes2drop , inplace=False)
df_Mg_start_low_end_only = df_Mg_start_low_end_only.reset_index(drop=True)
#---restrict to the columns of interest
# - Finding the columns for the probes of ATIP3
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
# - The value of the mean expression limit to go into the low expression range of MTUS1 : Mg
Mg = df_Mg_start_low_end_cols_restricted_sortedOnMeanRow.at[0,"mean_probeset_MTUS1"]
# - keep the sample name of the sample at the limit for later compuation for Std
# a list of samples on the decreasing order of the mean expression
list_samples_decreasing_order_mean_expr = df_Mg_start_low_end_cols_restricted_sortedOnMeanRow['Model'].tolist()
sample_at_the_limit = list_samples_decreasing_order_mean_expr[0] # for later computations on the same sample

# -----mean for top 20s HKGs
df_Std_start = df_joined
# - restrict the df
# restrict to rows to one sample (the sample limit whose mean expression acroos probes is Mg)
df_Std_start_restricted = df_Std_start.loc[df_Std_start['Model'] == sample_at_the_limit]
# redo index
df_Std_start_restricted = df_Std_start_restricted.reset_index(drop=True)
# drop unnecessary cols
df_Std_start_restricted = df_Std_start_restricted.drop(["Clusters"], axis=1)
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
# - The value of the mean expression of the top 20 HKGs as standard : Std
Std = df_Std_start_restricted.at[sample_at_the_limit,"mean_top20_HKGs_sample_limit"]

#>>>>>compute alpha as in Mg < alpha x Std
# at the frontier to enter the low end of ATIP3 expression values (MG), we have Mg = alpha x Std (solve it to find value alpha)
alpha = Mg / Std
#---overall results to report
dict_overall_results = {}
dict_overall_results["Searches"] = ["Mg","Std","alpha"]
dict_overall_results["Results"] = [Mg,Std,alpha]
# create the report df
df_overall_results = pd.DataFrame(dict_overall_results)
# save the report df
if cohort_used == "REMAGUS02":
	name_overall_results = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_overall_results.csv"
elif cohort_used == "REMAGUS04":
	name_overall_results = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_overall_results.csv"
else : # cohort_used == "MDAnderson":
	name_overall_results = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_overall_results.csv"

df_overall_results.to_csv(name_overall_results, index=None, header=True)






