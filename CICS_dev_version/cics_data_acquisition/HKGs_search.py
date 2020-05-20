#---------------------get the HKGs
#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import pandas as pd
import numpy as np # linear algebra and exploit arrays faster and easier computations
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
df_file = df_file.set_index('Model')
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
df_file_fts_only = df_file.drop(list_of_cols_to_remove, axis=1)

#>>>>>>>>>>>>>>>>>>>getting to the HKGs selection
# 1-additionnal requirements
# make df with new col as median intensity for all gene
df_file_fts_only_w_median = df_file_fts_only
df_file_fts_only_w_median["median_all_fts"] = df_file_fts_only_w_median.median(axis=1, skipna=True)
# keep the number of samples
num_of_samples = len(df_file_fts_only) #
# keep a list of the cols that are fts (long names of genes)
list_of_cols_being_fts = list(df_file_fts_only.columns) # or use list_of_cols_being_fts = list_of_cols[:len(list(df_file_fts_only_w_median.columns))-1]

# 2-make new line as for a gene, top_HKG_criteria_1 (for all samples, Y if sample value > new col val of the sample, else N)
dict_ft_topHKGcriteria1={}
for a_ft in list_of_cols_being_fts:
	num_of_samples_w_criteria1_valid = (df_file_fts_only_w_median[a_ft] > df_file_fts_only_w_median['median_all_fts']).sum()
	# dict_ft_topHKGcriteria1[a_ft] = num_of_samples_w_criteria1_valid # to test
	if num_of_samples_w_criteria1_valid == num_of_samples:
		dict_ft_topHKGcriteria1[a_ft]="Y"
	else:
		dict_ft_topHKGcriteria1[a_ft] = "N"
# make a new df with the criteria 1 result as row and the fts as fts
df_crit1_lineonly = pd.DataFrame(dict_ft_topHKGcriteria1, index=['top_HKG_criteria_1'])

# 3-make new line as, mean across samples for each gene
df_crit2 = df_file_fts_only[list_of_cols_being_fts]
df_crit2.loc["mean"] = df_file_fts_only[list_of_cols_being_fts].mean(axis=0)
# make new line as for a gene, sd related to mean
df_crit2.loc["std"] = df_file_fts_only[list_of_cols_being_fts].std(axis=0, skipna=True)
# make new line as for a gene, CV
df_crit2.loc['CV'] = df_crit2.loc['std'] / df_crit2.loc['mean']
# make new line as for a gene, top_HKG_criteria_2 (Y, if CV < 0.35, else N)
dict_ft_topHKGcriteria2={}
for a_ft in list_of_cols_being_fts:
	limit_val_of_CV = 0.35
	if df_crit2.at["CV",a_ft] < limit_val_of_CV:
		dict_ft_topHKGcriteria2[a_ft] = "Y"
	else:
		dict_ft_topHKGcriteria2[a_ft] = "N"
# make a new df with the criteria 1 result as row and the fts as fts
df_crit2_lineonly = pd.DataFrame(dict_ft_topHKGcriteria2, index=['top_HKG_criteria_2'])

# 4-make new line as for a gene, top_HKG_status, (Y, if top_HKG_criteria_1 =Y & top_HKG_criteria_2 =Y, else N)
dict_ft_topHKGstatus={}
for a_ft in list_of_cols_being_fts:
	if (df_crit1_lineonly.at["top_HKG_criteria_1",a_ft] == "Y") & (df_crit2_lineonly.at["top_HKG_criteria_2",a_ft] == "Y"):
		dict_ft_topHKGstatus[a_ft] = "Y"
	else:
		dict_ft_topHKGstatus[a_ft] = "N"
# the new df with top_HKG_status as row and fts as cols
df_topHKGstatus_lineonly = pd.DataFrame(dict_ft_topHKGstatus, index=['top_HKG_status'])

# 5-get list of cols/genes and df with only top_HKG_status=Y
list_of_cols_being_top_HKGs = []
list_of_cols = list(df_topHKGstatus_lineonly.columns)
for a_ft in list_of_cols:
	if df_topHKGstatus_lineonly.at["top_HKG_status",a_ft] == "Y":
		list_of_cols_being_top_HKGs.append(a_ft)
# the df with only cols as top_HKG_status=Y
df_topHKGs_cols = df_topHKGstatus_lineonly[list_of_cols_being_top_HKGs]

# 6-make overall df of results by restricting to these rows : mean across samples, sd related to mean, top_HKG_status
df_mean_sd_CV_of_HKGs = df_crit2[list_of_cols_being_top_HKGs].loc[["mean","std","CV"],:]
# transform cols in rows in previous df, sort rows by the col CV, and add col rank
df_mean_sd_CV_of_HKGs_to_sort = df_mean_sd_CV_of_HKGs.transpose()
df_mean_sd_CV_of_HKGs_sortedOnCV= df_mean_sd_CV_of_HKGs_to_sort.sort_values("CV", axis=0, ascending=True, kind='mergesort')
# df3_sorted_cv_then_std = df3.sort_values(["CV","std"], axis=0, ascending=[True,True]) #  not working to order using 2 cols
# change the fts/genes from an index to a col named "HKGs" ("HKGs" here are the HKGs from our 2 criterias)
df_topHKGs_sorted = df_mean_sd_CV_of_HKGs_sortedOnCV.rename_axis('HKGs').reset_index()
# get a new col "rank of CV"
df_topHKGs_sorted['CV_Rank'] = df_topHKGs_sorted['CV'].rank(ascending=1)
df_topHKGs_sorted['CV_Rank'] = df_topHKGs_sorted['CV_Rank'].astype(int) # this is just to make the new col values as int like 132 instead of float like 132.0
# a function to get the short gene name from the long column name
def full_name_ft2_short_name(full_name):
	list_from_split1 = full_name.split("GSas")
	if len(list_from_split1)==1:
		short_name = "NA"
	else :
		kept_from_list1 = list_from_split1[1]
		list_from_split2 = kept_from_list1.split("wGBANas")
		kept_from_list2 = list_from_split2[0]
		short_name = kept_from_list2
	return short_name
# get a new col "HKG_GS", remove the rows where "HKG_GS" = "NA", redo the index
df_topHKGs_sorted["HKG_GS"] = df_topHKGs_sorted["HKGs"].apply(lambda x: full_name_ft2_short_name(x))
len_b4_removal_NA_as_GS_probes = len(df_topHKGs_sorted)
indexes2drop = df_topHKGs_sorted[df_topHKGs_sorted['HKG_GS'] == "NA"].index
df_topHKGs_sorted_noNAasGS = df_topHKGs_sorted.drop(indexes2drop , inplace=False)
df_topHKGs_sorted_noNAasGS = df_topHKGs_sorted_noNAasGS.reset_index(drop=True)
len_after_removal_NA_as_GS_probes = len(df_topHKGs_sorted_noNAasGS)
num_probes_wo_GS = len_b4_removal_NA_as_GS_probes - len_after_removal_NA_as_GS_probes
print("Number of probes without GS known :", num_probes_wo_GS)
# get the first mentions only, in the CV ranking (for security mesures we rank using the ranks)
df_topHKGs_sorted_1stmentions_only = df_topHKGs_sorted_noNAasGS.sort_values('CV_Rank', ascending=True).drop_duplicates('HKG_GS').sort_index()
# full resultings dfs of the first mentions (using a corrected rank of CV) (2 dfs : 1 sorted on CV_Rank_no_dup, 1 sorted on alphabetical order of HKG_GS)
df_topHKGs_sorted_1stmentions_only['CV_Rank_no_dup'] = df_topHKGs_sorted_1stmentions_only['CV_Rank'].rank(ascending=1)
df_topHKGs_sorted_1stmentions_only['CV_Rank_no_dup'] = df_topHKGs_sorted_1stmentions_only['CV_Rank_no_dup'].astype(int) # this is just to make the new col values as int like 132 instead of float like 132.0
df_topHKGs_sorted_1stmentions_only_genesInAlphabetOrder = df_topHKGs_sorted_1stmentions_only.sort_values("HKG_GS", axis=0, ascending=True, kind='mergesort')
# resultings dfs restricted to the top 20 of the first mentions (using a corrected rank of CV) (2 dfs : 1 sorted on CV_Rank_no_dup, 1 sorted on alphabetical order of HKG_GS)
df_topHKGs_sorted_1stmentions_only_top20 = df_topHKGs_sorted_1stmentions_only.head(20)
df_topHKGs_sorted_1stmentions_only_top20_genesInAlphabetOrder = df_topHKGs_sorted_1stmentions_only_top20.sort_values("HKG_GS", axis=0, ascending=True, kind='mergesort')
# save the report df
if cohort_used == "REMAGUS02":
	ourHKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_HKGsCVnoDupRanked_report.csv"
	ourHKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_HKGsAlphabetOrder_report.csv"
	ourTop20HKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_Top20HKGsCVnoDupRanked_report.csv"
	ourTop20HKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_Top20HKGsAlphabetOrder.csv"
elif cohort_used == "REMAGUS04":
	ourHKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_HKGsCVnoDupRanked_report.csv"
	ourHKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_HKGsAlphabetOrder_report.csv"
	ourTop20HKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_Top20HKGsCVnoDupRanked_report.csv"
	ourTop20HKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_Top20HKGsAlphabetOrder.csv"
else : # cohort_used == "MDAnderson":
	ourHKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_HKGsCVnoDupRanked_report.csv"
	ourHKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_HKGsAlphabetOrder_report.csv"
	ourTop20HKGsCVnoDupRanked_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_Top20HKGsCVnoDupRanked_report.csv"
	ourTop20HKGsAlphabetOrder_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_Top20HKGsAlphabetOrder.csv"

df_topHKGs_sorted_1stmentions_only.to_csv(ourHKGsCVnoDupRanked_report, index=None, header=True)
df_topHKGs_sorted_1stmentions_only_genesInAlphabetOrder.to_csv(ourHKGsAlphabetOrder_report, index=None, header=True)
df_topHKGs_sorted_1stmentions_only_top20.to_csv(ourTop20HKGsCVnoDupRanked_report, index=None, header=True)
df_topHKGs_sorted_1stmentions_only_top20_genesInAlphabetOrder.to_csv(ourTop20HKGsAlphabetOrder_report, index=None, header=True)


# >>>>>>>>>>>>>>>>7- summary comparison of our our HKGs vs the top 2O HKGs advised for experiments
# make a dictionnary keys as cols and values as content of cols to create a summarizing df
dict_HKGs_advised_for_exp = {}
# create the 1st column of the future df
dict_HKGs_advised_for_exp["HKGs_advised"] = ["CALR","ACTG1","GAPDH","RPS27A","ACTB","RPS20","HNRPD","NACA","NONO","UBC",
											 "RPL38","RPL11","PTMAP7","GFRA4","RPL7","CDC42","EIF3H","RPS11","RPL26L1","UBE2D3"]
# create the 2nd column of the future df
dict_HKGs_advised_for_exp["Mean_Rank"] = list(range(1,21)) # value is a list from 1 to 20, step 1
# - making the two lists containing positions and names of our HKGs that are similar to one of the advised top 20 HKGs
# a list as container of lists of positions (1 list in it is for one advised HKG)
list_of_lists_of_ranks_of_advised_HKGs_found = []
# a list as container of lists of similar in our HKGs (1 list in it is for one advised HKG)
list_of_lists_of_similars_of_advised_HKGs_found = []
# the lists to explore in our HKGs (one is the HKGs and other is their respective CV_Rank_no_dup
topHKGs_sorted_1stmentions_only_as_list = df_topHKGs_sorted_1stmentions_only['HKG_GS'].tolist()
CVrank_of_topHKGs_sorted_1stmentions_only_as_list = df_topHKGs_sorted_1stmentions_only['CV_Rank_no_dup'].tolist()
# for each of the advised HKGs, make the search in our HKGs and keep similars and their ranks
for an_HKG_advised in dict_HKGs_advised_for_exp["HKGs_advised"]:
	list_of_ranks_of_the_advised_HKG=[]
	list_of_similars_of_the_advised_HKGs=[]
	for one_of_our_top_HKGs_1stmentions in topHKGs_sorted_1stmentions_only_as_list:
		if an_HKG_advised in one_of_our_top_HKGs_1stmentions:
			index_of_the_top_HKG_where_its_found = topHKGs_sorted_1stmentions_only_as_list.index(one_of_our_top_HKGs_1stmentions)
			CVrank_to_keep = CVrank_of_topHKGs_sorted_1stmentions_only_as_list[index_of_the_top_HKG_where_its_found]
			list_of_ranks_of_the_advised_HKG.append(CVrank_to_keep) # keep the position in our CVranks
			list_of_similars_of_the_advised_HKGs.append(one_of_our_top_HKGs_1stmentions)  # keep the similar HKG where the advised HKG has been found
	# keep all the list of all positions where the advised HKG has been found in our CVranks
	list_of_lists_of_ranks_of_advised_HKGs_found.append(list_of_ranks_of_the_advised_HKG)
	# keep all the list of similars HKGs where the advised HKG has been found in our CVranks
	list_of_lists_of_similars_of_advised_HKGs_found.append(list_of_similars_of_the_advised_HKGs)

# create the 3rd column of the future df
dict_HKGs_advised_for_exp["CvrankNoDup_where_advised_HKG_been_found"] = list_of_lists_of_ranks_of_advised_HKGs_found
# create the 4th column of the future df
dict_HKGs_advised_for_exp["Similar_HKGs_where_advised_HKG_been_found"] = list_of_lists_of_similars_of_advised_HKGs_found
# create the report df on the advised HKGs positions
df_HKGs_advised_for_exp = pd.DataFrame(dict_HKGs_advised_for_exp)
# save the report df
if cohort_used == "REMAGUS02":
	name_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R02_advised_HKGs_report.csv"
elif cohort_used == "REMAGUS04":
	name_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/R04_advised_HKGs_report.csv"
else : # cohort_used == "MDAnderson":
	name_report = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/cics_data_acquisition/output/MDA_advised_HKGs_report.csv"

df_HKGs_advised_for_exp.to_csv(name_report, index=None, header=True)

# END OF PART>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
