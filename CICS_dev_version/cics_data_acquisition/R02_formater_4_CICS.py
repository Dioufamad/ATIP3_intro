#>>>>>>>>>>>>>>>>>>>>>>>>>>> REMAGUS02 DATASET FORMATTING FOR CICS SCRIPT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>> RE-ENCODING
# the file we are given is a table with the intent to represent values, each one corresponding to a variable and a sample

# 1st issue : the file might not be in a supported encoding so we have to reencode it in UTF-8
# get the file encoding using the following
# file -i REMAGUS02-Données\ genomique_226x54676\ totales.txt
# change the file encoding and save the new version in .tsv format (use the previous encoding in all capitals letters)
# iconv -f UTF-16LE -t UTF-8//IGNORE REMAGUS02-Données\ genomique_226x54676\ totales.txt > output2.tsv

#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# import os
# import sys
import pandas as pd # for dataframes manipulation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # to change the Response values from string to classes 0 and 1 # not needed at the moment
import numpy as np # linear algebra and exploit arrays faster and easier computations
import os #for bash command lines in python
import locale
# import matplotlib.pyplot as plt
# from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')
# import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline
from pathlib import Path # to manage paths as into arguments
from cics_engines.data_engine1_mgmt import data_mgmt_6
#>>>>>>>>>>>>>>>>>>>>>>>>>>> Variables to initialise------------------------------------------
# # ---- a string saying if to make a log or not # tag caught
# tag_decision_make_log = "yes"
# # tag_decision_make_log = "no"
# #~~~~~~~~~~~~~ redirection from stdout to a .o file step 1/2 ~~~~~~~~~~~>
# # A log file name Output_following_trialNameGivenByUser.0 is created
# tag_num_trial = "8001"
# if tag_decision_make_log in ["yes", "y"]:
# 	logbasedir = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs"
# 	original_out = sys.stdout
# 	sys.stdout = open(logbasedir + "/" + "Output_following_" + tag_num_trial + ".o", 'w')
#----------start writing in log or terminal
print("Initialising environnement variables...")
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
# ----memory clearing choice
# clear_mem = "yes"
clear_mem = "no"
#----position of samples in the final table
tag_decision_move_samples_col_at_last_pos = "yes"
# tag_decision_move_samples_col_at_last_pos = "no"
#----choose between dropping rows and dropping columns when nan values are encountered
# 3 strategies : A-imputing the missing value , B-get rid of the column, C-get rid of the row
# A is not easy to implement so add it later ##!
# B means the features in question are not important for the model and we can do well without => tag_decision_nan_del_samples_or_fts = "F"
# C is we have enough samples to let go off somes. => tag_decision_nan_del_samples_or_fts = "S"
##! later give the user a choice between C and B ##! add also a 3rd option that will study the percentage of nan in columns or rows and make a choice past a threshold
# tag_decision_nan_del_samples_or_fts = "F" # stratefy B
tag_decision_nan_del_samples_or_fts = "S" # stratefy C
# -----needed to save the output dataset
basedir = str(Path()) #for setting the working directory to create the paths to the location of the output dataset
#---how to launch the script
# tag_decision_launching_way = "cmd_line"
tag_decision_launching_way = "line_by_line"
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
if command_center == "Gustave_Roussy":
	rest_of_abs_path_b4_content_root = "/home/amad/PycharmProjects/ATIP3_in_GR/"
else : # command_center = "Home"
	rest_of_abs_path_b4_content_root = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/"
print("command center used recognized...")
# ----for the cohort choice
# cohort_used = "REMAGUS02"
cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
# ----for the response strategy
# resp_used = "RCH3HSall"
# resp_used_in_full = "All the samples with -defined or not RCH and 3 hormonals status, are kept"
# resp_used = "RCH3HSdefined"
# resp_used_in_full = "Only defined -RCH and the 3 hormonals status- samples are kept"
# resp_used = "RCHdefined"
# resp_used_in_full = "Only defined RCH samples are kept"
# resp_used = "TNBCdefined"
# resp_used_in_full = "Only defined TNBC samples are kept"
resp_used = "RCHandTNBCdefined"
resp_used_in_full = "Only defined RCH and TNBC samples are kept"
print("For population restriction, the response strategy chosen is",resp_used_in_full,"(",resp_used,")...")
#---for the files to manipulate
# stock the file and its separator # whatever command_center is chosen
if cohort_used == "REMAGUS02":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS02_Donnees_genomiques_226x54676_totales.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS02-Données cliniques.xls" # @ home
	sheet_id = "extractionCNahmias"
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annotations_ready4use/REMAGUS02_PSI_GS_54675.csv"
	sep_in_file4annot = ","
elif cohort_used == "REMAGUS04":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS04normData_R04_142_CNahmias.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS04-Données cliniques.xlsx" # @ home
	sheet_id = "ExtracCNahmias2"
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annotations_ready4use/REMAGUS04_PSI_GS_22215.csv"
	sep_in_file4annot = ","
elif cohort_used == "MDAnderson":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/MDAnderson_MDA133_Expression.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/MDAnderson-MDA133CompleteInfo20060418.xls" # @ home
	sheet_id = "MDA133CompleteInfo20060418"
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annotations_ready4use/MDAnderson_PSI_GS_22215.csv" # uses the same as REMAGUS04 or made from copy of it
	sep_in_file4annot = ","
else: # when no cohort is chosen
	file_path = "Unknown"
	sep_in_file = "Unknown"
	supporting_file_path = "Unknown"
	sheet_id = "Unknown"
	file4annot_path = "Unknown"
	sep_in_file4annot = "Unknown"
	print("No cohort has been chosen for preprocessing")
print("paths to files manipulated and related specificities stored...")
# -----end of imports and settings
print("Necessary libraries imported.")
print("Environnement variables initialised.")
print("All imports and settings are successfully placed")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> README <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Welcome in the formatting tool for the Remagus02, Remagus04, and MDAanderson datasets in direction of CICS")
print("We suppose you have done the querying of a database and you have separated values files (csv,tsv,xls, etc.).")
print("Such values tables describe samples over multiples features ie rows are samples and features are columns or vice-versa.")
print("We will try to format it into this representation : ")
print("- a .csv file that have successively 3 groups of columns as features with features names as the titles of the columns")
print("+ 1 group of columns, as the responses that will be used to create the response column containing the classes used to extract rules by learning methods ")
print("+ 1 group of columns, as the features ie variables potentially used in the models by learning methods ")
print("+ 1 column as the samples that are the exemples used to extract rules by learning models")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESSING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# --------------proper processing
print("Starting preprocessing...")
# Objective : joining both tables (the one with the features and the one with the response) into one unique table Responses-Features-Samples
#>>>>>-----start of everything that is data file specific in the preprocessing
if cohort_used == "REMAGUS02" :
	print("Storing files contents in dataframes...")
	df_file = pd.read_csv(file_path,sep_in_file) ##! add skiprows=0 to skip lines 0 lines here, default is None
	df_sup_file = pd.read_excel(supporting_file_path,sheet_id)
	df_file4annot = pd.read_csv(file4annot_path,sep_in_file4annot)

	print("PART I -----> put in form the content of supplementary table (LEFT)...")
	print("-------step 1 : rename the targeted columns on both left and right dataframes")
	# we will need to capture columns and move them around or edit them for restricting, joining the columns etc.
	# it is better to name the manipulated columns and their give also their future names
	# needed columns : sample col, response col
	old_Samples_col_name_left = "CLETRI" # given samples col name by the user ##! to get from the argument
	Samples_col_name_left = "Model_bis" # for the left table
	old_Samples_col_name_right = "cletri"
	# NB : the 2 samples columns of the 2 tables must have different names to join them later
	Samples_col_name_right = "Model" # for the right table
	common_samples_id_prefix = "CLETRI"
	print("----step 2 : A-restricting the support info table to only the raw needed columns...There are : ") # strategy : selected only the needed columns because they are not a lot
	list_of_only_needed_cols_in_sup_table = [old_Samples_col_name_left]+["RCH","RO","RP","HER2"]
	for col_name in list_of_only_needed_cols_in_sup_table:
		print("-",col_name)
	# df_sup_file = df_sup_file[[old_Samples_col_name_left,old_Resp_col_name_left]]  #garb
	df_sup_file = df_sup_file[list_of_only_needed_cols_in_sup_table]

	print("-------step 2 : B-restriction of responses following chosen to be kept response")
	# lets define needed functions
	# - a function to get rid of undefined values in responses or samples
	def rows_w_nan_dropper(df_to_clean):
		print("-------step 4 : eliminating all samples unidentified or without response (with nan values)...")  # also a way to make sure the joining with the right table later goes smootly
		samples_b4_cleaning = len(df_to_clean.axes[0])
		df_to_clean.dropna(axis='index', inplace=True)
		samples_aft_cleaning = len(df_to_clean.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete samples info of the left table : ")  # a report on the losses while cleaning
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete samples info of the left table")
		else:
			print(lost_samples,"of",samples_b4_cleaning,"samples have been lost during the cleaning of the uncomplete samples info of the left table")
		return df_to_clean
	# - the fonction to get the triple neg bool values in the new resp col
	def tn_maker(row):
		if row["RO"] == 0 and row["RP"] == 0 and row["HER2"] == 0:
			return 1
		else:
			return 0

	print("-----> following a response choice, restricting the population to the individuals to use (LEFT)...")
	if resp_used == "RCHdefined":
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["RCH"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif (resp_used == "TNBCdefined") | (resp_used == "RCHandTNBCdefined"):
		old_Resp_col_name_left1 = "RO"  # capture of the 3 cols of interest
		old_Resp_col_name_left2 = "RP"
		old_Resp_col_name_left3 = "HER2"
		old_Resp_col_name_left = "TNBC"  # the name of the new resp col
		df_sup_file[old_Resp_col_name_left] = df_sup_file.apply(lambda row: tn_maker(row),axis=1)  # getting the new col at last pos
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		if resp_used == "TNBCdefined" :
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["TNBC"]
		else : # resp_used == "RCHandTNBCdefined"
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["RCH","TNBC"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif resp_used == "RCH3HSdefined":
		# list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left]+["RCH","RO","RP","HER2"] # already at this state # not needed but leave her for reference
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	else:  # no response is preferred ie all 4 are kept wo deleting nan values
		print("From the left table, all responses with or without undefined values are kept.")
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : A1-renaming the 2 kept columns (samples, response used to restrict population) for the left table... ")
	dict_keptlefttablecolsnames_old2new = {old_Samples_col_name_left: Samples_col_name_left, "RCH" : "BestResCat_as_RCH", "TNBC" : "BestResCat_as_TNBC" ,"RO" : "BestResCat_as_RO", "RP" : "BestResCat_as_RP", "HER2" : "BestResCat_as_HER2"}
	remaining_colsnames = df_sup_file.columns
	for a_remaining_colname in remaining_colsnames:
		df_sup_file.rename(columns={a_remaining_colname:dict_keptlefttablecolsnames_old2new[a_remaining_colname]}, inplace=True)
	num_resp_cols_from_sup = len(list(remaining_colsnames))-1 # keept this number for future indexing of responses columns group
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_sup_file[Samples_col_name_left] = common_samples_id_prefix + "_" + df_sup_file[Samples_col_name_left].astype(str) # sample_name is a string ColumnName_IdInColumn #(340,5)
	print("-------step 3 : C-facilitating the merge to come: sorting the left dataframe based on samples entries and dropping the rows that already exist")
	df_sup_file = data_mgmt_6(df_sup_file, Samples_col_name_left) # sorting
	df_sup_file_w_marked_as_duplicates_rows_only = df_sup_file[df_sup_file.duplicated()] # dropping duplicates rows
	df_sup_file_indexes_to_drop = df_sup_file_w_marked_as_duplicates_rows_only.index
	df_sup_file = df_sup_file.drop(df_sup_file_indexes_to_drop)
	df_sup_file = df_sup_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_sup_file_w_marked_as_duplicates_rows_only
		del df_sup_file_indexes_to_drop

	print("PART II -----> put in form the content of features table (RIGHT)")
	print("-------step 5 : changing columns into rows because the present rows are not the samples...")
	df_file = df_file.transpose()
	print("-------step 6 : making the index as a column (the index was the sample names)...")  # by resetting the index in a way to get the older index as a column)
	df_file = df_file.reset_index()
	print("-------step 7 : making a copy of the first line and use it as titles of the columns...")
	df_file.columns = df_file.iloc[0]
	print("-------step 8 : dropping the first line because it is now the titles of the columns...")
	df_file = df_file.drop(df_file.index[0])
	print("-------step 9 : resetting the index for a correct order (previous dropping messed it up)...") #the index is missing now a the 1st line entry value (0). reset it in a way to not get a new column
	df_file = df_file.reset_index(drop=True)
	print("-------step 10 : restricting the right table to only the needed columns (dropping unecessary columns)...") # not necessary columns are dropped because they are not a lot
	list_of_unecessary_cols_2_drop = ["CLETRI"] # a column that is just a repetiton of the sample names col
	df_file.drop(labels=list_of_unecessary_cols_2_drop, axis=1, inplace=True)
	print("-------step 11 : A-renaming the sample column... ")
	df_file.rename(columns={old_Samples_col_name_right: Samples_col_name_right}, inplace=True)
	print("-------step 11 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_file[Samples_col_name_right] = common_samples_id_prefix + "_" + df_file[Samples_col_name_right].astype(str) # sample_name is a string ColumnName_IdInColumn
	#>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : C-removing cells with nan...")
	# as response is clear and samples should be also, any nan left in the joinded table is in the features values
	if tag_decision_nan_del_samples_or_fts in ["samples", "S"]: # - choose between dropping rows and dropping columns # (prefer to drop rows)
		samples_b4_cleaning = len(df_file.axes[0])
		df_file.dropna(axis='index', inplace=True)  # losing samples
		samples_aft_cleaning = len(df_file.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning # lost_samples,"of",samples_b4_cleaning,"samples have been lost
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_samples,"of",samples_b4_cleaning,"samples have been lost during the cleaning of the uncomplete features info of the right table")
	elif tag_decision_nan_del_samples_or_fts in ["features", "F"]: # - choose between dropping rows and dropping columns # (prefer to drop cols)
		features_b4_cleaning = len(df_file.axes[1])
		df_file.dropna(axis='columns', inplace=True)  # losing fts
		features_aft_cleaning = len(df_file.axes[1])
		lost_features = features_b4_cleaning - features_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning
		if lost_features == 0:
			print("No features has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_features,"of",features_b4_cleaning,"samples have been lost during the cleaning of the uncomplete features info of the right table")
	else:  # estimate the % of nan values in the row and the column, then delete the axis with more % and if below a certain %, impute it if user allow it
		print("For the cleaning of the uncomplete features info of the joined table based on missing values percentage has still to be implemented")
	#>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : D-facilitating the merge to come: sorting the right dataframe based on samples entries and dropping the rows that already exist")
	df_file = data_mgmt_6(df_file, Samples_col_name_right) # sorting
	df_file_w_marked_as_duplicates_rows_only = df_file[df_file.duplicated()] # dropping duplicates rows
	df_file_indexes_to_drop = df_file_w_marked_as_duplicates_rows_only.index
	df_file = df_file.drop(df_file_indexes_to_drop)
	df_file = df_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file_w_marked_as_duplicates_rows_only
		del df_file_indexes_to_drop

	print("PART III -----> formatting every group of columns in the final frame for the content of the .csv response-features-samples files")
	print(">>>>>-the awaited configuration is :")
	print("- 1 group of columns (the responses) that can have anything as dtype, (int64/float64/object) and that is why we encode it in string (dtype object)")
	print("- 1 group of columns that is int64/float64 dtype but the one and the same dtype,  that we format in bools or floats")
	print("- 1 column strings (samples names) of dtype object")
	print("-------step 12 : joining of the features and response tables...") # joined on the sample name columns # inner: use intersection of keys from both frames
	df_joined = pd.merge(df_sup_file, df_file, how="inner", left_on=Samples_col_name_left, right_on=Samples_col_name_right)
	print("-------step 13 : dropping the extra sample column present after the joining of left and right tables...")
	df_joined.drop(labels=[Samples_col_name_left], axis=1, inplace=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file # clear memory
		del df_sup_file # clear memory


	print("-------step 17 : A-putting the samples column at last position (no impact on the CICS preprocessing) and tagging the 3 groups of columns to manipulate them easier......") # if decided by user,
	# strategy : move around the col names and use the new list to build new dataframe
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]:
		df_joined_initial_cols_list = list(df_joined.columns)
		resp_cols_list = df_joined_initial_cols_list[0:num_resp_cols_from_sup] # num_resp_cols_from_sup = the first 4 cols if the 4 responses kept initially
		samples_cols_list = [df_joined_initial_cols_list[num_resp_cols_from_sup]]
		feat_cols_list = df_joined_initial_cols_list[(num_resp_cols_from_sup+1):] # we want to get the following column thats why we make + 1
		df_joined_cols_reordered_list = resp_cols_list + feat_cols_list + samples_cols_list
		# df_joined_cols_reordered_list = [df_joined_initial_cols_list[0]] + df_joined_initial_cols_list[2:] + [df_joined_initial_cols_list[1]] # garb
		df_joined = df_joined[df_joined_cols_reordered_list]
		# the 3 groups of columns
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		feat_cols = df_joined.columns[num_resp_cols_from_sup:-1]
		samples_cols = df_joined.columns[-1]
		state_of_samples_pos = "at_last"    # a tag to known in the following the state of the position of the samples col
	else:
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		samples_cols = df_joined.columns[num_resp_cols_from_sup]
		feat_cols = df_joined.columns[(num_resp_cols_from_sup+1):]
		state_of_samples_pos = "not_at_last"

	#####-----checkpoint
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns) in order to know what dtypes to convert...")
	df_joined_preview = df_joined.iloc[:, list(range(6)) + [-5,-4,-3,-2,-1]]
	print(df_joined_preview.info()) # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----
	print("-------step 17 : B-formatting the dtypes of each group of columns : put the samples name in dtype string (object)...") ##! help also to do before computing on fts vlues
	df_joined[samples_cols] = df_joined[samples_cols].astype(str) # strings dtype is object so we have to find object

	print("-------step 17 : C1-formatting the dtypes of each group of columns : put the features values in float dtype...")
	# - replace the commas blocking the conversion of objects in floats
	df_joined_feat_cols_only_no_commas = df_joined[feat_cols].replace(",", ".", regex=True)
	# - a fast method used to change all fts values into floats
	# old_fts_col_names = df_joined[feat_cols].columns # not needed because old_fts_col_names is feat_cols
	df_fts_as_series = df_joined_feat_cols_only_no_commas.values.astype(np.float64)
	df_fts_back_as_df = pd.DataFrame(df_fts_as_series)
	df_fts_back_as_df.columns = feat_cols
	# df_joined[feat_cols] = df_fts_back_as_df
	# instead of putting the galerie of df_fts back in the df_joined, just take the 2 remaining groups of cols of df_joined and add them to df_fts to reform the old df_joined called now dframe
	#-putting back the resp cols
	for col_of_resp_to_put_back in resp_cols:
		# print(col_of_resp_to_put_back) # test
		original_index_of_resp_col_to_put_back = list(df_joined.columns).index(col_of_resp_to_put_back)
		df_fts_back_as_df.insert(original_index_of_resp_col_to_put_back, col_of_resp_to_put_back, df_joined[col_of_resp_to_put_back]) # reput the resp col at the front ## last_stop_to_retouch
	#-putting back sample col
	original_index_of_sample_col_to_put_back = list(df_joined.columns).index(samples_cols)
	df_fts_back_as_df.insert(original_index_of_sample_col_to_put_back, samples_cols,df_joined[samples_cols])
	# depending if the fts col are at the end or not, it is still the same with previous line # also len(df_fts_back_as_df.axes[0]) = what would be the index of a new col as last

	print("-------step 17 : C2-renaming the fts using a table of annotations...")
	#---Renaming the fts
	# - read the file containg the table of annotations
	# NB : putting back the fts_cols selector as it should be is not needed because still the same columns in the fts group
	old_fts_cols_list = feat_cols.values.tolist()	# old way => old_fts_cols_list = feat_cols.values.tolist()
	colname_of_previous_states = "PSI"
	df_file4annot.sort_values(colname_of_previous_states, axis=0, ascending=True, inplace=True, kind='mergesort') # sort the df following the values of the probesets
	colname_of_after_states = "GS"
	dict_previous_after_states = dict(zip(df_file4annot[colname_of_previous_states], df_file4annot[colname_of_after_states]))
	def fts_names_converter(old_list, dict2convertkeyinvalue):
		new_list = [item if not item in dict2convertkeyinvalue else dict2convertkeyinvalue[item] for item in old_list]
		return new_list
	new_fts_cols_list = fts_names_converter(old_fts_cols_list,dict_previous_after_states)
	dict2renamecols = dict(zip(old_fts_cols_list, new_fts_cols_list))
	df_fts_back_as_df.rename(columns=dict2renamecols, inplace=True)
	# lets put back the fts_cols selector as it should be in case we need it again
	feat_cols_list_b4_change = feat_cols
	if state_of_samples_pos == "at_last": # decide where are the fts
		feat_cols = df_fts_back_as_df.columns[num_resp_cols_from_sup:-1]
	else:
		feat_cols = df_joined.columns[(num_resp_cols_from_sup+1):]

	print("----step 17 : E-A report on if all the initial features names have been successfully found and changed...")
	# use this following to find common elements between old columns list and new columns list :
	# list1_as_set = set(list1) and set_intersection = list1_as_set.intersection(list2) and then intersection_as_list = list(intersection)
	list_of_unchanged_fts = list(set(list(feat_cols_list_b4_change)).intersection(list(feat_cols)))
	# list_of_unchanged_fts = [] # slow so deprecated
	# for ft_name in list(feat_cols):
	# 	if ft_name in list(feat_cols_list_b4_change):
	# 		list_of_unchanged_fts.append(ft_name)
	if len(list_of_unchanged_fts) != 0:
		print(len(list_of_unchanged_fts), "initial features have not been found in the conversion dictionnary, hence their names not changed.")
	else:
		print("All initial features have been found in the conversion dictionnary andchanged")

	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_fts_back_as_df_preview = df_fts_back_as_df.iloc[:, list(range(6)) + [-5,-4,-3,-2,-1]]
	print(df_fts_back_as_df_preview.info()) # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----

	# df ready for response computations
	df_aft_resp = df_fts_back_as_df

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_fts_back_as_df # clear memory
		del df_fts_as_series # clear memory
		del df_joined # cleaning



	print("-------step 18 : D2-formatting the dtypes of each group of columns : put the response values in 2 dtype string (object) Res and Sen (Neg and Pos) to be able to read easier any contigency table...")
	list_of_resp_col_indexes_for_ResSen_changes = [0]
	for a_resp_col in resp_cols:
		if list(resp_cols).index(a_resp_col) in list_of_resp_col_indexes_for_ResSen_changes:
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:, [a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2:  # start modifs
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res", "Sen"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			elif len(RespClasses_list_wo_nan) == 1:
				if RespClasses_list_wo_nan[0] == 1:
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Sen"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
				elif RespClasses_list_wo_nan[0] == 0:
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			else:  # there is more than 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # end modifs
		else: # the others responses that have Neg and Pos
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:,[a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2 :
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan,["Neg","Pos"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str) # response can be string dtype as it will be encoded later
			else : # there is not 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)


	# sort the dataframe entries following response col values and remake a new index (resp col 1 is used in multiple responses data)
	# sort the df following the values of the resp column
	df_aft_resp.sort_values(list(resp_cols)[0], axis=0, ascending=True, inplace=True, kind='mergesort')
	# after the precedent sort, the indexes are not in order. make a new order for them
	df_aft_resp = df_aft_resp.reset_index(drop="True")
	print("Describing the obtained final samples-features-response frame...")
	total_samples = len(df_aft_resp.axes[0])
	total_resps = len(list(resp_cols))
	total_feats = len(df_aft_resp.axes[1])-(total_resps+1) # withdraw of the total 1 sample col and the response col
	print("The frame to analyse has", total_samples,"samples,",total_feats ,"features and",total_resps,"response(s)")
	# rport on the responses individually
	for resp_name in resp_cols:
		resp_name_full_col = df_aft_resp.loc[:,[resp_name]] # get the sorted unique values in it
		classes_detected_incl_nan = sorted(resp_name_full_col.iloc[:, 0].unique())
		num_classes_detected_incl_nan = len(classes_detected_incl_nan)
		print("- Response", resp_name, "has",num_classes_detected_incl_nan," classes : ")
		for class_detected in classes_detected_incl_nan: # go over the present response
			count_of_class_detected = df_aft_resp[resp_name].value_counts()[class_detected] # before it was using dframe.iloc[:, 0]
			count_perc_of_class_detected = (count_of_class_detected / total_samples) * 100
			print("- - a class value", class_detected, "is found on", count_of_class_detected, "samples counting for",'{:.3f}'.format(count_perc_of_class_detected), "% of the samples")


	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_aft_resp_preview = df_aft_resp.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_aft_resp_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----


	dframe = df_aft_resp # the final dataframe to write in a .csv file

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_aft_resp # clear mem

	print("-----step 20 : Saving a copy of the final dataframe in a .csv file...")
	tag_ctype = "BRCA"
	tag_drugname = "REMAGUS02_NAC" # manually recordd in the treatments details files # NAC = NeoAdjuvant Chemotherapy
	tag_drugID = "Treatment11"
	# tag_resp_kept = resp_used
	tag_respType = str(cohort_used) + "x" + "NAC"+ "x" + str(total_samples) + "S" + "x" + str(total_feats) + "F" + "x" + str(len(list(resp_cols))) + "Ras" + str(resp_used)
	tag_profilename = "GEX"
	# the output path has 3 parts : the root until the ouput folder, the output folder, and the filename
	# - lets make the file name
	output_filename_for_final_dframe = tag_ctype + "_" + tag_drugID + "_" + tag_respType + "_" + tag_profilename + ".csv"
	# - lets make the root until the ouput folder (the output folder excluded)
	if tag_decision_launching_way in ["cmd_line","cl"]: # launch the script in a terminal
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		root_until_output_folder = os.getcwd() # obtained by changing directory
	else: # launch the script line by line
		root_until_output_folder = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version" # whetever the command center
	# - lets make the output folder
	output_folder_on_same_lvl_than_main_name = "outputs"
	# - lets extend the root path to contain the output folder
	root_until_output_folder_w_output_folder = os.path.join(root_until_output_folder,output_folder_on_same_lvl_than_main_name)
	if not os.path.exists(root_until_output_folder_w_output_folder):
		os.mkdir(root_until_output_folder_w_output_folder)
	# - lets make the full path to the file to save
	fullname = os.path.join(root_until_output_folder_w_output_folder,output_filename_for_final_dframe)
	# - lets use the full path to save the file
	dframe.to_csv(fullname, index=None, header=True)
	print("File saved !")
	print(cohort_used,"dataset formatting for CICS analysis is done!")
	print("the file location is :",fullname)
	##! also delete all the uneccesary variables got sooner
	##! also create a log of all operations

elif cohort_used == "REMAGUS04" :
	print("Storing files contents in dataframes...")
	df_file = pd.read_csv(file_path, sep_in_file)  ##! add skiprows=0 to skip lines 0 lines here, default is None
	df_sup_file = pd.read_excel(supporting_file_path, sheet_id)
	df_file4annot = pd.read_csv(file4annot_path, sep_in_file4annot)

	print("PART I -----> put in form the content of supplementary table (LEFT)...")
	print("-------step 1 : rename the targeted columns on both left and right dataframes")
	# we will need to capture columns and move them around or edit them for restricting, joining the columns etc.
	# it is better to name the manipulated columns and their give also their future names
	# needed columns : sample col, response col
	old_Samples_col_name_left = "cletri"  # given samples col name by the user ##! to get from the argument
	Samples_col_name_left = "Model_bis"  # for the left table
	old_Samples_col_name_right = "cletri"
	# NB : the 2 samples columns of the 2 tables must have different names to join them later
	Samples_col_name_right = "Model"  # for the right table
	common_samples_id_prefix = "CLETRI"
	print("----step 2 : A-restricting the support info table to only the raw needed columns...There are : ")  # strategy : selected only the needed columns because they are not a lot
	list_of_only_needed_cols_in_sup_table = [old_Samples_col_name_left] + ["rch", "ro", "rp", "her"]
	for col_name in list_of_only_needed_cols_in_sup_table:
		print("-", col_name)
	# df_sup_file = df_sup_file[[old_Samples_col_name_left,old_Resp_col_name_left]]  #garb
	df_sup_file = df_sup_file[list_of_only_needed_cols_in_sup_table]

	print("-------step 2 : B-restriction of responses following chosen to be kept response")


	# lets define needed functions
	# - a function to get rid of undefined values in responses or samples
	def rows_w_nan_dropper(df_to_clean):
		print("-------step 4 : eliminating all samples unidentified or without response (with nan values)...")  # also a way to make sure the joining with the right table later goes smootly
		samples_b4_cleaning = len(df_to_clean.axes[0])
		df_to_clean.dropna(axis='index', inplace=True)
		samples_aft_cleaning = len(df_to_clean.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete samples info of the left table : ")  # a report on the losses while cleaning
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete samples info of the left table")
		else:
			print(lost_samples, "of", samples_b4_cleaning, "samples have been lost during the cleaning of the uncomplete samples info of the left table")
		return df_to_clean


	# - the fonction to get the triple neg bool values in the new resp col
	def tn_maker(row):
		if row["ro"] == 0 and row["rp"] == 0 and row["her"] == 0:
			return 1
		else:
			return 0


	print("-----> following a response choice, restricting the population to the individuals to use (LEFT)...")
	if resp_used == "RCHdefined":
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["rch"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif (resp_used == "TNBCdefined") | (resp_used == "RCHandTNBCdefined"):
		old_Resp_col_name_left1 = "ro"  # capture of the 3 cols of interest
		old_Resp_col_name_left2 = "rp"
		old_Resp_col_name_left3 = "her"
		old_Resp_col_name_left = "TNBC"  # the name of the new resp col
		df_sup_file[old_Resp_col_name_left] = df_sup_file.apply(lambda row: tn_maker(row), axis=1)  # getting the new col at last pos
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		if resp_used == "TNBCdefined":
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["TNBC"]
		else:  # resp_used == "RCHandTNBCdefined"
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["rch", "TNBC"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif resp_used == "RCH3HSdefined":
		# list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left]+["RCH","RO","RP","HER2"] # already at this state # not needed but leave her for reference
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	else:  # no response is preferred ie all 4 are kept wo deleting nan values
		print("From the left table, all responses with or without undefined values are kept.")
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : A1-renaming the 2 kept columns (samples, response used to restrict population) for the left table... ")
	dict_keptlefttablecolsnames_old2new = {old_Samples_col_name_left: Samples_col_name_left, "rch": "BestResCat_as_RCH", "TNBC": "BestResCat_as_TNBC", "ro": "BestResCat_as_RO", "rp": "BestResCat_as_RP", "her": "BestResCat_as_HER2"}
	remaining_colsnames = df_sup_file.columns
	for a_remaining_colname in remaining_colsnames:
		df_sup_file.rename(columns={a_remaining_colname: dict_keptlefttablecolsnames_old2new[a_remaining_colname]}, inplace=True)
	num_resp_cols_from_sup = len(list(remaining_colsnames)) - 1  # keept this number for future indexing of responses columns group
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_sup_file[Samples_col_name_left] = common_samples_id_prefix + "_" + df_sup_file[Samples_col_name_left].astype(str)  # sample_name is a string ColumnName_IdInColumn #(340,5)
	print("-------step 3 : C-facilitating the merge to come: sorting the left dataframe based on samples entries and dropping the rows that already exist")
	df_sup_file = data_mgmt_6(df_sup_file, Samples_col_name_left)  # sorting
	df_sup_file_w_marked_as_duplicates_rows_only = df_sup_file[df_sup_file.duplicated()]  # dropping duplicates rows
	df_sup_file_indexes_to_drop = df_sup_file_w_marked_as_duplicates_rows_only.index
	df_sup_file = df_sup_file.drop(df_sup_file_indexes_to_drop)
	df_sup_file = df_sup_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_sup_file_w_marked_as_duplicates_rows_only
		del df_sup_file_indexes_to_drop

	print("PART II -----> put in form the content of features table (RIGHT)")
	print("-------step 5 : the columns name are shifted one column towards the front so we shift them back by one clumn...")
	old_cols_list = list(df_file.columns)
	new_cols_list = [old_Samples_col_name_right] + old_cols_list[:-1]
	df_file.columns = new_cols_list
	print("-------step 5 : changing columns into rows because the present rows are not the samples...")
	df_file = df_file.transpose()
	print("-------step 6 : making the index as a column (the index was the sample names)...")  # by resetting the index in a way to get the older index as a column)
	df_file = df_file.reset_index()
	print("-------step 7 : making a copy of the first line and use it as titles of the columns...")
	df_file.columns = df_file.iloc[0]
	print("-------step 8 : dropping the first line because it is now the titles of the columns...")
	df_file = df_file.drop(df_file.index[0])
	print("-------step 9 : resetting the index for a correct order (previous dropping messed it up)...")  # the index is missing now a the 1st line entry value (0). reset it in a way to not get a new column
	df_file = df_file.reset_index(drop=True)
	# print("-------step 10 : restricting the right table to only the needed columns (dropping unecessary columns)...")  # not necessary columns are dropped because they are not a lot
	# list_of_unecessary_cols_2_drop = ["CLETRI"]  # a column that is just a repetiton of the sample names col
	# df_file.drop(labels=list_of_unecessary_cols_2_drop, axis=1, inplace=True)
	print("-------step 11 : A-renaming the sample column... ")
	df_file.rename(columns={old_Samples_col_name_right: Samples_col_name_right}, inplace=True)
	print("-------step 11 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_file[Samples_col_name_right] = common_samples_id_prefix + "_" + df_file[Samples_col_name_right].astype(str)  # sample_name is a string ColumnName_IdInColumn
	# >>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : C-removing cells with nan...")
	# as response is clear and samples should be also, any nan left in the joinded table is in the features values
	if tag_decision_nan_del_samples_or_fts in ["samples", "S"]:  # - choose between dropping rows and dropping columns # (prefer to drop rows)
		samples_b4_cleaning = len(df_file.axes[0])
		df_file.dropna(axis='index', inplace=True)  # losing samples
		samples_aft_cleaning = len(df_file.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print(
			"Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning # lost_samples,"of",samples_b4_cleaning,"samples have been lost
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_samples, "of", samples_b4_cleaning, "samples have been lost during the cleaning of the uncomplete features info of the right table")
	elif tag_decision_nan_del_samples_or_fts in ["features", "F"]:  # - choose between dropping rows and dropping columns # (prefer to drop cols)
		features_b4_cleaning = len(df_file.axes[1])
		df_file.dropna(axis='columns', inplace=True)  # losing fts
		features_aft_cleaning = len(df_file.axes[1])
		lost_features = features_b4_cleaning - features_aft_cleaning
		print(
			"Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning
		if lost_features == 0:
			print("No features has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_features, "of", features_b4_cleaning, "samples have been lost during the cleaning of the uncomplete features info of the right table")
	else:  # estimate the % of nan values in the row and the column, then delete the axis with more % and if below a certain %, impute it if user allow it
		print("For the cleaning of the uncomplete features info of the joined table based on missing values percentage has still to be implemented")
	# >>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : D-facilitating the merge to come: sorting the right dataframe based on samples entries and dropping the rows that already exist")
	df_file = data_mgmt_6(df_file, Samples_col_name_right)  # sorting
	df_file_w_marked_as_duplicates_rows_only = df_file[df_file.duplicated()]  # dropping duplicates rows
	df_file_indexes_to_drop = df_file_w_marked_as_duplicates_rows_only.index
	df_file = df_file.drop(df_file_indexes_to_drop)
	df_file = df_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file_w_marked_as_duplicates_rows_only
		del df_file_indexes_to_drop

	print("PART III -----> formatting every group of columns in the final frame for the content of the .csv response-features-samples files")
	print(">>>>>-the awaited configuration is :")
	print("- 1 group of columns (the responses) that can have anything as dtype, (int64/float64/object) and that is why we encode it in string (dtype object)")
	print("- 1 group of columns that is int64/float64 dtype but the one and the same dtype,  that we format in bools or floats")
	print("- 1 column strings (samples names) of dtype object")
	print("-------step 12 : joining of the features and response tables...")  # joined on the sample name columns # inner: use intersection of keys from both frames
	df_joined = pd.merge(df_sup_file, df_file, how="inner", left_on=Samples_col_name_left, right_on=Samples_col_name_right)
	print("-------step 13 : dropping the extra sample column present after the joining of left and right tables...")
	df_joined.drop(labels=[Samples_col_name_left], axis=1, inplace=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file  # clear memory
		del df_sup_file  # clear memory

	print("-------step 17 : A-putting the samples column at last position (no impact on the CICS preprocessing) and tagging the 3 groups of columns to manipulate them easier......")  # if decided by user,
	# strategy : move around the col names and use the new list to build new dataframe
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]:
		df_joined_initial_cols_list = list(df_joined.columns)
		resp_cols_list = df_joined_initial_cols_list[0:num_resp_cols_from_sup]  # num_resp_cols_from_sup = the first 4 cols if the 4 responses kept initially
		samples_cols_list = [df_joined_initial_cols_list[num_resp_cols_from_sup]]
		feat_cols_list = df_joined_initial_cols_list[(num_resp_cols_from_sup + 1):]  # we want to get the following column thats why we make + 1
		df_joined_cols_reordered_list = resp_cols_list + feat_cols_list + samples_cols_list
		# df_joined_cols_reordered_list = [df_joined_initial_cols_list[0]] + df_joined_initial_cols_list[2:] + [df_joined_initial_cols_list[1]] # garb
		df_joined = df_joined[df_joined_cols_reordered_list]
		# the 3 groups of columns
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		feat_cols = df_joined.columns[num_resp_cols_from_sup:-1]
		samples_cols = df_joined.columns[-1]
		state_of_samples_pos = "at_last"  # a tag to known in the following the state of the position of the samples col
	else:
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		samples_cols = df_joined.columns[num_resp_cols_from_sup]
		feat_cols = df_joined.columns[(num_resp_cols_from_sup + 1):]
		state_of_samples_pos = "not_at_last"

	#####-----checkpoint
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns) in order to know what dtypes to convert...")
	df_joined_preview = df_joined.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_joined_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----
	print("-------step 17 : B-formatting the dtypes of each group of columns : put the samples name in dtype string (object)...")  ##! help also to do before computing on fts vlues
	df_joined[samples_cols] = df_joined[samples_cols].astype(str)  # strings dtype is object so we have to find object

	print("-------step 17 : C1-formatting the dtypes of each group of columns : put the features values in float dtype...")
	# - replace the commas blocking the conversion of objects in floats
	df_joined_feat_cols_only_no_commas = df_joined[feat_cols].replace(",", ".", regex=True) # anciently used is df_joined[feat_cols] as result var
	# - a fast method used to change all fts values into floats
	# old_fts_col_names = df_joined[feat_cols].columns # not needed because old_fts_col_names is feat_cols
	df_fts_as_series = df_joined_feat_cols_only_no_commas.values.astype(np.float64)
	df_fts_back_as_df = pd.DataFrame(df_fts_as_series)
	df_fts_back_as_df.columns = feat_cols
	# df_joined[feat_cols] = df_fts_back_as_df
	# instead of putting the galerie of df_fts back in the df_joined, just take the 2 remaining groups of cols of df_joined and add them to df_fts to reform the old df_joined called now dframe
	# -putting back the resp cols
	for col_of_resp_to_put_back in resp_cols:
		# print(col_of_resp_to_put_back) # test
		original_index_of_resp_col_to_put_back = list(df_joined.columns).index(col_of_resp_to_put_back)
		df_fts_back_as_df.insert(original_index_of_resp_col_to_put_back, col_of_resp_to_put_back, df_joined[col_of_resp_to_put_back])  # reput the resp col at the front ## last_stop_to_retouch
	# -putting back sample col
	original_index_of_sample_col_to_put_back = list(df_joined.columns).index(samples_cols)
	df_fts_back_as_df.insert(original_index_of_sample_col_to_put_back, samples_cols, df_joined[samples_cols])
	# depending if the fts col are at the end or not, it is still the same with previous line # also len(df_fts_back_as_df.axes[0]) = what would be the index of a new col as last

	print("-------step 17 : C2-renaming the fts using a table of annotations...")
	# ---Renaming the fts
	# - read the file containg the table of annotations
	# NB : putting back the fts_cols selector as it should be is not needed because still the same columns in the fts group
	old_fts_cols_list = feat_cols.values.tolist()  # old way => old_fts_cols_list = feat_cols.values.tolist()
	colname_of_previous_states = "PSI"
	df_file4annot.sort_values(colname_of_previous_states, axis=0, ascending=True, inplace=True, kind='mergesort')  # sort the df following the values of the probesets
	colname_of_after_states = "GS"
	dict_previous_after_states = dict(zip(df_file4annot[colname_of_previous_states], df_file4annot[colname_of_after_states]))


	def fts_names_converter(old_list, dict2convertkeyinvalue):
		new_list = [item if not item in dict2convertkeyinvalue else dict2convertkeyinvalue[item] for item in old_list]
		return new_list


	new_fts_cols_list = fts_names_converter(old_fts_cols_list, dict_previous_after_states)
	dict2renamecols = dict(zip(old_fts_cols_list, new_fts_cols_list))
	df_fts_back_as_df.rename(columns=dict2renamecols, inplace=True)
	# lets put back the fts_cols selector as it should be in case we need it again
	feat_cols_list_b4_change = feat_cols
	if state_of_samples_pos == "at_last":  # decide where are the fts
		feat_cols = df_fts_back_as_df.columns[num_resp_cols_from_sup:-1]
	else:
		feat_cols = df_joined.columns[(num_resp_cols_from_sup + 1):]

	print("----step 17 : E-A report on if all the initial features names have been successfully found and changed...")
	# use this following to find common elements between old columns list and new columns list :
	# list1_as_set = set(list1) and set_intersection = list1_as_set.intersection(list2) and then intersection_as_list = list(intersection)
	list_of_unchanged_fts = list(set(list(feat_cols_list_b4_change)).intersection(list(feat_cols)))
	# list_of_unchanged_fts = [] # slow so deprecated
	# for ft_name in list(feat_cols):
	# 	if ft_name in list(feat_cols_list_b4_change):
	# 		list_of_unchanged_fts.append(ft_name)
	if len(list_of_unchanged_fts) != 0 :
		print(len(list_of_unchanged_fts),"initial features have not been found in the conversion dictionnary, hence their names not changed.")
	else :
		print("All initial features have been found in the conversion dictionnary and changed")

	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_fts_back_as_df_preview = df_fts_back_as_df.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_fts_back_as_df_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----

	# df ready for response computations
	df_aft_resp = df_fts_back_as_df

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_fts_back_as_df  # clear memory
		del df_fts_as_series  # clear memory
		del df_joined  # cleaning

	print("-------step 18 : D2-formatting the dtypes of each group of columns : put the response values in 2 dtype string (object) Res and Sen (Neg and Pos) to be able to read easier any contigency table...")
	list_of_resp_col_indexes_for_ResSen_changes = [0]
	for a_resp_col in resp_cols:
		if list(resp_cols).index(a_resp_col) in list_of_resp_col_indexes_for_ResSen_changes:
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:, [a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2: # start modifs
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res", "Sen"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			elif len(RespClasses_list_wo_nan) == 1 :
				if RespClasses_list_wo_nan[0] == 1 :
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Sen"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
				elif RespClasses_list_wo_nan[0] == 0 :
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			else:  # there is more than 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)   # end modifs
		else:  # the others responses that have Neg and Pos
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:, [a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2:
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Neg", "Pos"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(
					str)  # response can be string dtype as it will be encoded later
			else:  # there is not 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)

	# sort the dataframe entries following response col values and remake a new index (resp col 1 is used in multiple responses data)
	# sort the df following the values of the resp column
	df_aft_resp.sort_values(list(resp_cols)[0], axis=0, ascending=True, inplace=True, kind='mergesort')
	# after the precedent sort, the indexes are not in order. make a new order for them
	df_aft_resp = df_aft_resp.reset_index(drop="True")
	print("Describing the obtained final samples-features-response frame...")
	total_samples = len(df_aft_resp.axes[0])
	total_resps = len(list(resp_cols))
	total_feats = len(df_aft_resp.axes[1]) - (total_resps + 1)  # withdraw of the total 1 sample col and the response col
	print("The frame to analyse has", total_samples, "samples,", total_feats, "features and", total_resps, "response(s)")
	# report on the responses individually
	for resp_name in resp_cols:
		resp_name_full_col = df_aft_resp.loc[:, [resp_name]]  # get the sorted unique values in it
		classes_detected_incl_nan = sorted(resp_name_full_col.iloc[:, 0].unique())
		num_classes_detected_incl_nan = len(classes_detected_incl_nan)
		print("- Response", resp_name, "has", num_classes_detected_incl_nan, " classes : ")
		for class_detected in classes_detected_incl_nan:  # go over the present response
			count_of_class_detected = df_aft_resp[resp_name].value_counts()[class_detected]  # before it was using dframe.iloc[:, 0]
			count_perc_of_class_detected = (count_of_class_detected / total_samples) * 100
			print("- - a class value", class_detected, "is found on", count_of_class_detected, "samples counting for", '{:.3f}'.format(count_perc_of_class_detected), "% of the samples")

	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_aft_resp_preview = df_aft_resp.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_aft_resp_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----

	dframe = df_aft_resp  # the final dataframe to write in a .csv file

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_aft_resp  # clear mem

	print("-----step 20 : Saving a copy of the final dataframe in a .csv file...")
	tag_ctype = "BRCA"
	tag_drugname = "REMAGUS04_NAC"  # manually recordd in the treatments details files # NAC = NeoAdjuvant Chemotherapy
	tag_drugID = "Treatment12"
	# tag_resp_kept = resp_used
	tag_respType = str(cohort_used) + "x" + "NAC" + "x" + str(total_samples) + "S" + "x" + str(total_feats) + "F" + "x" + str(len(list(resp_cols))) + "Ras" + str(resp_used)
	tag_profilename = "GEX"
	# the output path has 3 parts : the root until the ouput folder, the output folder, and the filename
	# - lets make the file name
	output_filename_for_final_dframe = tag_ctype + "_" + tag_drugID + "_" + tag_respType + "_" + tag_profilename + ".csv"
	# - lets make the root until the ouput folder (the output folder excluded)
	if tag_decision_launching_way in ["cmd_line", "cl"]:  # launch the script in a terminal
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		root_until_output_folder = os.getcwd()  # obtained by changing directory
	else:  # launch the script line by line
		root_until_output_folder = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version"  # whetever the command center
	# - lets make the output folder
	output_folder_on_same_lvl_than_main_name = "outputs"
	# - lets extend the root path to contain the output folder
	root_until_output_folder_w_output_folder = os.path.join(root_until_output_folder, output_folder_on_same_lvl_than_main_name)
	if not os.path.exists(root_until_output_folder_w_output_folder):
		os.mkdir(root_until_output_folder_w_output_folder)
	# - lets make the full path to the file to save
	fullname = os.path.join(root_until_output_folder_w_output_folder, output_filename_for_final_dframe)
	# - lets use the full path to save the file
	dframe.to_csv(fullname, index=None, header=True)
	print("File saved !")
	print(cohort_used, "dataset formatting for CICS analysis is done!")
	print("the file location is :", fullname)
##! also delete all the uneccesary variables got sooner
##! also create a log of all operations

elif cohort_used == "MDAnderson" :
	print("Storing files contents in dataframes...")
	df_file = pd.read_csv(file_path, sep_in_file,skiprows=6)  ##! add skiprows=0 to skip lines 0 lines here, default is None
	df_sup_file = pd.read_excel(supporting_file_path, sheet_id,skiprows=2)
	df_file4annot = pd.read_csv(file4annot_path, sep_in_file4annot)

	print("PART I -----> put in form the content of supplementary table (LEFT)...")
	print("-------step 1 : rename the targeted columns on both left and right dataframes")
	# we will need to capture columns and move them around or edit them for restricting, joining the columns etc.
	# it is better to name the manipulated columns and their give also their future names
	# needed columns : sample col, response col
	old_Samples_col_name_left = "idtext"  # given samples col name by the user ##! to get from the argument
	Samples_col_name_left = "Model_bis"  # for the left table
	old_Samples_col_name_right = "row.names"
	# NB : the 2 samples columns of the 2 tables must have different names to join them later
	Samples_col_name_right = "Model"  # for the right table
	common_samples_id_prefix = "CLETRI"
	print("----step 2 : A-restricting the support info table to only the raw needed columns...There are : ")  # strategy : selected only the needed columns because they are not a lot
	list_of_only_needed_cols_in_sup_table = [old_Samples_col_name_left] + ["pCR", "erpos", "prpos", "herpos"]
	for col_name in list_of_only_needed_cols_in_sup_table:
		print("-", col_name)
	# df_sup_file = df_sup_file[[old_Samples_col_name_left,old_Resp_col_name_left]]  #garb
	df_sup_file = df_sup_file[list_of_only_needed_cols_in_sup_table]

	print("----step 2 : rectifying the name of some samples...)")
	df_sup_file.ix[127, 'idtext'] = "PERU16"
	df_sup_file.ix[129, 'idtext'] = "PERU14"
	print("-------step 2 : B-restriction of responses following chosen to be kept response")


	# lets define needed functions
	# - a function to get rid of undefined values in responses or samples
	def rows_w_nan_dropper(df_to_clean):
		print("-------step 4 : eliminating all samples unidentified or without response (with nan values)...")  # also a way to make sure the joining with the right table later goes smootly
		samples_b4_cleaning = len(df_to_clean.axes[0])
		df_to_clean.dropna(axis='index', inplace=True)
		samples_aft_cleaning = len(df_to_clean.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete samples info of the left table : ")  # a report on the losses while cleaning
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete samples info of the left table")
		else:
			print(lost_samples, "of", samples_b4_cleaning, "samples have been lost during the cleaning of the uncomplete samples info of the left table")
		return df_to_clean


	# - the fonction to get the triple neg bool values in the new resp col
	def tn_maker(row):
		if row["erpos"] == 0 and row["prpos"] == 0 and row["herpos"] == 0:
			return 1
		else:
			return 0


	print("-----> following a response choice, restricting the population to the individuals to use (LEFT)...")
	if resp_used == "RCHdefined":
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["pCR"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif (resp_used == "TNBCdefined") | (resp_used == "RCHandTNBCdefined"):
		old_Resp_col_name_left1 = "erpos"  # capture of the 3 cols of interest
		old_Resp_col_name_left2 = "prpos"
		old_Resp_col_name_left3 = "herpos"
		old_Resp_col_name_left = "TNBC"  # the name of the new resp col
		df_sup_file[old_Resp_col_name_left] = df_sup_file.apply(lambda row: tn_maker(row), axis=1)  # getting the new col at last pos
		# df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1]  # restrict the population to the samples of interest ##! not needed put at the end
		if resp_used == "TNBCdefined":
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["TNBC"]
		else:  # resp_used == "RCHandTNBCdefined"
			list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left] + ["pCR", "TNBC"]
		df_sup_file = df_sup_file[list_of_cols_to_keep_after_rsp_choice_sup_table]
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	elif resp_used == "RCH3HSdefined":
		# list_of_cols_to_keep_after_rsp_choice_sup_table = [old_Samples_col_name_left]+["RCH","RO","RP","HER2"] # already at this state # not needed but leave her for reference
		df_sup_file = rows_w_nan_dropper(df_sup_file)
	else:  # no response is preferred ie all 4 are kept wo deleting nan values
		print("From the left table, all responses with or without undefined values are kept.")
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : A1-renaming the 2 kept columns (samples, response used to restrict population) for the left table... ")
	dict_keptlefttablecolsnames_old2new = {old_Samples_col_name_left: Samples_col_name_left, "pCR": "BestResCat_as_RCH", "TNBC": "BestResCat_as_TNBC", "erpos": "BestResCat_as_RO", "prpos": "BestResCat_as_RP", "herpos": "BestResCat_as_HER2"}
	remaining_colsnames = df_sup_file.columns
	for a_remaining_colname in remaining_colsnames:
		df_sup_file.rename(columns={a_remaining_colname: dict_keptlefttablecolsnames_old2new[a_remaining_colname]}, inplace=True)
	num_resp_cols_from_sup = len(list(remaining_colsnames)) - 1  # keept this number for future indexing of responses columns group
	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 3 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_sup_file[Samples_col_name_left] = common_samples_id_prefix + "_" + df_sup_file[Samples_col_name_left].astype(str)  # sample_name is a string ColumnName_IdInColumn #(340,5)
	print("-------step 3 : C-facilitating the merge to come: sorting the left dataframe based on samples entries and dropping the rows that already exist")
	df_sup_file = data_mgmt_6(df_sup_file, Samples_col_name_left)  # sorting
	df_sup_file_w_marked_as_duplicates_rows_only = df_sup_file[df_sup_file.duplicated()]  # dropping duplicates rows
	df_sup_file_indexes_to_drop = df_sup_file_w_marked_as_duplicates_rows_only.index
	df_sup_file = df_sup_file.drop(df_sup_file_indexes_to_drop)
	df_sup_file = df_sup_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_sup_file_w_marked_as_duplicates_rows_only
		del df_sup_file_indexes_to_drop

	print("PART II -----> put in form the content of features table (RIGHT)")
	# print("-------step 5 : the columns name are shifted one column towards the front so we shift them back by one clumn...")
	# old_cols_list = list(df_file.columns)
	# new_cols_list = [old_Samples_col_name_right] + old_cols_list[:-1]
	# df_file.columns = new_cols_list
	print("-------step 5 : changing columns into rows because the present rows are not the samples...")
	df_file = df_file.transpose()
	print("-------step 6 : making the index as a column (the index was the sample names)...")  # by resetting the index in a way to get the older index as a column)
	df_file = df_file.reset_index()
	print("-------step 7 : making a copy of the first line and use it as titles of the columns...")
	df_file.columns = df_file.iloc[0]
	print("-------step 8 : dropping the first line because it is now the titles of the columns...")
	df_file = df_file.drop(df_file.index[0])
	print("-------step 9 : resetting the index for a correct order (previous dropping messed it up)...")  # the index is missing now a the 1st line entry value (0). reset it in a way to not get a new column
	df_file = df_file.reset_index(drop=True)
	# print("-------step 10 : restricting the right table to only the needed columns (dropping unecessary columns)...")  # not necessary columns are dropped because they are not a lot
	# list_of_unecessary_cols_2_drop = ["CLETRI"]  # a column that is just a repetiton of the sample names col
	# df_file.drop(labels=list_of_unecessary_cols_2_drop, axis=1, inplace=True)
	print("-------step 11 : A-renaming the sample column... ")
	df_file.rename(columns={old_Samples_col_name_right: Samples_col_name_right}, inplace=True)
	print("-------step 11 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_file[Samples_col_name_right] = common_samples_id_prefix + "_" + df_file[Samples_col_name_right].astype(str)  # sample_name is a string ColumnName_IdInColumn
	# >>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : C-removing cells with nan...")
	# as response is clear and samples should be also, any nan left in the joinded table is in the features values
	if tag_decision_nan_del_samples_or_fts in ["samples", "S"]:  # - choose between dropping rows and dropping columns # (prefer to drop rows)
		samples_b4_cleaning = len(df_file.axes[0])
		df_file.dropna(axis='index', inplace=True)  # losing samples
		samples_aft_cleaning = len(df_file.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning # lost_samples,"of",samples_b4_cleaning,"samples have been lost
		if lost_samples == 0:
			print("No samples has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_samples, "of", samples_b4_cleaning, "samples have been lost during the cleaning of the uncomplete features info of the right table")
	elif tag_decision_nan_del_samples_or_fts in ["features", "F"]:  # - choose between dropping rows and dropping columns # (prefer to drop cols)
		features_b4_cleaning = len(df_file.axes[1])
		df_file.dropna(axis='columns', inplace=True)  # losing fts
		features_aft_cleaning = len(df_file.axes[1])
		lost_features = features_b4_cleaning - features_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the right table : ")  # a report on the losses while cleaning
		if lost_features == 0:
			print("No features has been lost during the cleaning of the uncomplete features info of the right table")
		else:
			print(lost_features, "of", features_b4_cleaning, "samples have been lost during the cleaning of the uncomplete features info of the right table")
	else:  # estimate the % of nan values in the row and the column, then delete the axis with more % and if below a certain %, impute it if user allow it
		print("For the cleaning of the uncomplete features info of the joined table based on missing values percentage has still to be implemented")
	# >>>>>>>>>>>>>>>>>>>>>>>>>
	print("-------step 11 : D-facilitating the merge to come: sorting the right dataframe based on samples entries and dropping the rows that already exist")
	df_file = data_mgmt_6(df_file, Samples_col_name_right)  # sorting
	df_file_w_marked_as_duplicates_rows_only = df_file[df_file.duplicated()]  # dropping duplicates rows
	df_file_indexes_to_drop = df_file_w_marked_as_duplicates_rows_only.index
	df_file = df_file.drop(df_file_indexes_to_drop)
	df_file = df_file.reset_index(drop=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file_w_marked_as_duplicates_rows_only
		del df_file_indexes_to_drop

	print("PART III -----> formatting every group of columns in the final frame for the content of the .csv response-features-samples files")
	print(">>>>>-the awaited configuration is :")
	print("- 1 group of columns (the responses) that can have anything as dtype, (int64/float64/object) and that is why we encode it in string (dtype object)")
	print("- 1 group of columns that is int64/float64 dtype but the one and the same dtype,  that we format in bools or floats")
	print("- 1 column strings (samples names) of dtype object")
	print("-------step 12 : joining of the features and response tables...")  # joined on the sample name columns # inner: use intersection of keys from both frames
	df_joined = pd.merge(df_sup_file, df_file, how="inner", left_on=Samples_col_name_left, right_on=Samples_col_name_right)
	print("-------step 13 : dropping the extra sample column present after the joining of left and right tables...")
	df_joined.drop(labels=[Samples_col_name_left], axis=1, inplace=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file  # clear memory
		del df_sup_file  # clear memory

	print("-------step 17 : A-putting the samples column at last position (no impact on the CICS preprocessing) and tagging the 3 groups of columns to manipulate them easier......")  # if decided by user,
	# strategy : move around the col names and use the new list to build new dataframe
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]:
		df_joined_initial_cols_list = list(df_joined.columns)
		resp_cols_list = df_joined_initial_cols_list[0:num_resp_cols_from_sup]  # num_resp_cols_from_sup = the first 4 cols if the 4 responses kept initially
		samples_cols_list = [df_joined_initial_cols_list[num_resp_cols_from_sup]]
		feat_cols_list = df_joined_initial_cols_list[(num_resp_cols_from_sup + 1):]  # we want to get the following column thats why we make + 1
		df_joined_cols_reordered_list = resp_cols_list + feat_cols_list + samples_cols_list
		# df_joined_cols_reordered_list = [df_joined_initial_cols_list[0]] + df_joined_initial_cols_list[2:] + [df_joined_initial_cols_list[1]] # garb
		df_joined = df_joined[df_joined_cols_reordered_list]
		# the 3 groups of columns
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		feat_cols = df_joined.columns[num_resp_cols_from_sup:-1]
		samples_cols = df_joined.columns[-1]
		state_of_samples_pos = "at_last"  # a tag to known in the following the state of the position of the samples col
	else:
		resp_cols = df_joined.columns[0:num_resp_cols_from_sup]
		samples_cols = df_joined.columns[num_resp_cols_from_sup]
		feat_cols = df_joined.columns[(num_resp_cols_from_sup + 1):]
		state_of_samples_pos = "not_at_last"

	#####-----checkpoint
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns) in order to know what dtypes to convert...")
	df_joined_preview = df_joined.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_joined_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----
	print("-------step 17 : B-formatting the dtypes of each group of columns : put the samples name in dtype string (object)...")  ##! help also to do before computing on fts vlues
	df_joined[samples_cols] = df_joined[samples_cols].astype(str)  # strings dtype is object so we have to find object

	print("-------step 17 : C1-formatting the dtypes of each group of columns : put the features values in float dtype...")
	# - replace the commas blocking the conversion of objects in floats
	df_joined_feat_cols_only_no_commas = df_joined[feat_cols].replace(",", ".", regex=True)  # anciently used is df_joined[feat_cols] as result var
	# - a fast method used to change all fts values into floats
	# old_fts_col_names = df_joined[feat_cols].columns # not needed because old_fts_col_names is feat_cols
	df_fts_as_series = df_joined_feat_cols_only_no_commas.values.astype(np.float64)
	df_fts_back_as_df = pd.DataFrame(df_fts_as_series)
	df_fts_back_as_df.columns = feat_cols
	# df_joined[feat_cols] = df_fts_back_as_df
	# instead of putting the galerie of df_fts back in the df_joined, just take the 2 remaining groups of cols of df_joined and add them to df_fts to reform the old df_joined called now dframe
	# -putting back the resp cols
	for col_of_resp_to_put_back in resp_cols:
		# print(col_of_resp_to_put_back) # test
		original_index_of_resp_col_to_put_back = list(df_joined.columns).index(col_of_resp_to_put_back)
		df_fts_back_as_df.insert(original_index_of_resp_col_to_put_back, col_of_resp_to_put_back, df_joined[col_of_resp_to_put_back])  # reput the resp col at the front ## last_stop_to_retouch
	# -putting back sample col
	original_index_of_sample_col_to_put_back = list(df_joined.columns).index(samples_cols)
	df_fts_back_as_df.insert(original_index_of_sample_col_to_put_back, samples_cols, df_joined[samples_cols])
	# depending if the fts col are at the end or not, it is still the same with previous line # also len(df_fts_back_as_df.axes[0]) = what would be the index of a new col as last

	print("-------step 17 : C2-renaming the fts using a table of annotations...")
	# ---Renaming the fts
	# - read the file containg the table of annotations
	# NB : putting back the fts_cols selector as it should be is not needed because still the same columns in the fts group
	old_fts_cols_list = feat_cols.values.tolist()  # old way => old_fts_cols_list = feat_cols.values.tolist()
	colname_of_previous_states = "PSI"
	df_file4annot.sort_values(colname_of_previous_states, axis=0, ascending=True, inplace=True, kind='mergesort')  # sort the df following the values of the probesets
	colname_of_after_states = "GS"
	dict_previous_after_states = dict(zip(df_file4annot[colname_of_previous_states], df_file4annot[colname_of_after_states]))


	def fts_names_converter(old_list, dict2convertkeyinvalue):
		new_list = [item if not item in dict2convertkeyinvalue else dict2convertkeyinvalue[item] for item in old_list]
		return new_list


	new_fts_cols_list = fts_names_converter(old_fts_cols_list, dict_previous_after_states)
	dict2renamecols = dict(zip(old_fts_cols_list, new_fts_cols_list))
	df_fts_back_as_df.rename(columns=dict2renamecols, inplace=True)
	# lets put back the fts_cols selector as it should be in case we need it again
	feat_cols_list_b4_change = feat_cols
	if state_of_samples_pos == "at_last":  # decide where are the fts
		feat_cols = df_fts_back_as_df.columns[num_resp_cols_from_sup:-1]
	else:
		feat_cols = df_joined.columns[(num_resp_cols_from_sup + 1):]

	print("----step 17 : E-A report on if all the initial features names have been successfully found and changed...")
	# use this following to find common elements between old columns list and new columns list :
	# list1_as_set = set(list1) and set_intersection = list1_as_set.intersection(list2) and then intersection_as_list = list(intersection)
	list_of_unchanged_fts = list(set(list(feat_cols_list_b4_change)).intersection(list(feat_cols)))
	# list_of_unchanged_fts = [] # slow so deprecated
	# for ft_name in list(feat_cols):
	# 	if ft_name in list(feat_cols_list_b4_change):
	# 		list_of_unchanged_fts.append(ft_name)
	if len(list_of_unchanged_fts) != 0:
		print(len(list_of_unchanged_fts), "initial features have not been found in the conversion dictionnary, hence their names not changed.")
	else:
		print("All initial features have been found in the conversion dictionnary and changed")

	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_fts_back_as_df_preview = df_fts_back_as_df.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_fts_back_as_df_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----

	# df ready for response computations
	df_aft_resp = df_fts_back_as_df

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_fts_back_as_df  # clear memory
		del df_fts_as_series  # clear memory
		del df_joined  # cleaning

	print("-------step 18 : D2-formatting the dtypes of each group of columns : put the response values in 2 dtype string (object) Res and Sen (Neg and Pos) to be able to read easier any contigency table...")
	list_of_resp_col_indexes_for_ResSen_changes = [0]
	for a_resp_col in resp_cols:
		if list(resp_cols).index(a_resp_col) in list_of_resp_col_indexes_for_ResSen_changes:
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:, [a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2:  # start modifs
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res", "Sen"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			elif len(RespClasses_list_wo_nan) == 1:
				if RespClasses_list_wo_nan[0] == 1:
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Sen"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
				elif RespClasses_list_wo_nan[0] == 0:
					df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Res"], inplace=True)
					df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			else:  # there is more than 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # end modifs
		else:  # the others responses that have Neg and Pos
			# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
			# get the response column in order to get the sorted unique values in it
			RespBin = df_aft_resp.loc[:, [a_resp_col]]
			# get the sorted unique values in it
			RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
			RespClasses_list_wo_nan = [x for x in RespClasses_list if str(x) != 'nan']
			if len(RespClasses_list_wo_nan) == 2:
				df_aft_resp[a_resp_col].replace(RespClasses_list_wo_nan, ["Neg", "Pos"], inplace=True)
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)  # response can be string dtype as it will be encoded later
			else:  # there is not 2 classes...response can be string dtype as it will be encoded later
				df_aft_resp[a_resp_col] = df_aft_resp[a_resp_col].astype(str)

	# sort the dataframe entries following response col values and remake a new index (resp col 1 is used in multiple responses data)
	# sort the df following the values of the resp column
	df_aft_resp.sort_values(list(resp_cols)[0], axis=0, ascending=True, inplace=True, kind='mergesort')
	# after the precedent sort, the indexes are not in order. make a new order for them
	df_aft_resp = df_aft_resp.reset_index(drop="True")
	print("Describing the obtained final samples-features-response frame...")
	total_samples = len(df_aft_resp.axes[0])
	total_resps = len(list(resp_cols))
	total_feats = len(df_aft_resp.axes[1]) - (total_resps + 1)  # withdraw of the total 1 sample col and the response col
	print("The frame to analyse has", total_samples, "samples,", total_feats, "features and", total_resps, "response(s)")
	# report on the responses individually
	for resp_name in resp_cols:
		resp_name_full_col = df_aft_resp.loc[:, [resp_name]]  # get the sorted unique values in it
		classes_detected_incl_nan = sorted(resp_name_full_col.iloc[:, 0].unique())
		num_classes_detected_incl_nan = len(classes_detected_incl_nan)
		print("- Response", resp_name, "has", num_classes_detected_incl_nan, " classes : ")
		for class_detected in classes_detected_incl_nan:  # go over the present response
			count_of_class_detected = df_aft_resp[resp_name].value_counts()[class_detected]  # before it was using dframe.iloc[:, 0]
			count_perc_of_class_detected = (count_of_class_detected / total_samples) * 100
			print("- - a class value", class_detected, "is found on", count_of_class_detected, "samples counting for", '{:.3f}'.format(count_perc_of_class_detected), "% of the samples")

	####-----
	print("----A checkpoint to check on joined table dtypes (first 5 columns and last 5 columns)...")
	df_aft_resp_preview = df_aft_resp.iloc[:, list(range(6)) + [-5, -4, -3, -2, -1]]
	print(df_aft_resp_preview.info())  # on the model of df_joined[df_joined.columns[:10]].dtypes
	####-----

	dframe = df_aft_resp  # the final dataframe to write in a .csv file

	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_aft_resp  # clear mem

	print("-----step 20 : Saving a copy of the final dataframe in a .csv file...")
	tag_ctype = "BRCA"
	tag_drugname = "MDAnderson_NAC"  # manually recordd in the treatments details files # NAC = NeoAdjuvant Chemotherapy
	tag_drugID = "Treatment13"
	# tag_resp_kept = resp_used
	tag_respType = str(cohort_used) + "x" + "NAC" + "x" + str(total_samples) + "S" + "x" + str(total_feats) + "F" + "x" + str(len(list(resp_cols))) + "Ras" + str(resp_used)
	tag_profilename = "GEX"
	# the output path has 3 parts : the root until the ouput folder, the output folder, and the filename
	# - lets make the file name
	output_filename_for_final_dframe = tag_ctype + "_" + tag_drugID + "_" + tag_respType + "_" + tag_profilename + ".csv"
	# - lets make the root until the ouput folder (the output folder excluded)
	if tag_decision_launching_way in ["cmd_line", "cl"]:  # launch the script in a terminal
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		root_until_output_folder = os.getcwd()  # obtained by changing directory
	else:  # launch the script line by line
		root_until_output_folder = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version"  # whetever the command center
	# - lets make the output folder
	output_folder_on_same_lvl_than_main_name = "outputs"
	# - lets extend the root path to contain the output folder
	root_until_output_folder_w_output_folder = os.path.join(root_until_output_folder, output_folder_on_same_lvl_than_main_name)
	if not os.path.exists(root_until_output_folder_w_output_folder):
		os.mkdir(root_until_output_folder_w_output_folder)
	# - lets make the full path to the file to save
	fullname = os.path.join(root_until_output_folder_w_output_folder, output_filename_for_final_dframe)
	# - lets use the full path to save the file
	dframe.to_csv(fullname, index=None, header=True)
	print("File saved !")
	print(cohort_used, "dataset formatting for CICS analysis is done!")
	print("the file location is :", fullname)
##! also delete all the uneccesary variables got sooner
##! also create a log of all operations
else :
	print("no dataset known added for treatment")


# #<~~~~~~~~~~~~~ stop redirection of stdout to a .o file step 2/2 ~~~~~~~~~~~
# if tag_decision_make_log in ["yes", "y"]:
# 	sys.stdout = original_out

#<<<<<<<----end of all that is data file specific about the preprocessing

###===============================================================================================================###
###===============================================================================================================###
###===============================================================================================================###

	###===============================================================================================================###
	# ==========> put this at the end (not necessary already done)


