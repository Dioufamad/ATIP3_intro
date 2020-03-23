#>>>>>>>>>>>>>>>>>>>>>>>>>>> REMAGUS02 DATASET FORMATTING FOR CICS SCRIPT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>> RE-ENCODING
# the file we are given is a table with the intent to represent values, each one corresponding to a variable and a sample

# 1st issue : the file might not be in a supported encoding so we have to reencode it in UTF-8
# get the file encoding using the following
# file -i REMAGUS02-Données\ genomique_226x54676\ totales.txt
# change the file encoding and save the new version in .tsv format (use the previous encoding in all capitals letters)
# iconv -f UTF-16LE -t UTF-8//IGNORE REMAGUS02-Données\ genomique_226x54676\ totales.txt > output2.tsv

#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import os
import pandas as pd # for dataframes manipulation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # to change the Response values from string to classes 0 and 1 # not needed at the moment
import numpy as np # linear algebra and exploit arrays faster and easier computations
import os #for bash command lines in python
import locale
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')
# import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline
from pathlib import Path # to manage paths as into arguments
#>>>>>>>>>>>>>>>>>>>>>>>>>>> Variables to initialise------------------------------------------
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
cohort_used = "REMAGUS02"
# cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
#---for the files to manipulate
# stock the file and its separator # whatever command_center is chosen
if cohort_used == "REMAGUS02":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS02_Donnees_genomiques_226x54676_totales.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS02-Données cliniques.xls" # @ home
	sheet_id = "extractionCNahmias"
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annot/REMAGUS02_PSI_GS_54675.csv"
	sep_in_file4annot = ","
elif cohort_used == "REMAGUS04":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS04normData_R04_142_CNahmias.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS04-Données cliniques.xls" # @ home
	sheet_id = "extractionCNahmias" ##! to check for change
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annot/REMAGUS04_PSI_GS_22215.csv"
	sep_in_file4annot = ","
elif cohort_used == "MDAnderson":
	file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/MDAnderson_MDA133_Expression.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/MDAnderson-MDA133CompleteInfo20060418.xls" # @ home
	sheet_id = "extractionCNahmias" ##! to check for change
	file4annot_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/3c_data_trial1/annot/REMAGUS04_PSI_GS_22215.csv" # uses the same as REMAGUS04
	sep_in_file4annot = ","
else: # when no cohort is chosen
	print("No cohort has been chosen for ppreprocessing")
print("paths to files manipulated and related specificities stored...")
# ----for the response strategy
resp_strategy = "SCcR" # Severe Cases centered Response
# resp_strategy = "TcR" # Treatment centered Response
dict_resp_str_fullnames = {"SCcR":"Severe Cases centered Response","TcR":"Treatment centered Response"}
resp_str_chosen_fullname = dict_resp_str_fullnames[resp_strategy]
print("response strategy is",resp_str_chosen_fullname,"(",resp_strategy,")...")
# -----end of imports and settings
print("Necessary libraries imported.")
print("Environnement variables initialised.")
print("All imports and settings are successfully placed")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> README <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Welcome in the formatting tool for the Remagus02 dataset in direction to CICS")
print("We suppose you have done the querying of a database and you have separated values files (csv,tsv,xls, etc.).")
print("Such values tables describe samples over multiples features, rows are samples and features are columns or vice-versa.")
print("We will try to format it into this representation : ")
print("- a .csv file that have succesively 3 groups of columns as features with features names as the titles of the columns")
print("+ 1 column as the classes and is the response used to extract rules by learning methods ")
print("+ multiples columns, each one as a feature titled the feature name")
print("+ 1 column as the samples and titled Model")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESSING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# --------------proper processing
print("Starting preprocessing...")
# Objective : joining both tables (the one with the features and the one with the response) into one unique table Response-Features-Samples
#>>>>>-----start of everything that is data file specific in the preprocessing
if cohort_used == "REMAGUS02" :
	print("Storing files contents in dataframes...")
	df_file = pd.read_csv(file_path,sep_in_file) ##! add skiprows=0 to skip lines 0 lines here, default is None
	df_sup_file = pd.read_excel(supporting_file_path,sheet_id)
	df_file4annot = pd.read_csv(file4annot_path,sep_in_file4annot)
	print("PART I -----> put in form the content of supplementary table (LEFT)...")
	print("-------step 1 : rename the targeted columns on both columns")
	# we will need to capture columns and move them around or edit them for restricting, joining the columns etc.
	# it is better to name the manipulated columns and their give also their future names
	# needed columns : sample col, response col
	old_Samples_col_name_left = "CLETRI" # given samples col name by the user ##! to get from the argument
	Samples_col_name_left = "Model_bis" # for the left table
	old_Samples_col_name_right = "cletri"
	# NB : the 2 samples columns of the 2 tables must have different names to join them later
	Samples_col_name_right = "Model" # for the right table
	common_samples_id_prefix = "CLETRI"
	print("-----> following a response strategy, restricting the population to the individuals to use (LEFT)...")
	if resp_strategy == "SCcR":
		old_Resp_col_name_left1 = "HER2" # capture of the 3 cols of interest
		old_Resp_col_name_left2 = "RO"
		old_Resp_col_name_left3 = "RP"
		def tn_maker(row): # the fonction to get the triple neg bool values in the new resp col
			if row["HER2"]==0 and row["RO"]==0 and row["RP"]==0 :
				return 1
			else:
				return 0
		old_Resp_col_name_left = "SCcR1" # the name of the new resp col
		df_sup_file[old_Resp_col_name_left] = df_sup_file.apply(lambda row: tn_maker(row), axis=1) # egtting the new col at last pos
		df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1] # restrict the population to the samples of interest
		Resp_col_name_left = "BestResCatSCcRleft"
	elif resp_strategy == "TcR":
		old_Resp_col_name_left = "RCH" # given resp col name by the user ##! to get from the argument # capture of the 1 col of interest
		df_sup_file = df_sup_file.loc[df_sup_file[old_Resp_col_name_left] == 1] # restrict the population to the samples of interest
		Resp_col_name_left = "BestResCatTcRleft"
	else: # no response strategy chosen
		old_Resp_col_name_left = "UnknownResp"
		Resp_col_name_left = "UnknownResp"
		print("No response strategy chosen. Please choose one.")
	##!  make her a final condittion for a simple one col extration as response (eg the RCH for all what is going with it in the features
	print("----step 2 : restricting the support info table to only the needed columns...") # strategy : selected only the needed columns because they are not a lot
	df_sup_file = df_sup_file[[old_Samples_col_name_left,old_Resp_col_name_left]]
	print("-------step 3 : A-renaming the 2 kept columns (samples, response used to restrict population)for the left table... ")
	df_sup_file.rename(columns={old_Samples_col_name_left: Samples_col_name_left, old_Resp_col_name_left : Resp_col_name_left}, inplace=True)
	print("-------step 3 : B-renaming the samples for traceability of sample id nature (said nature as a prefix in capital letters...")
	df_sup_file[Samples_col_name_left] = common_samples_id_prefix + "_" + df_sup_file[Samples_col_name_left].astype(str) # sample_name is a string ColumnName_IdInColumn
	print("-------step 4 : eliminating all samples unidentified or without response (with nan values)...") # also a way to make sure the joining with the right table later goes smootly
	samples_b4_cleaning = len(df_sup_file.axes[0])
	df_sup_file.dropna(axis='index', inplace=True)
	samples_aft_cleaning = len(df_sup_file.axes[0])
	lost_samples = samples_b4_cleaning - samples_aft_cleaning
	print("Report on the losses during the cleaning of the uncomplete samples info of the left table : ") # a report on the losses while cleaning
	if lost_samples==0:
		print("No samples has been lost during the cleaning of the uncomplete samples info of the left table")
	else:
		print(lost_samples,"samples has been lost during the cleaning of the uncomplete samples info of the left table")

	print("PART II -----> put in form the content of features table (RIGHT)")
	print("-------step 5 : changing columns into rows because the present rows are not the samples...")
	df_file = df_file.transpose()
	print("-------step 6 : making the index as a column (the index was the sample names)...")  # by resetting the index in a way to get the older index as a column)
	df_file = df_file.reset_index()
	print("-------step 7 : making a copy of the first line as titles of the columns...")
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

	# >>>>>-----formatting every group of columns in the final frame for the content of the .csv response-features-samples files
	# the awaited configuration is :
	# - 1 column that can have anything as dtype (the response) and that is why we encode it
	# - and a bunch that is int64/float64/object but the one and the same type that we previously formatted in bools or floats
	# - 1 column of dtype object (samples names)
	print("-------step 12 : joining of the features and response tables...") # joined on the sample name columns
	df_joined = pd.merge(df_sup_file, df_file, how="inner", left_on=Samples_col_name_left, right_on=Samples_col_name_right)
	print("-------step 13 : dropping the extra sample column present after the joining of left and right tables...")
	df_joined.drop(labels=[Samples_col_name_left], axis=1, inplace=True)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del df_file # clear memory
		del df_sup_file # clear memory
	print("-------step 14 : moving the response col from left tablle to 1st position of joined table...") # store the resp col that is last, drop it from the df and then insert it again at the 1st position of the df
	Resp_col_to_move = df_joined[Resp_col_name_left]
	df_joined.drop(labels=[Resp_col_name_left], axis=1, inplace=True)
	df_joined.insert(0, Resp_col_name_left, Resp_col_to_move)
	if clear_mem in ["yes", "y"]:
		print("**clearing memory...")
		del Resp_col_to_move # clear memory
	print("-------step 15 : putting the samples column at last position (not impact on the CICS preprocessing)...") # if decided by user,
	# strategy : move around the col names and use the new list to build new dataframe
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]:
		df_joined_initial_cols_list = list(df_joined.columns)
		df_joined_cols_reordered_list = [df_joined_initial_cols_list[0]] + df_joined_initial_cols_list[2:] + [df_joined_initial_cols_list[1]]
		df_joined = df_joined[df_joined_cols_reordered_list]
	print("-------step 16 : dropping all cells with nan...")
	# as response is clear and samples should be also, any nan left in the joinded table is in the features values
	# choose between dropping rows and dropping columns
	if tag_decision_nan_del_samples_or_fts in ["samples", "S"]:
		samples_b4_cleaning = len(df_joined.axes[0])
		df_joined.dropna(axis='index',inplace = True) # losing samples
		samples_aft_cleaning = len(df_joined.axes[0])
		lost_samples = samples_b4_cleaning - samples_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the joined table : ") # a report on the losses while cleaning
		if lost_samples==0:
			print("No samples has been lost during the cleaning of the uncomplete features info of the joined table")
		else:
			print(lost_samples,"samples has been lost during the cleaning of the uncomplete features info of the joined table")
	elif tag_decision_nan_del_samples_or_fts in ["features", "F"]:
		features_b4_cleaning = len(df_joined.axes[1])
		df_joined.dropna(axis='columns',inplace = True) # losing fts
		features_aft_cleaning = len(df_joined.axes[1])
		lost_features = features_b4_cleaning - features_aft_cleaning
		print("Report on the losses during the cleaning of the uncomplete features info of the joined table : ") # a report on the losses while cleaning
		if lost_features==0:
			print("No features has been lost during the cleaning of the uncomplete features info of the joined table")
		else:
			print(lost_features,"features has been lost during the cleaning of the uncomplete features info of the joined table")
	else : # estimate the % of nan values in the row and the column, then delete the axis with more % and if below a certain %, impute it if user allow it
		print("For the cleaning of the uncomplete features info of the joined table based on missing values percentage has still to be implemented")
	####-----A checkpoint to check on data

	# use this to get a peak at the dtypes in the final dataframe first 10 columns
	# df.iloc[:, list(range(6)) + [-5,-4,-3,-2,-1]]
	# df_joined[df_joined.columns[:10]].dtypes
	####-----

	# -------step 17 : giving names to the each of the 3 groups of columns to manipulate them in group using a name
	# sampl_col = df_joined.columns[0] is already Samples_col_name_right
	# resp_col = df_joined.columns[1] is already Resp_col_name_left
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]: # decide where are the fts
		feat_cols = df_joined.columns[1:-1]
	else:
		feat_cols = df_joined.columns[2:]
	# -------step 18 : formatting the dtypes of each group of columns (help also to do before computing on fts vlues)
	# 1- put the samples name in dtype string (object)
	df_joined[Samples_col_name_right] = df_joined[Samples_col_name_right].astype(str) # strings dtype is object so we have to find object


	# 2- put the response values in 2 strings Res and Sen to be able to read easier any contigency,
	# that is if there are only 2 classes detected. Else, change them into string and leave them like that to be encoded later
	# get the response column in order to get the sorted unique values in it
	RespBin = df_joined.loc[:,[Resp_col_name_left]]
	# get the sorted unique values in it
	RespClasses_list = sorted(RespBin.iloc[:, 0].unique())
	if len(RespClasses_list) == 2 :
		df_joined[Resp_col_name_left].replace(RespClasses_list,["Res","Sen"], inplace=True)
		df_joined[Resp_col_name_left] = df_joined[Resp_col_name_left].astype(str) # response can be string dtype as it will be encoded later
	else : # there is not 2 classes...response can be string dtype as it will be encoded later
		df_joined[Resp_col_name_left] = df_joined[Resp_col_name_left].astype(str)


	# 3- put the features values in float dtype ##! to do before the response)
	# - replace the commas blocking the conversion of objects in floats
	df_joined[feat_cols] = df_joined[feat_cols].replace(",", ".", regex=True)
	# - a fast method used to change all fts values into floats
	old_fts_col_names = df_joined[feat_cols].columns
	df_fts_as_series = df_joined[feat_cols].values.astype(np.float64)
	df_fts_back_as_df = pd.DataFrame(df_fts_as_series)
	df_fts_back_as_df.columns = old_fts_col_names
	# instead of putting the galerie of df_fts back in the df_joined, just take the 2 remaining cols of df_joined and add them to df_fts to reform the old df_joined called now dframe
	df_fts_back_as_df.insert(0, Resp_col_name_left, df_joined[Resp_col_name_left]) # reput the resp col at the front
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]: # depending if the fts col are at the end or not, reput them at the end or not
		df_fts_back_as_df.insert(len(df_fts_back_as_df.axes[1]), Samples_col_name_right, df_joined[Samples_col_name_right]) # len(df_fts_back_as_df.axes[0]) = what would be the index of a new col as last
	else:
		df_fts_back_as_df.insert(1, Samples_col_name_right, df_joined[Samples_col_name_right])

	#er##join these two parts in this order

	#---Renaming the fts
	# - read the file containg the table of annotations
	old_fts_cols_list = feat_cols.values.tolist()
	# old_fts_cols_list.sort()
	colname_of_previous_states = "PSI"
	df_file4annot.sort_values(colname_of_previous_states, axis=0, ascending=True, inplace=True, kind='mergesort') # sort the df following the values of the probesets
	# list_of_previous_states = df_file4annot[colname_of_previous_states].tolist()
	colname_of_after_states = "GS"
	# list_of_after_states = df_file4annot[colname_of_after_states].tolist()
	dict_previous_after_states = dict(zip(df_file4annot[colname_of_previous_states], df_file4annot[colname_of_after_states]))
	def fts_names_converter(old_list, dict2convertkeyinvalue):
		new_list = [item if not item in dict2convertkeyinvalue else dict2convertkeyinvalue[item] for item in old_list]
		return new_list
	new_fts_cols_list = fts_names_converter(old_fts_cols_list,dict_previous_after_states)
	dict2renamecols = dict(zip(old_fts_cols_list, new_fts_cols_list))
	df_fts_back_as_df.rename(columns=dict2renamecols, inplace=True)
	# lets put back the fts_cols selector as it should be
	if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]: # decide where are the fts
		feat_cols = df_fts_back_as_df.columns[1:-1]
	else:
		feat_cols = df_fts_back_as_df.columns[2:]







	###response after this

	dframe = df_fts_back_as_df
	del df_fts_back_as_df # cleaning
	del df_fts_as_series # cleaning
	del df_joined # cleaning

	####-----A checkpoint to check on data
	# dframe[dframe.columns[:10]].dtypes
	# dframe.info() # for an overall summary of remaining dtypes
	####-----

	# -------step 19 : formatting the response column, ordering it by class, displaying a report on the classes
	RespBin = dframe.loc[:,[Resp_col_name_left]]  # get the 1st column of data ... # anciently it was dframe[Resp_col_name] but gives a series instead of a df
	RespClasses = sorted(RespBin.iloc[:, 0].unique())
	binary_classes_le = LabelEncoder()  # the encoder
	binary_classes_le.fit(RespClasses)  # encode the classes to memorize
	encoded_classes = binary_classes_le.classes_ ##! change it into a list to access it directly (list of cols of array, same as getting the cols of a df)
	del RespBin # clear mem
	# del RespClasses # clear mem
	# sort the dataframe entries following response col values and remake a new index
	# sort the df following the values of the resp column
	dframe.sort_values(Resp_col_name_left, axis=0, ascending=True, inplace=True, kind='mergesort')
	# after the precedent sort, the indexes are not in order. make a new order for them
	dframe = dframe.reset_index(drop="True")
	print("Describing the obtained final samples-features-response frame...")
	total_samples = len(dframe.axes[0])
	total_feats = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
	print("The frame to analyse has ", total_samples,"samples and ",total_feats ,"features")
	print("Among",total_samples,"samples,",len(encoded_classes),"classes has been detected as being : {}.".format(' and '.join(str(class_value) for class_value in encoded_classes)))
	for class_value in encoded_classes:
		class_size = dframe.iloc[:, list(dframe).index(Resp_col_name_left)].value_counts()[encoded_classes[list(encoded_classes).index(class_value)]] # before it was using dframe.iloc[:, 0]
		class_size_perc = (class_size / total_samples)*100
		print("The class value",class_value,"is found on",class_size,"samples counting for",'{:.3f}'.format(class_size_perc),"% of the samples")

	# -------step 20 : Save a copy of the final dframe
	print("The final dataframe (dframe) is ready ! Lets save it in a .csv file...")
	tag_ctype = "BRCA"
	tag_drugname = "REMAGUS02_NAC" # manually recordd in the treatments details files
	tag_drugID = "Treatment11"
	tag_respType = str(cohort_used) + "x" + "NAC"+ "x" + str(total_samples) + "x" + str(total_feats)
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
	# - lets us the full path to save the file
	dframe.to_csv(fullname, index=None, header=True)
	print("File saved !")
	print(cohort_used,"dataset formatting for CICS analysis is done!")
	##! also delete all the uneccesary variables got sooner
	##! also create a log of all operations


	####-----A checkpoint to check on data
	# ---> Let's take a first look at our dataset to see what we're working with!
	# dframe[dframe.columns[:10]].head()
	# ----> Let's find out about the data types we have accross columns :
	# dframe.info()
	####-----

elif cohort_used == "REMAGUS04" :
	print("dataset treatment to add")
elif cohort_used == "MDAnderson" :
	print("dataset treatment to add")
else :
	print("no dataset known added for treatment")
#<<<<<<<----end of all that is data file specific about the preprocessing
