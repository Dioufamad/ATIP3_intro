#>>>>>>>>>>>>>>>>>>>>>>>>>>> REMAGUS02 ANNOTATIONS FORMATTING FOR EQUIVALENT TABLE CONTAINING PROBESETID-GENESSYMBOLS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import os #for bash command lines in python
import pandas as pd # for dataframes manipulation
import numpy as np # linear algebra and exploit arrays faster and easier computations
import locale
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path # to manage paths as into arguments
from cics_engines.data_engine1_mgmt import data_mgmt_6
#>>>>>>>>>>>>>>>>>>>>>>>>>>> Variables to initialise------------------------------------------
print("Initialising environnement variables...")
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
# -----needed to save the output dataset
basedir = str(Path()) #for setting the working directory to create the paths to the location of the output dataset
#---how to launch the script
# tag_decision_launching_way = "cmd_line"
tag_decision_launching_way = "line_by_line"
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
# ----for the cohort choice
# cohort_used = "REMAGUS02"
cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
print("All imports and settings are successfully placed")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> README <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<##! rewrite later
# print("Welcome in the formatting tool for the Remagus02 dataset in direction to CICS")
# print("We suppose you have done the querying of a database and you have separated values files (csv,tsv,xls, etc.).")
# print("Such values tables describe samples over multiples features, rows samples and features as columns or vice-versa.")
# print("We will try to format it into this representation : ")
# print("- a .csv file that have succesively 3 groups of columns as features with features names as the titles of the columns")
# print("+ 1 column as the classes and titled BestResCategory ")
# print("+ multiples columns, each one as a feature tityle the feature name")
# print("+ 1 column as the samples and titled Model")
# print("Necessary libraries imported.")
# print("Environnement variables initialised.")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESSING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Storing data files...")
# stock the file...
if command_center == "Gustave_Roussy" :
	file_path = "/atip3_material/3c_data_trial1/annotations/Annotations.xlsx"  # @ GR
else :  # command_center == "Home"
	file_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/atip3_material/3c_data_trial1/annotations/Annotations.xlsx" # @ GR
#...and its separator
sep_in_file = "\t"
#...specify the sheet to use
if cohort_used == "REMAGUS02" :
	sheet_id = "REMAGUS02"
else:
	sheet_id = "REMAGUS04-MDAnderson"
# --------------proper processing
print("Preprocessing...")
# Objective : obtaining a table with only 2 cols : the probesets and the gene symbols
#>>>>>-----start of everything that is data file specific in the preprocessing
# if cohort_used == "REMAGUS02" : # uncomment and this whole following part is under the if
#make a df out of the file (# ---> how to stock a dataset)
df_sup_file = pd.read_excel(file_path,sheet_id,skiprows=4) ##! add skiprows=4 to skip lines 4 lines here, default is None
# -------step 1 : rename the targeted columns
# it is better to name the manipulated columns and their give also their future names
# needed columns : probeset id col, gene bank accession umber col, gene symbol col
old_probes_id_col = "ID"
old_accessions_num_col = "GB_ACC"
old_gene_symbol_col = "Gene Symbol"
probes_id_col = "PSI"
accessions_num_col = "GBAN"
gene_symbol_col = "GS"
# 1 - change the 2 cols names
df_sup_file.rename(columns={old_probes_id_col: probes_id_col, old_accessions_num_col : accessions_num_col,old_gene_symbol_col : gene_symbol_col}, inplace=True)
# 2 - replace the nan values by a name telling what is missing explicitly
df_sup_file[probes_id_col].fillna("NA", inplace=True) # in the others column we will have a chance to put NA for the unknown values cells, we input it here also
df_sup_file[accessions_num_col].fillna("NA", inplace=True)
df_sup_file[gene_symbol_col].fillna("NA", inplace=True)
df_sup_file[accessions_num_col] = "GBANas" + df_sup_file[accessions_num_col].astype(str) + "w" + "PSIas" + df_sup_file[probes_id_col].astype(str)
df_sup_file[gene_symbol_col] = "GSas" + df_sup_file[gene_symbol_col].astype(str)
df_sup_file[gene_symbol_col] = df_sup_file[gene_symbol_col].astype(str) + "w" + df_sup_file[accessions_num_col].astype(str)
# - the 3rd column has to be kept only but duplicates can occurs it it :
# - - if so, we have rename to all ocurrences with all 3 original col components and after that we delete the duplicates except first
df_sup_file = data_mgmt_6(df_sup_file, probes_id_col) # sorting
df_sup_file_w_marked_as_duplicates_rows_only = df_sup_file[df_sup_file.duplicated()] # dropping duplicates rows # for all duplicates do duplicated(subset=["GS"],keep=False)
df_sup_file_indexes_to_drop = df_sup_file_w_marked_as_duplicates_rows_only.index
df_sup_file = df_sup_file.drop(df_sup_file_indexes_to_drop)
df_sup_file = df_sup_file.reset_index(drop=True)
# 3 - dropping the extra accession number column
df_sup_file.drop(labels=[accessions_num_col], axis=1, inplace=True)
#====>--step 4 : eliminate all probesets with nan values (not needed except for logging issue)
probesets_b4_cleaning = len(df_sup_file.axes[0])
df_sup_file.dropna(axis='index', inplace=True)
probesets_aft_cleaning = len(df_sup_file.axes[0])
lost_probesets = probesets_b4_cleaning - probesets_aft_cleaning
print("Report on the losses during the cleaning of the uncomplete probesets info of the left table : ") # a report on the losses while cleaning nan values
if lost_probesets==0:
	print("No probesets has been lost during the cleaning of the uncomplete samples info of the left table")
else:
	print(lost_probesets,"probesets has been lost during the cleaning of the uncomplete samples info of the left table")
	# -------step 9 : # the index is maybe missing now rows. reset it in a way to not get a new column
	df_sup_file = df_sup_file.reset_index(drop=True)
#====>--end of step 4 : (not needed except for logging issue)

#<<<<<<<----end of all that is data file specific about the preprocessing

# >>>>>-----formatting the two colums in the final frame
# the awaited configuration is :
# - 1 column with strings (dtype object) as the probeset name
# - 1 column with strings (dtype object) as the gene symbol
# -------step 18 : formatting the dtypes of each group of columns
df_sup_file[probes_id_col] = df_sup_file[probes_id_col].astype(str) # strings dtype is object so we have to find object
df_sup_file[gene_symbol_col] = df_sup_file[gene_symbol_col].astype(str) # strings dtype is object so we have to find object
####-----A checkpoint to check on data
# df_sup_file.info() # for an overall summary of remaining dtypes
####-----

# give an overview of what we are working with here before saving it in a file
print("Describing the obtained final samples-features-response frame...")
total_probesets = len(df_sup_file.axes[0])
print("The frame to convert",cohort_used,"dataset account for",total_probesets,"probesets")
# -------step 20 : Save a copy of the final dframe
print("The final frame is ready ! Lets save it in a .csv file...")
# the output path has 3 parts : the root until the ouput folder, the output folder, and the filename
# - lets make the file name
tag_cohort = cohort_used
tag_probesets_col = probes_id_col
tag_gene_symbol_col = gene_symbol_col
tag_num_probesets = str(total_probesets)
output_filename_for_final_dframe = str(cohort_used) + "_" + tag_probesets_col + "_" + tag_gene_symbol_col + "_" + tag_num_probesets + ".csv"
# - lets make the root until the ouput folder (the output folder excluded)
if tag_decision_launching_way in ["cmd_line","cl"]: # launch the script in a terminal
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	root_until_output_folder = os.getcwd() # obtained by changing directory
else: # launch the script line by line
	if command_center == "Gustave_Roussy":
		root_until_output_folder = "/home/amad/PycharmProjects/ATIP3_in_GR/CICS/CICS_dev_version"
	else: # command_center == "Home"
		root_until_output_folder = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version"
# - lets make the output folder
output_folder_on_same_lvl_than_main_name = "outputs"
# - lets extend the root path to contain the output folder
root_until_output_folder_w_output_folder = os.path.join(root_until_output_folder,output_folder_on_same_lvl_than_main_name)
if not os.path.exists(root_until_output_folder_w_output_folder):
	os.mkdir(root_until_output_folder_w_output_folder)
# - lets make the full path to the file to save
fullname = os.path.join(root_until_output_folder_w_output_folder,output_filename_for_final_dframe)
# - lets us the full path to save the file
df_sup_file.to_csv(fullname, index=None, header=True)
print("File saved !")
print(cohort_used,"probesets annotation formatting for features names conversion is done!")
print("File location is",fullname)
##! also delete all the uneccesary variables got sooner
##! also create a log of all operations

### EoF
