import os
import pandas as pd
import sys
import time
#------

#------> Paths to data sources
### mandatory folders to use if arguments for input files not given
basedir = os.getcwd() #for setting the working directory
data_parent_folder = "PDX__data" # the folder containing all data
drugs_data_folder = "inp" # all drugs info are consolidated here so mandatory folder for drugs data
### an interchangeable folder for the cases profiles data : uncomment one ad data_path1 is modified
profile_data_folder = "processedData" # all data
# profile_data_folder = "new_data2/newbies" # all data
# profile_data_folder = "processedDataTest_1C_1T"
# profile_data_folder = "processedDataTest_1C_1T/processedDataTest.CN" # 1 CN
# profile_data_folder = "processedDataTest_1C_1T/processedDataTest.CNA" # 1 CNA
# profile_data_folder = "processedDataTest_1C_1T/processedDataTest.GEX" # 1 GEX
# profile_data_folder = "processedDataTest_1C_1T/processedDataTest.SNV" # 1 SNV
# profile_data_folder = "processedDataTest_1C_3T"
# profile_data_folder = "processedDataTest_1C_2T_var"
# profile_data_folder = "processedDataTest_2C_1T"
# profile_data_folder = "processedDataTest_1C_1T_2P_issue_dframe" # problematic shortest test
# profile_data_folder = "processedDataTest_1C_1T_1P" # okay shortest test #gex as the one profile in it
# profile_data_folder = "processedDataTest_1C_1T_1P_issue_dframe" # shortest test discrete #cna as the one profile in it
### Paths needed to extract data (uncomment to activate a data_path1 for data matrix and a data_path2 for names of the drugs tested
data_profiles_path = basedir+"/"+data_parent_folder+"/"+profile_data_folder # the profiles data
data_drugs_path = basedir+"/"+data_parent_folder+"/"+drugs_data_folder # the drugs data (not used here for now)

# get the time to name the files
timestr = time.strftime("%Y%m%d-%H%M%S")
# print(timestr) " for testing

#========>going through the profiles files
#>>>>Redirecting stdout to .o file
original_out = sys.stdout
sys.stdout = open(basedir + "/" + "PDX__data" + "/" + "stats_on_data" + "/" + "stats_at_date_" + timestr + ".txt", 'w')
# print files found characteristics
print("The following files have been found in the directory",data_profiles_path," and are described with following characteristics to the analysis :")
for root, directories, filenames in os.walk(data_profiles_path):
	for file in filenames:
		# print(file)
		# printing each file short info
		df = pd.read_csv(os.path.join(root, file), sep=",")
		print(file,"has",df.shape[0]," samples (rows) and",((df.shape[1])-2),"features (cols)")
####end of printing data characteristics
sys.stdout = original_out # linked to testing redirection of stdout




