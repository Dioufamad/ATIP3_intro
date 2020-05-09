import os
import pandas as pd
import time
#------

#------> Paths to data sources
### mandatory folders to use if arguments for input files not given
basedir = os.getcwd() #for setting the working directory
data_parent_folder = "PDX__data" # the folder containing all data
drugs_data_folder = "inp" # all drugs info are consolidated here so mandatory folder for drugs data
### an interchangeable folder for the cases profiles data : uncomment one ad data_path1 is modified
profile_data_folder = "processedData" # all data
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

# ===================in the end, the drugs names will be needed so let us also extract them
df_treatments = pd.DataFrame()  # a df to stock our frame contains names of the treatments
for root, directories, filenames in os.walk(data_drugs_path):
	for file in filenames:
		if '_treatments' in file:
			# print(os.path.join(root,file))
			print("Analysing the following drugs information files in this directory : ",data_drugs_path)
			# print(file)
			df_treatments = pd.read_csv(os.path.join(root, file))
df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].copy()
df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].astype(str)
df_treatments["Treatment_ID"] = 'Treatment' + df_treatments["Treatment_ID"].astype(str)
mydrugs = df_treatments.set_index('Treatment_ID').T.to_dict('list')  # a dict that will be called will filling the result file -returned-
# temporary fix for having drugnames as list of one element ##!! correct it later by change exactly whats done earlier
for key in list(mydrugs.keys()):
	value_of_key = mydrugs[key]
	new_value_of_key = value_of_key[0]
	mydrugs[key] = new_value_of_key

#========>going through the profiles files
#>>>>introducing the big df for caracteristics
big_df = pd.DataFrame(columns=["Ctype","TreatmentID","TreatmentName","Profile","# Samples","#Features"])
# print files found characteristics
print("The following files have been found in the directory",data_profiles_path," and will be described with their characteristics of content in the resulting frame.")
for root, directories, filenames in os.walk(data_profiles_path):
	for file in filenames:
		print(file)
		# stock in df to know shape (# of rows and cols)
		df = pd.read_csv(os.path.join(root, file))
		# creating a df with all info lines and complete it with results
		info_material = file
		info_material_no_dot = info_material.replace('.', '_')
		info_material_splited_by_ = info_material_no_dot.split("_")
		index_line_to_write_in_bigdf = len(big_df)
		# adding a row of : # ctype value # TreatmentID value # drugname value # profiletype value # number of samples value # number of features value
		big_df.loc[index_line_to_write_in_bigdf] = [info_material_splited_by_[0],
			info_material_splited_by_[1],
			mydrugs[info_material_splited_by_[1]],
			info_material_splited_by_[3],
			df.shape[0],
			((df.shape[1])-2)]

# sort the df following the the treatments and the ctypes : for one treatment, what are the ctype applied to and
# what are the results
big_df.sort_values("Ctype", axis=0, ascending=True, inplace=True, kind='mergesort') # priority 2 for order is ctypes
big_df.sort_values("TreatmentID", axis=0, ascending=True, inplace=True, kind='mergesort') # priority 1 for order is treatments
# after the precedent sort, the indexes are not in order. make a new order for them
big_df = big_df.reset_index(drop="True")
# push our big_df into a .csv file
output_filename_for_bigdf = basedir + "/" + "outputs" + "/" + "Evolution_on_all_prod_with_PDX_data_started_" + timestr + ".csv"
big_df.to_csv(output_filename_for_bigdf, index=None, header=True)
# make another code snippet to sort again when values are obtained