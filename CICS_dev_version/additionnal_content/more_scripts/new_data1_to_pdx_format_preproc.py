##### script for the datamanagement to create Linh's datasets like files for adealo's data :
import pandas as pd # for dataframes manipulation
import os


# Step 1 : lets capture drugid and drugnames (drug data) to isolate it in a file as a source of drug data
# lets intialise here our collectors
list_col_of_drugnames =[]
# lets go through our batch of files
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/new_data1"): # change here folder of prof in ctype1
	for file in filenames:
		print(file)
		df_new = pd.read_csv(os.path.join(root, file), sep="\t") # , header=None # , sep='delimiter', header=None helps pandas to know the right separator to use if others separators exist in the file
		# getting the different treatments
		# df_col_of_drugs = df_new.loc[:, ["drug_name"]]
		unique_list_of_drug_for_present_frame = df_new["drug_name"].unique().tolist()
		for drug_met in unique_list_of_drug_for_present_frame :
			if drug_met not in list_col_of_drugnames:
				list_col_of_drugnames.append(drug_met)
		# creating the different
# making the final data for the treatments after having reviewed all the analysed datasets
list_col_of_drugnames_sorted = sorted(list_col_of_drugnames)
list_of_IDs_for_all_drugs_met = list(range(1,len(list_col_of_drugnames_sorted)+1))
dict_drugID_drugName = {'Treatment_ID' : list_of_IDs_for_all_drugs_met,'Treatment_Details' : list_col_of_drugnames_sorted}
df_of_treatments_data = pd.DataFrame(dict_drugID_drugName)
# saving the treatments data to a file knowing all the treatments have been met and their info registered
filename_for_drugs_data = "new_treatments_data.csv"
# save df as a .csv file ready for analysis
df_of_treatments_data.to_csv(filename_for_drugs_data, index=None, header=True)
#=============================================================
# Step 2 : lets subdivise the big datasets
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/new_data1"): # change here folder of prof in ctype1
	for file in filenames:
		print(file)
		filename1 = file
		filename1_corrected = filename1.replace('.', '_')
		elts_of_filename1 = filename1_corrected.split("_")
		# lets capture ctype, profile and the type of response to put in the created files filenames
		ctype1 = elts_of_filename1[0] # ctype caught
		profile1 = elts_of_filename1[1][:3]  # take the first 3 letters of the 2nd elt of the split # profile caught
		response_type1 = "BestResCat" # response type caught
		print("-> Going through a file with ctype as",ctype1,"and profile as",profile1)
		# making the Response and Model columns
		df_new = pd.read_csv(os.path.join(root, file), sep="\t") # , header=None # , sep='delimiter', header=None helps pandas to know the right separator to use if others separators exist in the file
		df_new = df_new.rename(columns={"measure_of_response": "BestResCategory"})
		df_new['Model'] = df_new['bcr_patient_barcode'].str.cat(df_new['Aliquot_id'], sep="asPBwAid")
		df_new.drop(labels=['bcr_patient_barcode'], axis=1, inplace=True)
		df_new.drop(labels=['Aliquot_id'], axis=1, inplace=True)
		# making the file for each treatment
		# - getting the different treatments in the file
		# this_df_col_of_drugs = df_new.loc[:, ["drug_name"]]
		this_df_unique_list_of_drugs = df_new["drug_name"].unique().tolist()
		# - for each treatment, saving a file restricted to that treatment
		for drug_met in this_df_unique_list_of_drugs:
			# get the proper id in all the treatments
			index_drug_for_id_search = dict_drugID_drugName["Treatment_Details"].index(drug_met)
			id_for_drug = dict_drugID_drugName["Treatment_ID"][index_drug_for_id_search]
			print("----> for", drug_met, "ID is", id_for_drug)
			drugid = "Treatment" + str(id_for_drug) # drugid caught
			# lets create the restricted df to only this drug met
			df_restricted = df_new.loc[df_new["drug_name"] == drug_met]
			df_restricted = df_restricted.reset_index(drop="True")
			df_restricted.drop(labels=['drug_name'], axis=1, inplace=True)
			# saving the treatments data to a file knowing all the treatments have been met and their info registered
			df_restricted_filename = ctype1 + "_" + drugid + "_" + response_type1 + "_" + profile1 + ".csv"
			# save df as a .csv file ready for analysis
			df_restricted.to_csv(df_restricted_filename, index=None, header=True)
			print("----> new file created for as ", df_restricted_filename)