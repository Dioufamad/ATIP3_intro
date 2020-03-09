###--------------------- This is the location of some functions caring for data management-----------------------

###---------------------IMPORTS
import os # for files and directories exploration
import pandas as pd # for dataframes manipulation
# from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
import locale
from slate_engines.data_engine2_allocation import add_entry_in_dict # to update the data_loadout_right (dictionnary) following profiles exist (add to values of a key) or not (create a new key and add as first value)
#====================================================================

# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#====================================================================

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Data management strategy 1 (for NIBR-PDXE data) : GET ALL THE VARIABLES OF THE PROBLEM
# we are trying to catch :
# + the cell lines (instances)
# + the drug used
# + the cancer type on which it have been used
# + the response of the drug used on that cancer type
# + the features :
# ++ SNV
# ++ CNA
# ++ RNA (GEx)
# ++ Meth
###===> we will .... (explain clearly later)

# =================== extract the raw tables for each profiles with the related response (each -returned- means it has to be returned by the function)
def data_mgmt_1(data_path1,data_path2,previous_Resp_col_name,previous_Samples_col_name):
	# initialisations of collectors
	data_loadout_left = []  # a list of all df for each sample_id-cancertype_treatment_id-profile_type (to be concatenated later)
	data_loadout_right = {} # a list of all sample_id-resp-features_of_GEX_profile (to not concantenate because not sure if all features are a alike in each source) -returned-
	data_loadouts_origin_files = {} #----origin files loadout # a list of all the files used as sources of the load_outs -returned-
	# data files exploration and management for collection in two loadouts
	print("Following files included to the analysis as sources of features information :")
	for root, directories, filenames in os.walk(data_path1):
		for file in filenames:
			# print(os.path.join(root,file)) # for testing (display the full path of the file)
			print(file)
			# "df"+str(k)
			df_right = pd.read_csv(os.path.join(root, file))
			# we need controlled columns names for precise data management
			Resp_col_name = "Resp_Class"
			Samples_col_name1 = "Sample_id_bis"
			Samples_col_name2 = "Sample_id"
			Condittion_col_name = 'Condittion' # previously 'Cancer_type'
			TreatmentID_col_name = 'Treatment_id' # always been 'Treatment_id'
			Layer_probed_col_name = 'Layer_probeb' # previously 'Profile_type'
			Unifier_key_col_name = 'Unifier_key'
			# rename the column for response and samples, to control their names and target them more easily throughout future processes in the analysis
			df_right = df_right.rename(columns={previous_Resp_col_name: Resp_col_name}) # previous_Resp_col_name = "BestResCategory" or defined by argument
			df_right = df_right.rename(columns={previous_Samples_col_name: Samples_col_name1})  # previous_Samples_col_name = "Model" or defined by argument
			# move them response column at start and then the samples columns, because they are renamed in controlled names, they can be targeted wherever they are in the dataframe created from the dataset
			#-for the response col
			Resp_column_to_move = df_right[Resp_col_name]
			df_right.drop(labels=[Resp_col_name], axis=1, inplace=True)
			df_right.insert(0, Resp_col_name, Resp_column_to_move)
			# -for the samples col
			Samples_column_to_move = df_right[Samples_col_name1]
			df_right.drop(labels=[Samples_col_name1], axis=1, inplace=True)
			df_right.insert(0, Samples_col_name1, Samples_column_to_move)
			unifier_builder1 = file
			unifier_builder1 = unifier_builder1.replace('.', '_')
			unifier_builder1 = unifier_builder1.split("_")
			profile_type_found = unifier_builder1[3]
			unifier_builder1 = [unifier_builder1[i] for i in [0, 1, 3]]
			unifier_builder2 = "_".join(unifier_builder1)
			df_right.insert(0, Unifier_key_col_name, unifier_builder2)
			# lets make sure the load out going to the right is without nan
			df_right.dropna(axis='columns')
			# df_right=df_right.drop_duplicates()
			# adding the df_right to the dict of data_loadout_right
			add_entry_in_dict(data_loadout_right, profile_type_found, df_right)

			# the dataframe unifierkey-sample_id-resp-features is tucked away for later
			# lets create the other dataframe
			df_left = pd.DataFrame()
			df_left.insert(0, Samples_col_name1, df_right[Samples_col_name1])
			df_left = df_left.rename(columns={Samples_col_name1: Samples_col_name2})  # rename this column to drope asily the other one later after unification for looping on data
			df_left.insert(1, Condittion_col_name, unifier_builder1[0])
			df_left.insert(2, TreatmentID_col_name, unifier_builder1[1])
			df_left.insert(3, Layer_probed_col_name, unifier_builder1[2])
			# # lets make sure the load out going to the left is without nan # not needed because we built it
			# df_left.dropna()
			data_loadout_left.append(df_left)
			# the dataframe sample_id-cancertype-treatment_id-profile_type is kept away
			# the file used is kept in case we need to check concordances (at each index of the 3 data_loadout list, the elements are related)
			# adding the origin files to the dict of data_loadouts_origin_files
			add_entry_in_dict(data_loadouts_origin_files, profile_type_found, file)
			# ....another file is parsed to do the same two dataframes creations or the loop is ended
			# file1=file #uncomment for testing (show the current file)
	# #=========================================================================================================
	# =================== reduce the left load out to only keep unique sets in it
	# we can concatenate all of the dataframes contained in the list because we are sure that they have the same columns
	data_loadout_left_all = pd.concat(data_loadout_left)
	# lets drop the entries with nan
	# final_data_loadout_left.dropna()
	# we obtain a big dataframe of 114 rows x 4 cols
	data_loadout_left_all.reset_index(drop=True)  # reseting the index for a continuous count from zero at the first line until x with x the number of samples/biological problem -returned-
	# ===================in the end, the drugs names will be needed so let us also extract them
	df_treatments = pd.DataFrame()  # a df to stock our frame contains names of the treatments
	for root, directories, filenames in os.walk(data_path2):
		for file in filenames:
			if '_treatments' in file:
				# print(os.path.join(root,file))
				print("Following files included to the analysis as sources of treatments information : ")
				print(file)
				df_treatments = pd.read_csv(os.path.join(root, file))
	df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].copy()
	df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].astype(str)
	df_treatments["Treatment_ID"] = 'Treatment' + df_treatments["Treatment_ID"].astype(str)
	dict_TreatmentID_TreatmentName = df_treatments.set_index('Treatment_ID').T.to_dict('list')  # a dict that will be called will filling the result file -returned-
	#temporary fix for having drugnames as list of one element ##!! correct it later by change exactly whats done earlier
	for key in list(dict_TreatmentID_TreatmentName.keys()):
		value_of_key = dict_TreatmentID_TreatmentName[key]
		new_value_of_key = value_of_key[0]
		dict_TreatmentID_TreatmentName[key] = new_value_of_key
	################# Building elements to loop on for analysis :
	# loop 1 : on the cancertype. it needs a set of all the cancertype in the data_loadout_left
	set_of_all_condittions_found = set(data_loadout_left_all[Condittion_col_name])
	set_of_all_condittions_found = {c for c in set_of_all_condittions_found if c == c}
	set_of_all_condittions_found = sorted(set_of_all_condittions_found) # -returned-
	# ----------an testing idea is to only launch a test analysis with the ctypes of interest
	# ex: BRCA,COADREAD,LUAD,SCLC,SKCM
	# a_given_list_of_ctypes = ["BRCA","COADREAD","LUAD","SCLC","SKCM"]
	# ctypes = filter(lambda i: i in a_given_list_of_ctypes,ctypes)
	#  -------------
	# lets show the cancertypes we will run with
	print("Based on collected information, here is a set of cancer types to explore : ", set_of_all_condittions_found) # rectify as for printing elts in list

	# loop 2 : on the treatment_id. it needs a set of all the treatment_ids in the data_loadout_left
	set_of_all_TreatmentIDs_found = set(data_loadout_left_all[TreatmentID_col_name])
	set_of_all_TreatmentIDs_found = {d for d in set_of_all_TreatmentIDs_found if d == d}
	set_of_all_TreatmentIDs_found = sorted(set_of_all_TreatmentIDs_found) # -returned-
	# ----------an testing idea is to only launch a test analysis with the drugs of interest
	# ex: 'Treatment15', 'Treatment17', 'Treatment18'
	# a_given_list_of_drugs = ['Treatment15', 'Treatment17', 'Treatment18']
	# drugs = filter(lambda i: i in a_given_list_of_drugs,drugs)
	#  -------------
	# lets show the drugs we will run with
	print("Based on collected information, here is a set of drugs to explore : ", set_of_all_TreatmentIDs_found)

	# loop 3 : on the profile_type. it needs a set of all the profiles_types in the data_loadout_left
	set_of_all_profiles_found = set(data_loadout_left_all[Layer_probed_col_name])
	set_of_all_profiles_found = {p for p in set_of_all_profiles_found if p == p}
	set_of_all_profiles_found = sorted(set_of_all_profiles_found) # -returned-
	# ----------an testing idea is to only launch a test analysis with the profiles of interest
	# ex: 'GEX'
	# a_given_list_of_mylabels = ['GEX']
	# mylabels = filter(lambda i: i in a_given_list_of_mylabels,mylabels)
	#  -------------
	# lets show the drugs we will run with
	print("Based on collected information, here is a set of profiles to explore : ", set_of_all_profiles_found)
	return data_loadout_left_all, data_loadout_right, data_loadouts_origin_files, \
		set_of_all_condittions_found, set_of_all_TreatmentIDs_found, dict_TreatmentID_TreatmentName, set_of_all_profiles_found, \
		Resp_col_name,Samples_col_name1,Samples_col_name2,Unifier_key_col_name,\
		Condittion_col_name,TreatmentID_col_name,Layer_probed_col_name

# =================== join the left table appropriate part (ctype,drug, profile, unifier) and the right table corresponding (unifier,response, features of the profile)
# lets define a preliminary function that create a unifier
def unifier_creator(ctype, drug, featuretype):
	unifier_material = [ctype, drug, featuretype] # this is the material for the unifier
	unifier = "_".join(unifier_material)
	return unifier
# lets define a preliminary function that choose the corresponding data_loadout_right regarding the presently analysed profile
def data_loadout_right_corresponder(featuretype,data_loadout_right):
	# lets find the appropriate dataframe with corresponding features and response
	corresponding_data_loadout_right = data_loadout_right[featuretype]
	return corresponding_data_loadout_right

# finally the function that join left side data and right side data using a unifier
def data_mgmt_2(unifier,featureframe,corresponding_data_loadout_right,Samples_col_name1,Samples_col_name2,Unifier_key_col_name):
	dframe = pd.DataFrame()
	profile_data_archived = pd.DataFrame()
	for dataframe in corresponding_data_loadout_right:
		if not corresponding_data_loadout_right.index(dataframe) == (len(corresponding_data_loadout_right)-1):
			if not dataframe.iloc[0][Unifier_key_col_name] == unifier:
				pass
			else :
				dframe = pd.merge(featureframe, dataframe, how="inner", left_on=Samples_col_name2, right_on=Samples_col_name1)  # link with cosmic ids
				dframe.drop(labels=[Samples_col_name1, Unifier_key_col_name], axis=1, inplace=True)
				# dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(bool)  # change values 0 and 1 of cna into true and false
				profile_data_archived = dataframe
				break
		else : # the last element of the list is treated differently to send a message of data not found
			if not dataframe.iloc[0][Unifier_key_col_name] == unifier:
				print("No correspondant feature data found for ", " ".join([str(x) for x in unifier.split("_")]))
			else :
				dframe = pd.merge(featureframe, dataframe, how="inner", left_on=Samples_col_name2, right_on=Samples_col_name1)  # link with cosmic ids
				dframe.drop(labels=[Samples_col_name1, Unifier_key_col_name], axis=1, inplace=True)
				# dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(bool)  # change values 0 and 1 of cna into true and false
				profile_data_archived = dataframe
	index_starting_fts_cols = 5 # index from which fts started in the final dframe
	return dframe,profile_data_archived,index_starting_fts_cols

# ---------------------------change values of fts into floats or bool
def data_mgmt_5(dframe,index_of_1st_ft,feature_val_type):
	# change values 0 and 1 of binaries profiles into true and false or if reals, make them floats
	if feature_val_type == "cat" : # case of binary profiles
		# dframe[list(dframe)[index_of_1st_ft:]] = dframe[list(dframe)[index_of_1st_ft:]].astype(bool)  # deprecated because sets off a SettingwithCopyWarning
		# dframe = dframe.astype({list(dframe)[index_of_1st_ft:]: bool}) # deprecated because not efficient enough (same for the for loop version of this)
		# dframe[list(dframe)[index_of_1st_ft:]] = dframe[list(dframe)[index_of_1st_ft:]].astype('bool') # latest solution but still sets off a SettingwithCopyWarning
		# -----------> use following in case of SettingwithCopyWarning set off
		for col_as_ft_going_bool in list(dframe)[index_of_1st_ft:]:
			if dframe[col_as_ft_going_bool].dtypes != "bool":
				# print("The feature ", col_as_ft_going_bool,  "was found to not be bool and will be forced into it...")
				dframe[col_as_ft_going_bool] = dframe[col_as_ft_going_bool].astype('bool')
		print("All features columns changed into boolean dtype...")
	else : # if a case of real values profiles
		# dframe[list(dframe)[index_of_1st_ft:]] = dframe[list(dframe)[index_of_1st_ft:]].astype(float) # deprecated because sets off a SettingwithCopyWarning
		# dframe[list(dframe)[index_of_1st_ft:]] = dframe[list(dframe)[index_of_1st_ft:]].astype('float64') # deprecated because sets off a SettingwithCopyWarning
		# dframe[list(dframe)[index_of_1st_ft:]] = dframe[list(dframe)[index_of_1st_ft:]].apply(pd.to_numeric)
		# dframe.loc[:,list(dframe)[index_of_1st_ft:]] = dframe.loc[:,list(dframe)[index_of_1st_ft:]].astype(float) # latest solution but still sets off a SettingwithCopyWarning
		# -----------> new method (useful also this a verification process in case some columns are "int64" instead of float64...
		for col_as_ft_going_float in list(dframe)[index_of_1st_ft:]:
			if dframe[col_as_ft_going_float].dtypes != "float64":
				# print("The feature ", col_as_ft_going_float,  "was found to not be float and will be forced into it...")
				dframe[col_as_ft_going_float] = dframe[col_as_ft_going_float].astype('float64')
		print("All features columns changed into float64 dtype...")
	return dframe

# ---------------------------sort the dataframe entries following response col values and remake a new index
def data_mgmt_6(dframe,Resp_col_name):
	# sort the df following the values of the resp column
	dframe.sort_values(Resp_col_name, axis=0, ascending=True, inplace=True, kind='mergesort')
	# after the precedent sort, the indexes are not in order. make a new order for them
	dframe = dframe.reset_index(drop="True")
	return dframe

# ---------------------------reduction of the final dataframe for testing purposes
def reduction_of_dataset_for_testing_purpose(dframe,Resp_col_name,classif_reduc_data_sn):
	dataBin = dframe.loc[:,[Resp_col_name]] # extract the two predicted classes (binary classif)
	RespClasses = sorted(dataBin.iloc[:, 0].unique())
	class_neg_indexes = dframe.index[dframe[Resp_col_name] == RespClasses[0]].tolist()  # isolate the uniques values for each group of the two response in the population
	class_pos_indexes = dframe.index[dframe[Resp_col_name] == RespClasses[1]].tolist()
	if (classif_reduc_data_sn % 2) == 0 : # if the number of samples to keep is even, keep half of it in each of the classes (binary classif). if odd, make the pos class have on more sample than neg class
		num_class_neg_indexes_to_keep = int(classif_reduc_data_sn / 2)
		num_class_pos_indexes_to_keep = int(classif_reduc_data_sn / 2)
	else:
		num_class_neg_indexes_to_keep = int(classif_reduc_data_sn / 2)
		num_class_pos_indexes_to_keep = int(classif_reduc_data_sn / 2) + 1
	class_neg_indexes_to_keep = class_neg_indexes[:num_class_neg_indexes_to_keep] # getting the list of indexes correspond in each class to top "half of what we want to keep"
	class_pos_indexes_to_keep = class_pos_indexes[:num_class_pos_indexes_to_keep]
	all_indexes_to_keep = class_neg_indexes_to_keep + class_pos_indexes_to_keep # putting the indexes together (sum of lists is a list)
	dframe_reduced = dframe.iloc[all_indexes_to_keep,:] # make a new df with it from the old one
	# # sorting again the df and rebuilding index in right order is done in following operations wether dataset is reduced or not so no need to bother doing it here
	return dframe_reduced

# ---------------change values of response into floats or bool ##!! review where to put in the script ##!! to remake to format response
# def data_mgmt_3(dframe,task_type,Resp_col_name):
# 	if task_type == "Regr":
# 		# change values of response into floats
# 		dframe[Resp_col_name] = dframe[Resp_col_name].astype(float)
# 		print("Response values formated for a regression analysis")
# 	elif task_type == "Classif" :
# 		# change values of response into classes (0 and 1)
# 		le = LabelEncoder()
# 		y_encoded = le.fit_transform(dframe[Resp_col_name])
# 		dframe[Resp_col_name] = y_encoded
# 		print("Response values formated for a classification analysis")
# 	return dframe

# # ------------------------------------------------------------##!! to remake as mgmt6 as an option
# def data_mgmt_4(dframe,Resp_col_name): # to eliminate features with only zeros as values, sort the dataframe entries following repsonse col values and remake a new index
# 	# enlever les colonnes nayant que des zeros (feature ne servant a aucune differenciation)
# 	# (dframe != 0).any(axis=0) est un bool avec False pour col n'a que des zeros;
# 	# df.loc[:,x] = toutes les lignes des col avec la cond x; puis une liste de telles cols est prise, puis le df de cette liste de col est prise
# 	dframe = dframe[list(dframe.loc[:, (dframe != 0).any(axis=0)])]
# 	# sort the df following the values of the resp column
# 	dframe.sort_values(Resp_col_name, axis=0, ascending=True, inplace=True, kind='mergesort')
# 	# after the precedent sort, the indexes are not in order. make a new order for them
# 	dframe = dframe.reset_index(drop="True")
# 	return dframe
# # -----------------------------------------------------------------

# a newest version of the function to decide the feature values type
def feature_values_type_caracterisation(a_feature_type,list_of_cat_fts):
	cat_fts_types_endings = ["V","v","A","a"] # e.g. : a_feature_type = "SNV", a_feature_type = "snv", a_feature_type = "GEXA", a_feature_type = "cna"
	if (a_feature_type in list_of_cat_fts) | (a_feature_type.endswith(tuple(cat_fts_types_endings))):
		feature_val_type = "cat"
	else:
		feature_val_type = "real"
	return feature_val_type
