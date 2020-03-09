###--------------------- By Diouf, This is the location2 of some functions unused but that can come in handy-----------------------
# ##!! implement in place of eliminate_non_variable_fts() from fs.engine
# def dropping_only_zeros_variables_from_aframe(frame):
# 	# testing df
# 	# frame = pd.DataFrame([[1, 2, 3, 0], [0, 0, 0, 0], [4, 5, 6, 0], [0, 0, 0, 0]], columns=list('abcd'))
# 	# dropping on the rows axis
# 	frame.drop(frame.index[frame.apply(lambda row: (row.nunique(dropna=False) == 1) & (row.unique()[0] == 0), axis=1)], axis=0)
# 	# dropping on the cols axis
# 	frame.drop(frame.columns[frame.apply(lambda row: (row.nunique(dropna=False) == 1) & (row.unique()[0] == 0), axis=0)], axis=1)
# 	# explained :
# 	# frame.drop(a_list_of_index_values, axis=0) drops on the rows axis and frame.drop(a_list_of_col_names, axis=1) drops on the cols axis
# 	# frame.columns[list_of_booleans] is a list of cols that correspond to positions of booleans that are True
# 	# frame.index[list_of_booleans] is a list of indexes ie rownames that correspond to positions of booleans that are True
# 	# frame.apply(lambda x: function K on x, axis=1) iterates on the axis (chosen with 1 for rows and 0 for cols in Python3.7)
# 	# and execute a function K on each x with x being each elementary element met when iterating over the axis chosen. frame.apply() returns a list containing the result
# 	# the result is here a list of booleans obtained from 2 condittions
# 	# condittion 1 : True if each col/row has only one unique value
# 	# condittion 2 : True if each col/row is such that the list of unique values has as first element a zero. in the event of first condittion being True, that is the only value in the list
# 	return frame
#--------------------------calculations that are simple yet tricky
# import decimal
# context = decimal.getcontext()
# >>> context.rounding = decimal.ROUND_HALF_UP
# >>> round(decimal.Decimal('2.5'), 0)
# Decimal('3')
# def rounding_to_half_down():
# 	context = decimal.getcontext()
# 	context.rounding = decimal.ROUND_HALF_DOWN
# 	round(decimal.Decimal('2.75'), 0)
#
# 	return
# import decimal
# decimal.getcontext().rounding = decimal.ROUND_HALF_DOWN
# int(decimal.Decimal(2.2).to_integral_value())
# type(int(decimal.Decimal(2.75).to_integral_value()))
#-------------------------------

# old version of data_mgmt_2
# def data_mgmt_2_old(ctype, drug, featuretype, data_loadout_right_snv, data_loadout_right_cna, data_loadout_right_gex, data_loadout_right_cn, drugframe):
#     unifier_builder3 = [ctype, drug, featuretype]  # this is the material for the unifier
#     dframe = pd.DataFrame()
# 	profile_data = pd.DataFrame()
#     # snv_data = pd.DataFrame()
#     # cna_data = pd.DataFrame()
#     # gex_data = pd.DataFrame()
#     # cn_data = pd.DataFrame()
#     if featuretype == "SNV":  # snv condittion
#         # lets find the appropriate dataframe with corresponding features and response
#         # snv_data = pd.DataFrame()
#         for snv_dataframe in data_loadout_right_snv:
#             if snv_dataframe.iloc[0]['Unifier_key'] == "_".join(unifier_builder3):
#                 snv_data = snv_dataframe
#                 dframe = pd.merge(drugframe, snv_data, how="inner", left_on="Sample_id", right_on="Sample_id_bis")  # link with cosmic ids
#                 dframe.drop(labels=['Sample_id_bis', 'Unifier_key'], axis=1, inplace=True)
#                 dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(bool)  # change values 0 and 1 of cna into true and false
#                 break
#             else:
#                 pass
#                 if data_loadout_right_snv[-1].equals(snv_dataframe):
#                     print ("No correspondant feature data found for ", ctype, drug, featuretype)
#     elif featuretype == "CNA":  # cna condittion
#         # cna_data = pd.DataFrame()
#         for cna_dataframe in data_loadout_right_cna:
#             if cna_dataframe.iloc[0]['Unifier_key'] == "_".join(unifier_builder3):
#                 cna_data = cna_dataframe
#                 dframe = pd.merge(drugframe, cna_data, how="inner", left_on="Sample_id", right_on="Sample_id_bis")  # link with cosmic ids
#                 dframe.drop(labels=['Sample_id_bis', 'Unifier_key'], axis=1, inplace=True)
#                 dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(bool)  # change values 0 and 1 of cna into true and false
#                 break
#             else:
#                 pass
#                 if data_loadout_right_cna[-1].equals(cna_dataframe):
#                     print ("No correspondant feature data found for ", ctype, drug, featuretype)
#     elif featuretype == "GEX":  # gex condittion
#         # gex_data = pd.DataFrame()
#         for gex_dataframe in data_loadout_right_gex:
#             if gex_dataframe.iloc[0]['Unifier_key'] == "_".join(unifier_builder3):
#                 gex_data = gex_dataframe
#                 dframe = pd.merge(drugframe, gex_data, how="inner", left_on="Sample_id", right_on="Sample_id_bis")  # link with cosmic ids
#                 dframe.drop(labels=['Sample_id_bis', 'Unifier_key'], axis=1, inplace=True)
#                 dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(float)  # change values of cna into floats
#                 break
#             else:
#                 pass
#                 if data_loadout_right_gex[-1].equals(gex_dataframe):
#                     print ("No correspondant feature data found for ", ctype, drug, featuretype)
#     elif featuretype == "CN":  # can only be cn condittion for sure because thats the only one non explored in mylabels (list of the profiles)
#         # cn_data = pd.DataFrame()
#         for cn_dataframe in data_loadout_right_cn:
#             if cn_dataframe.iloc[0]['Unifier_key'] == "_".join(unifier_builder3):
#                 cn_data = cn_dataframe
#                 dframe = pd.merge(drugframe, cn_data, how="inner", left_on="Sample_id", right_on="Sample_id_bis")  # link with cosmic ids
#                 dframe.drop(labels=['Sample_id_bis', 'Unifier_key'], axis=1, inplace=True)
#                 dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(float)  # change values of cna into floats
#                 break
#             else:
#                 pass
#                 if data_loadout_right_cn[-1].equals(cn_dataframe):
#                     print ("No correspondant feature data found for ", ctype, drug, featuretype)
#     return dframe, snv_data, cna_data, gex_data, cn_data  # a full return with last component from right to control
#     # return dframe  # a return with only the dframe that will be used in the rest of the code (use this for cleaner code)

# old version of mgmt_1
# # =================== extract the raw tables for each profiles with the related response (each -returned- means it has to be returned by the function)
# def data_mgmt_1(data_path1,data_path2,Resp_col_name):
# 	data_loadout_left = []  # a list of all df for each sample_id-cancertype_treatment_id-profile_type (to be concatenated later)
# 	data_loadout_right_gex = []  # a list of all sample_id-resp-features_of_GEX_profile (to not concantenate because not sure if all features are a alike in each source) -returned-
# 	data_loadout_right_cn = []  # a list of all sample_id-resp-features_of_CN_profile (to not concantenate because not sure if all features are a alike in each source) -returned-
# 	data_loadout_right_snv = []  # a list of all sample_id-resp-features_of_SNV_profile (to not concantenate because not sure if all features are a alike in each source) -returned-
# 	data_loadout_right_cna = []  # a list of all sample_id-resp-features_of_CNA_profile (to not concantenate because not sure if all features are a alike in each source) -returned-
# 	data_loadouts_origin_files_for_gex = []  # a list of all the files used as sources of the load_outs -returned-
# 	data_loadouts_origin_files_for_cn = []  # a list of all the files used as sources of the load_outs -returned-
# 	data_loadouts_origin_files_for_snv = []  # a list of all the files used as sources of the load_outs -returned-
# 	data_loadouts_origin_files_for_cna = []  # a list of all the files used as sources of the load_outs -returned-
# 	print("The following files have been included to the analysis :")
# 	for root, directories, filenames in os.walk(data_path1):
# 		for file in filenames:
# 			# print (file)
# 			if file.endswith('_GEX.csv'):
# 				# print(os.path.join(root,file))
# 				print(file)
# 				# "df"+str(k)
# 				df_right = pd.read_csv(os.path.join(root, file))
# 				df_right = df_right.rename(columns={"BestResCategory": Resp_col_name})  # rename the cloumn for response into Resp_class
# 				df_right = df_right.rename(columns={"Model": "Sample_id_bis"})
# 				column_to_move = df_right['Sample_id_bis']
# 				df_right.drop(labels=['Sample_id_bis'], axis=1, inplace=True)
# 				df_right.insert(0, 'Sample_id_bis', column_to_move)
# 				unifier_builder1 = file
# 				unifier_builder1 = unifier_builder1.replace('.', '_')
# 				unifier_builder1 = unifier_builder1.split("_")
# 				unifier_builder1 = [unifier_builder1[i] for i in [0, 1, 3]]
# 				unifier_builder2 = "_".join(unifier_builder1)
# 				df_right.insert(0, 'Unifier_key', unifier_builder2)
# 				# lets make sure the load out going to the right is without nan
# 				df_right.dropna()
# 				# df_right=df_right.drop_duplicates()
# 				data_loadout_right_gex.append(df_right)
# 				# the dataframe unifierkey-sample_id-resp-features is tucked away for later
# 				# lets create the other dataframe
# 				df_left = pd.DataFrame()
# 				df_left.insert(0, 'Sample_id_bis', df_right['Sample_id_bis'])
# 				df_left = df_left.rename(columns={"Sample_id_bis": "Sample_id"})  # rename this column to drope asily the other one later after unification for looping on data
# 				df_left.insert(1, 'Cancer_type', unifier_builder1[0])
# 				df_left.insert(2, 'Treatment_id', unifier_builder1[1])
# 				df_left.insert(3, 'Profile_type', unifier_builder1[2])
# 				# lets make sure the load out going to the left is without nan
# 				df_left.dropna()
# 				data_loadout_left.append(df_left)
# 				# the dataframe sample_id-cancertype-treatment_id-profile_type is kept away
# 				data_loadouts_origin_files_for_gex.append(file)
# 			# the file used is kept in case we need to check concordances (at each index of the 3 data_loadout list, the elements are related)
# 			# another file is parsed to do the same two dataframes creations or the loop is ended
# 			# file1=file #uncomment to test the current file
# 			if file.endswith('_CN.csv'):
# 				# print(os.path.join(root,file))
# 				print(file)
# 				# "df"+str(k)
# 				df_right = pd.read_csv(os.path.join(root, file))
# 				df_right = df_right.rename(
# 					columns={"BestResCategory": Resp_col_name})  # rename the cloumn for response into Resp_class
# 				df_right = df_right.rename(columns={"Model": "Sample_id_bis"})
# 				column_to_move = df_right['Sample_id_bis']
# 				df_right.drop(labels=['Sample_id_bis'], axis=1, inplace=True)
# 				df_right.insert(0, 'Sample_id_bis', column_to_move)
# 				unifier_builder1 = file
# 				unifier_builder1 = unifier_builder1.replace('.', '_')
# 				unifier_builder1 = unifier_builder1.split("_")
# 				unifier_builder1 = [unifier_builder1[i] for i in [0, 1, 3]]
# 				unifier_builder2 = "_".join(unifier_builder1)
# 				df_right.insert(0, 'Unifier_key', unifier_builder2)
# 				# lets make sure the load out going to the right is without nan
# 				df_right.dropna()
# 				# df_right=df_right.drop_duplicates()
# 				data_loadout_right_cn.append(df_right)
# 				# the dataframe unifierkey-sample_id-resp-features is tucked away for later
# 				# lets create the other dataframe
# 				df_left = pd.DataFrame()
# 				df_left.insert(0, 'Sample_id_bis', df_right['Sample_id_bis'])
# 				df_left = df_left.rename(columns={
# 					"Sample_id_bis": "Sample_id"})  # rename this column to drope asily the other one later after unification for looping on data
# 				df_left.insert(1, 'Cancer_type', unifier_builder1[0])
# 				df_left.insert(2, 'Treatment_id', unifier_builder1[1])
# 				df_left.insert(3, 'Profile_type', unifier_builder1[2])
# 				# lets make sure the load out going to the left is without nan
# 				df_left.dropna()
# 				data_loadout_left.append(df_left)
# 				# the dataframe sample_id-cancertype-treatment_id-profile_type is kept away
# 				data_loadouts_origin_files_for_cn.append(file)
# 			# the file used is kept in case we need to check concordances (at each index of the 3 data_loadout list, the elements are related)
# 			# another file is parsed to do the same two dataframes creations or the loop is ended
# 			# file1=file #uncomment to test the current file
# 			if file.endswith('_SNV.csv'):
# 				# print(os.path.join(root,file))
# 				print(file)
# 				# "df"+str(k)
# 				df_right = pd.read_csv(os.path.join(root, file))
# 				df_right = df_right.rename(
# 					columns={"BestResCategory": Resp_col_name})  # rename the cloumn for response into Resp_class
# 				df_right = df_right.rename(columns={"Model": "Sample_id_bis"})
# 				column_to_move = df_right['Sample_id_bis']
# 				df_right.drop(labels=['Sample_id_bis'], axis=1, inplace=True)
# 				df_right.insert(0, 'Sample_id_bis', column_to_move)
# 				unifier_builder1 = file
# 				unifier_builder1 = unifier_builder1.replace('.', '_')
# 				unifier_builder1 = unifier_builder1.split("_")
# 				unifier_builder1 = [unifier_builder1[i] for i in [0, 1, 3]]
# 				unifier_builder2 = "_".join(unifier_builder1)
# 				df_right.insert(0, 'Unifier_key', unifier_builder2)
# 				# lets make sure the load out going to the right is without nan
# 				df_right.dropna()
# 				# df_right=df_right.drop_duplicates()
# 				data_loadout_right_snv.append(df_right)
# 				# the dataframe unifierkey-sample_id-resp-features is tucked away for later
# 				# lets create the other dataframe
# 				df_left = pd.DataFrame()
# 				df_left.insert(0, 'Sample_id_bis', df_right['Sample_id_bis'])
# 				df_left = df_left.rename(columns={
# 					"Sample_id_bis": "Sample_id"})  # rename this column to drope asily the other one later after unification for looping on data
# 				df_left.insert(1, 'Cancer_type', unifier_builder1[0])
# 				df_left.insert(2, 'Treatment_id', unifier_builder1[1])
# 				df_left.insert(3, 'Profile_type', unifier_builder1[2])
# 				# lets make sure the load out going to the left is without nan
# 				df_left.dropna()
# 				data_loadout_left.append(df_left)
# 				# the dataframe sample_id-cancertype-treatment_id-profile_type is kept away
# 				data_loadouts_origin_files_for_snv.append(file)
# 			# the file used is kept in case we need to check concordances (at each index of the 3 data_loadout list, the elements are related)
# 			# another file is parsed to do the same two dataframes creations or the loop is ended
# 			# file1=file #uncomment to test the current file
# 			if file.endswith('_CNA.csv'):
# 				# print(os.path.join(root,file))
# 				print(file)
# 				# "df"+str(k)
# 				df_right = pd.read_csv(os.path.join(root, file))
# 				df_right = df_right.rename(
# 					columns={"BestResCategory": Resp_col_name})  # rename the cloumn for response into Resp_class
# 				df_right = df_right.rename(columns={"Model": "Sample_id_bis"})
# 				column_to_move = df_right['Sample_id_bis']
# 				df_right.drop(labels=['Sample_id_bis'], axis=1, inplace=True)
# 				df_right.insert(0, 'Sample_id_bis', column_to_move)
# 				unifier_builder1 = file
# 				unifier_builder1 = unifier_builder1.replace('.', '_')
# 				unifier_builder1 = unifier_builder1.split("_")
# 				unifier_builder1 = [unifier_builder1[i] for i in [0, 1, 3]]
# 				unifier_builder2 = "_".join(unifier_builder1)
# 				df_right.insert(0, 'Unifier_key', unifier_builder2)
# 				# lets make sure the load out going to the right is without nan
# 				df_right.dropna()
# 				# df_right=df_right.drop_duplicates()
# 				data_loadout_right_cna.append(df_right)
# 				# the dataframe unifierkey-sample_id-resp-features is tucked away for later
# 				# lets create the other dataframe
# 				df_left = pd.DataFrame()
# 				df_left.insert(0, 'Sample_id_bis', df_right['Sample_id_bis'])
# 				df_left = df_left.rename(columns={
# 					"Sample_id_bis": "Sample_id"})  # rename this column to drope asily the other one later after unification for looping on data
# 				df_left.insert(1, 'Cancer_type', unifier_builder1[0])
# 				df_left.insert(2, 'Treatment_id', unifier_builder1[1])
# 				df_left.insert(3, 'Profile_type', unifier_builder1[2])
# 				# lets make sure the load out going to the left is without nan
# 				df_left.dropna()
# 				data_loadout_left.append(df_left)
# 				# the dataframe sample_id-cancertype-treatment_id-profile_type is kept away
# 				data_loadouts_origin_files_for_cna.append(file)
# 			# the file used is kept in case we need to check concordances (at each index of the 3 data_loadout list, the elements are related)
# 			# another file is parsed to do the same two dataframes creations or the loop is ended
# 			# file1=file #uncomment to test the current file
# 	# #=========================================================================================================
# 	# =================== reduce the left load out to only keep unique sets in it
# 	# we can concatenate all of the dataframes contained in the list because we are sure that they have the same columns
# 	data_loadout_left_all = pd.concat(data_loadout_left)
# 	# lets drop the entries with nan
# 	# final_data_loadout_left.dropna()
# 	# we obtain a big dataframe of 114 rows x 4 cols
# 	data_loadout_left_all.reset_index(drop=True)  # reseting the index for a continuous count from zero at the first line until x with x the number of samples/biological problem -returned-
# 	# ===================in the end, the drugs names will be needed so let us also extract them
# 	df_treatments = pd.DataFrame()  # a df to stock our frame contains names of the treatments
# 	for root, directories, filenames in os.walk(data_path2):
# 		for file in filenames:
# 			if '_treatments' in file:
# 				# print(os.path.join(root,file))
# 				print("The following file is a source of the treatments names : ", file)
# 				df_treatments = pd.read_csv(os.path.join(root, file))
# 	df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].copy()
# 	df_treatments = df_treatments[["Treatment_ID", "Treatment_Details"]].astype(str)
# 	df_treatments["Treatment_ID"] = 'Treatment' + df_treatments["Treatment_ID"].astype(str)
# 	mydrugs = df_treatments.set_index('Treatment_ID').T.to_dict('list')  # a dict that will be called will filling the result file -returned-
# 	#temporary fix for having drugnames as list of one element ##!! correct it later by change exactly whats done earlier
# 	for key in list(mydrugs.keys()):
# 		value_of_key = mydrugs[key]
# 		new_value_of_key = value_of_key[0]
# 		mydrugs[key] = new_value_of_key
# 	################# Building elements to loop on for analysis :
# 	# loop 1 : on the cancertype. it needs a set of all the cancertype in the data_loadout_left
# 	ctypes = set(data_loadout_left_all["Cancer_type"])
# 	ctypes = {c for c in ctypes if c == c}
# 	ctypes = sorted(ctypes) # -returned-
# 	# ----------an testing idea is to only launch a test analysis with the ctypes of interest
# 	# ex: BRCA,COADREAD,LUAD,SCLC,SKCM
# 	# a_given_list_of_ctypes = ["BRCA","COADREAD","LUAD","SCLC","SKCM"]
# 	# ctypes = filter(lambda i: i in a_given_list_of_ctypes,ctypes)
# 	#  -------------
# 	# lets show the cancertypes we will run with
# 	print("Analysing with these cancer types : ", ctypes) # rectify as for printing elts in list
#
# 	# loop 2 : on the treatment_id. it needs a set of all the treatment_ids in the data_loadout_left
# 	drugs = set(data_loadout_left_all["Treatment_id"])
# 	drugs = {d for d in drugs if d == d}
# 	drugs = sorted(drugs) # -returned-
# 	# ----------an testing idea is to only launch a test analysis with the drugs of interest
# 	# ex: 'Treatment15', 'Treatment17', 'Treatment18'
# 	# a_given_list_of_drugs = ['Treatment15', 'Treatment17', 'Treatment18']
# 	# drugs = filter(lambda i: i in a_given_list_of_drugs,drugs)
# 	#  -------------
# 	# lets show the drugs we will run with
# 	print("Analysing with these drugs : ", drugs)
#
# 	# loop 3 : on the profile_type. it needs a set of all the profiles_types in the data_loadout_left
# 	mylabels = set(data_loadout_left_all["Profile_type"])
# 	mylabels = {p for p in mylabels if p == p}
# 	mylabels = sorted(mylabels) # -returned-
# 	# ----------an testing idea is to only launch a test analysis with the profiles of interest
# 	# ex: 'GEX'
# 	# a_given_list_of_mylabels = ['GEX']
# 	# mylabels = filter(lambda i: i in a_given_list_of_mylabels,mylabels)
# 	#  -------------
# 	# lets show the drugs we will run with
# 	print("Analysing with these profiles : ", mylabels)
# 	return data_loadout_left_all, data_loadout_right_gex, data_loadout_right_cn, data_loadout_right_snv, data_loadout_right_cna, data_loadouts_origin_files_for_gex, data_loadouts_origin_files_for_cn, data_loadouts_origin_files_for_snv, data_loadouts_origin_files_for_cna, mydrugs, ctypes, drugs, mylabels
#
# # ======

# oldest model for the mcc computation from the Nguyen's R code
# calculate.mcc = function(predict, observe){
#                 TP = length(which(predict == "Sen" & observe == "Sen"))    # cells predicted sensitive which are actually sensitive (cell sensitive is +ve) ## its good as like said earlier sen is the pos class
#                 TN = length(which(predict == "Res" & observe == "Res"))  # cells predicted resistant which are actually resistant (cell resistant is -ve)
#                 FP = length(which(predict == "Sen" & observe == "Res")) # cells predicted sensitive which are actually resistant (wrong prediction)
#                 FN = length(which(predict == "Res" & observe == "Sen"))       # cells predicted resistant which are actually sensitive (wrong prediction)
#                 dum1 = ((TP*TN)-(FP*FN))
#                 dum2 = sqrt(1.0*(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # 1.0 makes the multiplication float, thus avoiding integer overflow # remake formula to follow cycle of Pedro schematics
#                 if(dum2 == 0) dum2 = 1
#                 mcc = dum1/dum2
#               } ##def isolate



# -- model of a function to catch the metrics value from a classification_report():
# def classif_report(y_true,y_pred,sorted_list_of_the_classes):
# 	the_report = classification_report(y_true, y_pred, target_names=sorted_list_of_the_classes, output_dict=True)
# 	class0_precision = the_report[sorted_list_of_the_classes[0]]["precision"]
# 	class0_recall = the_report[sorted_list_of_the_classes[0]]["recall"]
# 	class0_f1score = the_report[sorted_list_of_the_classes[0]]["f1-score"]
# 	class0_support  = the_report[sorted_list_of_the_classes[0]]["support"]
# 	class1_precision = the_report[sorted_list_of_the_classes[1]]["precision"]
# 	class1_recall = the_report[sorted_list_of_the_classes[1]]["recall"]
# 	class1_f1score = the_report[sorted_list_of_the_classes[1]]["f1-score"]
# 	class1_support = the_report[sorted_list_of_the_classes[1]]["support"]
# 	return

#use this function for at least three classes classification ##!! modify first same as the cm_binary report
# def pd_ml_classif_report_on_cm_for_multi_classes(y_true,y_pred,sorted_list_of_the_classes):
# 	confusion_matrix = ConfusionMatrix(y_true.iloc[:,0].tolist(), y_pred.iloc[:,0].tolist())
# 	# confusion_matrix.stats() # for the full the full classif as a dict of 3 entries ; 3rd entries with key "class" is a dataframe with all metrics as rows and each col is a class
# 	# for the class0
# 	class0_acc = confusion_matrix.stats()["class"].loc["ACC: Accuracy"][sorted_list_of_the_classes[0]] #return[0]
# 	class0_MCC = confusion_matrix.stats()["class"].loc["MCC: Matthews correlation coefficient"][sorted_list_of_the_classes[0]] #return[1]
# 	class0_precision = confusion_matrix.stats()["class"].loc["TNR=SPC: (Specificity)"][sorted_list_of_the_classes[0]] #return[2]
# 	class0_recall = confusion_matrix.stats()["class"].loc["TPR: (Sensitivity, hit rate, recall)"][sorted_list_of_the_classes[0]] #return[3]
# 	class0_fpr = confusion_matrix.stats()["class"].loc["FPR: False-out"][sorted_list_of_the_classes[0]]  # return[4]
# 	class0_f1score = confusion_matrix.stats()["class"].loc["F1 score"][sorted_list_of_the_classes[0]] #return[5]
# 	class0_support = confusion_matrix.stats()["class"].loc["P: Condition positive"][sorted_list_of_the_classes[0]] #return[6]
# 	class0_num_tests_w_outcome_as_class_pos = confusion_matrix.stats()["class"].loc["Test outcome positive"][sorted_list_of_the_classes[0]] #return[7]
# 	# for the class1
# 	class1_acc = confusion_matrix.stats()["class"].loc["ACC: Accuracy"][sorted_list_of_the_classes[1]] #return[8]
# 	class1_MCC = confusion_matrix.stats()["class"].loc["MCC: Matthews correlation coefficient"][sorted_list_of_the_classes[1]] #return[9]
# 	class1_precision = confusion_matrix.stats()["class"].loc["TNR=SPC: (Specificity)"][sorted_list_of_the_classes[1]] #return[10]
# 	class1_recall = confusion_matrix.stats()["class"].loc["TPR: (Sensitivity, hit rate, recall)"][sorted_list_of_the_classes[1]] #return[11]
# 	class1_fpr = confusion_matrix.stats()["class"].loc["FPR: False-out"][sorted_list_of_the_classes[1]]  # return[12]
# 	class1_f1score = confusion_matrix.stats()["class"].loc["F1 score"][sorted_list_of_the_classes[1]] #return[13]
# 	class1_support = confusion_matrix.stats()["class"].loc["P: Condition positive"][sorted_list_of_the_classes[1]] #return[16]
# 	class1_num_tests_w_outcome_as_class_pos = confusion_matrix.stats()["class"].loc["Test outcome positive"][sorted_list_of_the_classes[1]] #return[17]
# 	return class0_acc,class0_MCC,class0_precision,class0_recall,class0_fpr,class0_f1score,class0_support,class0_num_tests_w_outcome_as_class_pos,class1_acc,class1_MCC,class1_precision,class1_recall,class1_fpr,class1_f1score,class1_support,class1_num_tests_w_outcome_as_class_pos

# deprecated version of geting a dict of mcc values (done over iterating on the models predictions cols (new version does that as a prelimiary)
# def calculate_mcc_singular(dict_of_mcc_values_by_mdl_to_update,frame_of_all_mdls_called_predictions_w_infos,index_start_space_of_cols_as_mdls,endplus1_space_of_cols_as_mdls, frame_of_observations_to_compare_with,resp_col,RespClasses): # space_of_cols_indexes_to_consider_as_mdls = 2: or if native = :
# 	# index_start_space_of_cols_as_mdls = 2 and endplus1_space_of_cols_as_mdls = len(endplus1_space_of_cols_as_mdls.columns)
# 	frame_of_all_mdls_called_predictions = frame_of_all_mdls_called_predictions_w_infos.iloc[:,index_start_space_of_cols_as_mdls:endplus1_space_of_cols_as_mdls]
# 	for one_mdl_called_predictions_as_a_column in list(frame_of_all_mdls_called_predictions):
# 		the_feat_col = pd.Categorical(frame_of_all_mdls_called_predictions[one_mdl_called_predictions_as_a_column], categories=[RespClasses[0], RespClasses[1]])
# 		the_resp_col = pd.Categorical(frame_of_observations_to_compare_with[resp_col], categories=[RespClasses[0], RespClasses[1]])
# 		contingency_table_filled = pd.crosstab(the_feat_col, the_resp_col,dropna=False)
# 		TP = contingency_table_filled.loc[RespClasses[1]][RespClasses[1]]
# 		TN = contingency_table_filled.loc[RespClasses[0]][RespClasses[0]]
# 		FP = contingency_table_filled.loc[RespClasses[1]][RespClasses[0]]
# 		FN = contingency_table_filled.loc[RespClasses[0]][RespClasses[1]]
# 		# calculation of a mcc
# 		mcc_numerator = (TP*TN)-(FP*FN)
# 		candidate_for_mcc_denominateur = sqrt((TP+FP)*(FP+TN)*(TN+FN)*(FN+TP))
# 		if candidate_for_mcc_denominateur == 0: # condittion of existence of mcc mathematically
# 			mcc_denominateur = 1
# 		else:
# 			mcc_denominateur = candidate_for_mcc_denominateur
# 		mcc = mcc_numerator / mcc_denominateur
# 		# dict_of_mcc_values_by_mdl_to_update has been created before all the workings here
# 		dict_of_mcc_values_by_mdl_to_update[one_mdl_called_predictions_as_a_column] = mcc # update the dict
# 	return dict_of_mcc_values_by_mdl_to_update
#=================old metrics functions

# # Step 2 : get true or False values in place of classes relatively to out class1
# y_true_as_array_of_TF = np.where(y_true_as_array == RespClasses[1], True, False)
# y_pred_as_array_of_TF = np.where(y_pred_as_array == RespClasses[1], True, False)
# # Step 3 : make lour entries as lists
# y_true_as_TF_list = y_true_as_array_of_TF.tolist()
# y_pred_as_TF_list = y_pred_as_array_of_TF.tolist()
# confusion_matrix = ConfusionMatrix(y_true.iloc[:,0].tolist(), y_pred.iloc[:,0].tolist()) # old line working w python 2.7

#===========old classif functions
# def classifier_introduction2_dflt0(classifier_version,mtry,ntrees,class1,prop_class2,class2,prop_class1):
# 	if classifier_version =="RF_dflt1":
# 		model = RandomForestClassifier()
# 	return model
# def classifier_introduction2(classifier_version,mtry,ntrees,class1,prop_class2,class2,prop_class1):
# 	if classifier_version =="RF_dflt1":
# 		model = RandomForestClassifier(max_features=mtry,n_estimators=ntrees,class_weight={class1:prop_class2,class2:prop_class1})
# 		# tune with this under :
# 		# - randomstate is none by dflt so it pick up the seed in np.random.seed()
# 		# - mtry is "max_features" so max_features = mtry
# 		# - n_estimators is ntrees
# 		# the class_weight is 2 classes and each one attached the other class proportion
# 		# the importance is by default because the gini criterion already by default
# 		## test the classweight auto
# 		# randomForest(ol2_train_x, ol2_train_y, mtry=mtry_mdl_allfts, importance=TRUE, ntree=ntree, classwt=c(Res=prop.sens_mdl_allfts, Sen=prop.res_mdl_allfts))
# 	return model

# # new addictions to the classifiers module
# from xgboost import XGBClassifier
# model = XGBClassifier()
# ol2_x_ar = np.array(ol2_train_x) # a multi_d array is needed
# ol2_y_ar = np.array(ol2_train_y.iloc[:,0].tolist()) # the array of train_y has to be 1_d ie created from a list made from the response column info
# model_fitted = model.fit(ol2_x_ar,ol2_y_ar) # training
# model_fitted
# # default params of xgboost
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#               n_jobs=1, nthread=None, objective='binary:logistic',
#               random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=True, subsample=1)
# # recommended parals from Stephan
# XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
# 														 colsample_bytree=0.8, silent=1,
# 														 nthread=num_cores, seed=aseed)
# num_cores = 38 or multiprocessing.cpu_count() - 1 ; aseed = arg
# #
# ol2_tx_ar = np.array(ol2_test_x)
# ol2_preds_proba = model_fitted.predict_proba(ol2_tx_ar) # prediction to get probs
# ol2_preds = model_fitted.predict(ol2_tx_ar)
# # this process works for RF also
# from sklearn.ensemble import RandomForestClassifier
# model2 = RandomForestClassifier()
# model2_fitted = model2.fit(ol2_x_ar,ol2_y_ar)
# /home/diouf/anaconda3/envs/classhd37_env2/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
#   "10 in version 0.20 to 100 in 0.22.", FutureWarning)
# model2_fitted
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=10,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
# ol2_predsRF = model2_fitted.predict(ol2_tx_ar)
# ol2_predsprobsRF = model2_fitted.predict_proba(ol2_tx_ar)
# #this process is for

# this process is for
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Classification algs 3 primary functions v3 (before DNN integration)
import numpy as np # for np.shape(x) giving n rows and k cols of a array as (n,k)
import pandas as pd # for pushing dataframes functions in order to report or keep track
import locale
# ---random forest implementation
from sklearn.ensemble import RandomForestClassifier
from math import sqrt # for mtry computation
from enginesV3.fs_engine import length_features_list # for numbers of fts computation in mtry
#--- XGBoost implementation
from xgboost import XGBClassifier
#--- GBM implementation
from sklearn.ensemble import GradientBoostingClassifier

#--- XGBoost installation ##!! deal with later
# from xgboost.sklearn import XGBClassifier
# import py_x
# import xgboost as xgb
#-- any other alg installation

###---------------------IMPORTS COMPLEMENTARY FOR REGRESSION

#====================================================================

# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#====================================================================

# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>functions for proper Classification algorithms
# # -------------------------------functions for XGBoost ##!! define later
# # def classifier_introduction(classifier_version,num_cores,aseed):
# # 	if classifier_version == "XGBoost_C_1" :
# # 		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700, colsample_bytree=0.8, silent=1, nthread=num_cores, seed=aseed)  # stock the method
# # 	return model
# #
# # def classifier_model_training(model,train_x,train_y):
# # 	model_fitted = model.fit(train_x, train_y)  # fit the method to the training data to get a classifier and predict with it later
# # 	return model_fitted
# #
# # def classifier_model_prediction(model_fitted,test_x):
# # 	model_prediction = model_fitted.predict(test_x)
# # 	return model_prediction
#
# #-------------------------------functions for RF
# def classifier_introduction(tag_alg,tag_alg_mark, trainframe_x, trainframe_y,aseed):
# 	actual_seed_classifier = np.random.seed(aseed)  # given to be able to use a random seed that stays the same acrross repetitons of the same experiement
# 	if tag_alg == "RF": # case of RF as alg
# 		if tag_alg_mark == "Mark1": #codename = RF_default (see Sl_algs_descriptor.txt for params value)
# 			model = RandomForestClassifier(random_state=actual_seed_classifier)
#
# 		elif tag_alg_mark == "Mark2":  ##!!!## new linh params # codename = RF_Linh_params
# 			# ntrees = 10 # testing
# 			ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=actual_seed_classifier)
# 		elif tag_alg_mark == "Mark3":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs100
# 			ntrees = 100  # testing
# 			# ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=actual_seed_classifier)
# 		elif tag_alg_mark == "Mark4":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs10
# 			ntrees = 10  # testing
# 			# ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=actual_seed_classifier)
# 		# tune with this under :
# 		# - randomstate is none by dflt so it pick up the seed in np.random.seed()
# 		# - mtry is "max_features" so max_features = mtry
# 		# - n_estimators is ntrees
# 		# the class_weight is 2 classes and each one attached the other class proportion
# 		# the importance is by default because the gini criterion already by default
# 		## test the classweight auto
# 		# randomForest(ol2_train_x, ol2_train_y, mtry=mtry_mdl_allfts, importance=TRUE, ntree=ntree, classwt=c(Res=prop.sens_mdl_allfts, Sen=prop.res_mdl_allfts))
# 	elif tag_alg == "XGB": # case of XGBoost as alg
# 		num_cores_heavy = 38 ##!! to check how it fits in with out multiprocessing
# 		num_cores_light = 2
# 		if tag_alg_mark == "Mark1":  # codename = XGB_default (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(seed=actual_seed_classifier)
# 		if tag_alg_mark == "Mark2":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
# 														 colsample_bytree=0.8, silent=1,
# 														 nthread=num_cores_heavy, seed=actual_seed_classifier)
# 		if tag_alg_mark == "Mark2nonthread":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
# 														 colsample_bytree=0.8, silent=1,
# 														 seed=actual_seed_classifier) # nthread=num_cores_heavy not used because it risks undeertaking the allocated cores to il1 xproc
# 	elif tag_alg == "GBM":  # case of GBM as alg
# 		if tag_alg_mark == "Mark1":  # codename = GBM_default (see Sl_algs_descriptor.txt for params value)
# 			model = GradientBoostingClassifier(random_state=actual_seed_classifier)
# 	return model
#
# def classifier_model_training(model, train_x, train_y):
# 	# the fit function in the case of most algs take 2 arrays in the next specifications otherwise error can be sent :
# 	# (the array_x is multi_dimensional ie shape(n_samples,#fts) and the array_y is 1_dimensional ie shape(n_samples,)
# 	train_x_as_array = np.array(train_x) # a multi_d array is needed
# 	train_y_as_array = np.array(train_y.iloc[:, 0].tolist())  # the array of train_y has to be 1_d ie created from a list made from the response column info
# 	model_fitted = model.fit(train_x_as_array, train_y_as_array)  # fit the method to the training data to get a classifier and predict with it later
# 	return model_fitted
#
# def classifier_model_prediction(model_fitted, test_x, prediction_type):
# 	# same as in training the function used here ask for an array_x that is multi_dimensional ie shape(n_samples,#fts)
# 	test_x_as_array = np.array(test_x)
# 	if prediction_type == "prob":
# 		model_prediction = model_fitted.predict_proba(test_x_as_array) # putting brackts aroung fix the need to reshpe error if working with series
# 	elif prediction_type == "pred":
# 		model_prediction = model_fitted.predict(test_x_as_array)  # putting brackts aroung fix the need to reshpe error
# 	return model_prediction
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#>>>>>>>>>>>>>>>>>>>>>>DNN models separated codes
#####intro
# elif tag_alg == "DNN":  # case of DNN as alg
	# 	if tag_alg_mark == "Mark1":  # codename = DNN_default (see Sl_algs_descriptor.txt for params value)
	# 		print("in_dev")
	# 	if tag_alg_mark == "Mark2":  # codename = DNN_recommended_zhou2019_v1 (see Sl_algs_descriptor.txt for params value)
	# 		# create larger model
	# 		model = Sequential()
	# 		# # optional to use dropout on the input layer
	# 		# model.add(Dropout(0.6, input_shape=(trainframe_x.shape[1],)))
	# 		# model.add(Dense(512, kernel_initializer='normal', activation='relu'))
	# 		model.add(Dense(512, input_dim=trainframe_x.shape[1], kernel_initializer='normal',kernel_regularizer=l2(1e-05)))
	# 		model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
	# 		# epsilon = 1e-05 to satisfy Theano demands for portability, mode = 0 feature-wise normalization,
	# 		# axis is left to default instead of 1 for samples (for better management )
	# 		# by default if weights=None:
	# 		# beta_init = 'zero', gamma_init = 'one'
	# 		# beta_regularizer=None,gamma_regularizer=None
	# 		# beta_constraint = None, gamma_constraint = None
	# 		# moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones'
	# 		model.add(Activation('relu'))
	# 		model.add(Dropout(0.6))
	# 		model.add(Dense(256, kernel_initializer='normal', kernel_regularizer=l2(1e-05))) # replaced by a line : activation='relu'
	# 		model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
	# 		model.add(Activation('relu'))
	# 		model.add(Dropout(0.6))
	# 		model.add(Dense(64, kernel_initializer='normal', kernel_regularizer=l2(1e-05)))
	# 		model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
	# 		model.add(Activation('relu'))
	# 		model.add(Dropout(0.6))
	# 		model.add(Dense(1, kernel_initializer='normal'))
	# 		model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
	# 		model.add(Activation('sigmoid'))
	# 		# Compile model
	# 		adagrad_mark1_as_opt = optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0)
	# 		model.compile(loss='binary_crossentropy', optimizer=adagrad_mark1_as_opt)
# ####training
# 	# np.random.seed(aseed)
# 	# if tag_alg == "DNN": # requires some hyperparameters to be passed into the fit function
# 	# 	if tag_alg_mark == "Mark1":  # codename = DNN_default (see Sl_algs_descriptor.txt for params value)
# 	# 		print("training_in_dev")
# 	# 	elif tag_alg_mark == "Mark2": # codename = DNN_recommended_zhou2019_v1 (see Sl_algs_descriptor.txt for params value)
# 	# 		# following type of profile, scale values of features (if not categrical values) or not
# 	# 		if featuretype not in classif_list_cat_fts:
# 	# 			train_x_as_array = StandardScaler().fit_transform(trainframe_x)
# 	# 		else:
# 	# 			train_x_as_array = np.array(trainframe_x)
# 	# 		# for the responses classes, the values have to be encoded (using here an encoding done just after data_management)
# 	# 		# binary_classes_le.classes_ gives an array of the classes
# 	# 		train_y_as_array = np.array(binary_classes_le.transform(trainframe_y.iloc[:, 0].tolist())) # get array from frames of thruths with values encoded
# 	# 		# make the training
# 	# 		model.fit(train_x_as_array, train_y_as_array, epochs=100, batch_size=100, verbose=0) # giving this to a variable only stock a history object
# 	# 		# do this to store the model fitted
# 	# 		model_fitted = model
# 	# else: # a normal fit function used
# #<<<<<<<<<<<<<<<<<<<
#
# ####---->TESTING THE METRICS INNER WORKING (PANDAS_ML)
# #####fine il1_job
# def il1_job(respective_fold_id):  # respective_fold_id = 2 for testing
# 	print(bcolors.OKBLUE + " >>> - Task running IL1's fold", respective_fold_id, "has started being carried out..." + bcolors.ENDC)
# 	print(bcolors.OKBLUE + " >>> - Ranking the features in the fold to get the FS for every MCs to estimate next" + bcolors.ENDC)
# 	# >>>>>>>>>>>>PRE-BUILDING THE SETS NEEDED
# 	# ---1-making of trainframe and valframe
# 	il1_train_data = ol2_train_data.drop(respective_fold_id)  # Set the training set of 36 (by dropping one row/sample)
# 	il1_test_data = ol2_train_data.loc[[respective_fold_id], :]  # Set the validation set of 1 (by keeping only that previously dropped sample)
# 	# trainframe_y & valframe_y
# 	il1_train_y = il1_train_data.loc[:, [Resp_col_name]]  # train_y (needed to train)
# 	# il1_test_y = il1_test_data.loc[:, [Resp_col_name]]  # val_y (not needed because prediction estimation uses the resp_col_nme of the dataset directly restricted to the indexes implicated)
# 	# trainframe_x & valframe_x
# 	il1_train_x = il1_train_data[list(il1_train_data)[5:]]  # train_x (needed to train) (## should not transmit to self but to new name)
# 	il1_test_x = il1_test_data[list(il1_test_data)[5:]]  # val_x (needed to predict)
# 	# <<<<<<<<<<<<<<END OF PRE-BUILDING THE SETS NEEDED
# 	if classif_omc_search_type == "OMC":
# 		# ~~~~~~~~~~~~~~~~~~~~~ optional part (depending on omc search chosen)
# 		# >>>>>>>>>>>>>>START OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
# 		# ---2-estimate the fts with two-sided fisher exact test p-values or two sided unpaired t-test p-values
# 		# we are about to train a model so lets re set the seed at its rightful place in case a function from a module changed it
# 		np.random.seed(aseed)
# 		# ---introduce the model
# 		rf1 = classifier_introduction(tag_alg, tag_alg_mark, il1_train_x, il1_train_y, aseed)  ## class weighting with classwt #to be check if not inversed
# 		# ---train the model on train_x,train_y
# 		rf1_fitted = classifier_model_training(rf1, il1_train_x, il1_train_y)
# 		# -- mk test for each fold
# 		il1_pred_2probs_w_allfts = classifier_model_prediction(rf1_fitted, il1_test_x, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 		il1_pred_call_w_allfts = []
# 		il1_pred_call_w_allfts = prediction_calling(il1_pred_2probs_w_allfts, il1_pred_call_w_allfts, classif_thres, rf1_fitted.classes_, separate_value_for_short_random_actions)
# 	##maybe add this!!! ---adding raw predictions to a keep tracker of predictions in folds
# 	##maybe add this!!! ---adding the call to a collector for stats or metrics
# 	# <<<<<<<<<<<<<<END OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
# 	# ~~~~~~~~~~~~~~~~~~~~~ end of optional part (depending on omc search chosen)
# 	# >>>>>>>>>>>>>>ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
# 	# ---process of limiting training data to only fts who shows variation (at least one variation in a sample)
# 	# il1_train_x = eliminate_non_variable_fts(il1_train_x) # no need bbcuz we manage the exception
# 	# ---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (a vector of fts's p values in the order of the fts)
# 	il1_fts_ranking = ranker_by_pval_v2(il1_train_x, il1_train_y, featuretype, Resp_col_name)
# 	# <<<<<<<<<<<END OF ESTIMATING FTS CONTRIBUTION FOR FTS-RANKING
# 	# >>>>>>>>>>>MCs PREDICTIONS MAKING
# 	# ---3-training a model for each MC and test it to compare them and take the best OMC
# 	il2_col_il2_pred_call_mc = []  # a collector of prediction called at each loop on complxities
# 	# mc = 2 # trial
# 	# print "- Estimating the MCs"
# 	print(bcolors.FAIL + " >>>> START OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_id, "training data. Going for", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
# 	for mc in list(range(2, (max_complexity_of_tested_MCs + 1))):  # inner loop 2 # loop on the MCs
# 		print(bcolors.FAIL + " >>>> - In Fold", respective_fold_id, ", MC of", mc, "features has started being made and estimated..." + bcolors.ENDC)
# 		topfeats = il1_fts_ranking[1][0:mc]
# 		## isolate as ranking function for p_value
# 		# rank it, keep this rank from original, get the names, get the top MC or those names) ## try to gives the rank at end of estimating fts to select in it directly here
# 		# set up a training of the built with elected fts MC model
# 		il2_train_mc_x = il1_train_x.loc[:, topfeats]  # train_x restricted to MC fts
# 		il2_test_mc_x = il1_test_x.loc[:, topfeats]  # test_x restricted to MC fts
# 		np.random.seed(aseed)
# 		# ---introduce the model
# 		rf2 = classifier_introduction(tag_alg, tag_alg_mark, il2_train_mc_x, il1_train_y, aseed)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
# 		# ---train the model on train_x,train_y
# 		rf2_fitted = classifier_model_training(rf2, il2_train_mc_x, il1_train_y)
# 		# -- mk test for each fold
# 		il2_pred_2probs_w_mc = classifier_model_prediction(rf2_fitted, il2_test_mc_x, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 		il2_pred_call_mc = []
# 		il2_pred_call_mc = prediction_calling(il2_pred_2probs_w_mc, il2_pred_call_mc, classif_thres, rf2_fitted.classes_, separate_value_for_short_random_actions)
# 		il2_col_il2_pred_call_mc.append(il2_pred_call_mc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
# 	# ------end of inner loop2
# 	print(bcolors.FAIL + " <<<< END OF INNER LOOP 2 (IL2): Making MCs and their predictions on IL1 present fold", respective_fold_id, "training data. Gone through", len(list(range(2, (max_complexity_of_tested_MCs + 1)))), "MCs" + bcolors.ENDC)
# 	# make a df that shows all found for this fold of il1 including all in il2 mcs
# 	# OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame()  # the part of the OMC quest wide col for a fold
# 	the_fold_column_content = [respective_fold_id] * len(il1_test_data.index.tolist())  # col fold
# 	the_il1_test_data_indexes_column_content = il1_test_data.index.tolist()  # col indexes in il1_test_data
# 	# il1_pred_call_w_allfts # is the [preds called for allfts on il1 to compare with the mcs models prediction] column content
# 	# il2_pred_call_mc # is the [each mc preds called] column content and a lot are in store into il2_col_il2_pred_call_mc so use a * in front of it in the zip to extract its content at the same level
# 	if classif_omc_search_type == "OMC":
# 		list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, il1_pred_call_w_allfts, *il2_col_il2_pred_call_mc))  # old zipping coming with including allfts model in estimations of mcc for omc mdls # zip col1,col2,col3, and extrcted col4
# 	else:
# 		list_of_all_cols_for_a_fold_zipped = list(zip(the_fold_column_content, the_il1_test_data_indexes_column_content, *il2_col_il2_pred_call_mc))  # zip col1,col2, and extrcted col4
#
# 	OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold = pd.DataFrame(list_of_all_cols_for_a_fold_zipped)  # init the part of the OMC quest wide col for a fold # make a df of it
# 	tuple_result = (tuple([respective_fold_id, OMC_quest_allfts_vs_MCs_wide_col_for_each_il1_fold]))  # keep a tuple of the fold id and the mini df produced (later this is used to put in the right order the results and make a full df of them)
# 	print(bcolors.OKBLUE + " <<<  - Task running Fold", respective_fold_id, ",among", len_il1_folds, "folds, is done. Carried out by " + current_process().name + bcolors.ENDC)
# 	return tuple_result
# ####launch this
# for j in ol2_folds:  # outer loop 2 (each fold for LOO on the training set
# 	print(bcolors.WARNING + " >> - OL2's fold", j, "has started being used..." + bcolors.ENDC)
# 	print(bcolors.WARNING + " >> -- Training and prediction of an Allfts model" + bcolors.ENDC)
# 	# mk training and testing data for each fold of LOOCV on all the data (use j to def sets and get the predictions )
#
# 	# ============================BUILD TRAINING SET AND TEST SET OF THE FOLD============================================================
# 	ol2_train_data = dframe.drop(j)  # Set the training set (for a row j of data, select all rows of data without the row of index 1 (here the frame start index at 1)
# 	ol2_test_data = dframe.loc[[j], :]  # Set the validation set (for a row j of data, select only the row j)
# 	# ....train_y, test_y of trainset_fold
# 	ol2_train_y = ol2_train_data.loc[:, [Resp_col_name]]  # keep the training set response column values (37 values) (for the row j of data analysed, the 1st value)
# 	ol2_test_y = ol2_test_data.loc[:, [Resp_col_name]]  # same (1 values)
# 	# .....train_x, test_x of trainset_fold
# 	ol2_train_x = ol2_train_data[list(ol2_train_data)[5:]]  # keep only the features (index 1 to ncol) #here are dropped the model and the response #odd code tho because _c(i,j) drops cols i to j
# 	ol2_test_x = ol2_test_data[list(ol2_test_data)[5:]]
#
# 	# ============================================== ALL FTS MODEL : ONE OF THE MODEL COMPARED (TRAINING AND STAASHING OF PREDICTION==============================
# 	# ---training the allfts model
# 	# ---introduce the model
# 	mdl_allfts = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x, ol2_train_y, aseed)  ## class weighting with classwt #to be check if not inversed
# 	# ---train the model on train_x,train_y
# 	mdl_allfts_fitted = classifier_model_training(mdl_allfts, ol2_train_x, ol2_train_y)
# 	# -- mk test for each fold
# 	ol2_pred_2probs_mdl_allfts = classifier_model_prediction(mdl_allfts_fitted, ol2_test_x, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 	ol2_pred_call_mdl_allfts = []  # last_stop_end_we
# 	ol2_pred_call_mdl_allfts = prediction_calling(ol2_pred_2probs_mdl_allfts, ol2_pred_call_mdl_allfts, classif_thres, mdl_allfts_fitted.classes_, separate_value_for_short_random_actions)
# 	# ---adding raw predictions to a keep tracker of predictions in folds
# 	print_pred_all_col = raw_predictions_pusher(ol2_pred_2probs_mdl_allfts, j, [j], mdl_allfts_fitted, print_pred_all_col, ol2_pred_call_mdl_allfts, aseed)
# 	# ---adding the call to a collector for stats or metrics
# 	ol2_col_pred_call_1by_seed_w_allfts = called_predictions_pusher(ol2_pred_call_mdl_allfts, [j], ol2_col_pred_call_1by_seed_w_allfts)
# 	# ----------------------end of all_model predictions making and stashing
#
# 	# ============================================== OMC TUNING 1/2 : OMC MODEL RESEARCH OVER MULTIPLES FOLDS =======================================
# 	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of tested portion
# 	### IL1 will be multiproccessed, each fold a process using Process
# 	# Step 1 : initialise the needed parameters
# 	OMC_quest_allfts_vs_MCs_wide_col = []  # intialise a big list to put in it all the allfts anc omc predictions called to compute mcc on it and compare the (MCs,allfts) models
# 	# >>>>>>>>>>>MAKE # OF FOLDS FOR CV
# 	il1_folds = ol2_train_data.index  # create the folds for the LOO for the omc model
# 	len_il1_folds = len(il1_folds)
# 	# l = 1 # testing
# 	# il1_folds = [1,2,3] #testing
# 	# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# 	print(bcolors.OKBLUE + " >>> START OF INNER LOOP 1 (IL1): LOOCV on OL2 present fold", j, "training data. Going for", len_il1_folds, "folds" + bcolors.ENDC)
# 	print(bcolors.OKBLUE + " >>> With each fold, training and prediction with many MCs and thus determining later OMC on one IL1 loop" + bcolors.ENDC)
# 	# for l in il1_folds: # this was the start of  # inner loop 1 #parallelise 37 jobs on each of on 37 folds/samples # the packages necessary are given # the results are combined in row after row fashion to form a matrix #and that in order of the samples
# 	_MP_STOP_SIGNAL = False  # multi processing stop signal
# 	# Step 2 : describe the function that each process will launch when it will accept a job in a queue
# 	bench_of_results = [] # for testing # a collector for the result of each fold carried out to sort later and make an ordered df (a shared list is used in worker function of xproc)
#
# 	# fixate a seed for all this xproc part so that variactions dont happen when the env is copied for each process
# 	np.random.seed(separate_value_for_short_random_actions)
# 	for fold in il1_folds:
# 		a_tuple_result = il1_job(fold)
# 		bench_of_results.append(a_tuple_result)
# 	bench_of_results_list_of_tuples_sorted_on_fold_id = sorted(bench_of_results, key=itemgetter(0))
# 	for a_tuple in bench_of_results_list_of_tuples_sorted_on_fold_id:
# 		OMC_quest_allfts_vs_MCs_wide_col.append(a_tuple[1])
# 	# ----&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# 	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of tested portion
# 	# end of inner loop 1 (official end)
# 	print(bcolors.OKBLUE + " <<< END OF INNER LOOP 1 (IL1): LOOCV on OL2 present fold", j, ". IL1 counted", len_il1_folds, "folds" + bcolors.ENDC)
# 	# lets concatenate all dfs, one from each fold of inner loop 1 into one (parse on it later, restrict to fold, get mcc ie many mcc for many folds of a models s med.mcc)
# 	OMC_quest_allfts_vs_MCs_wide = pd.concat(OMC_quest_allfts_vs_MCs_wide_col)
# 	# lets rename the columns properly to what they correspond
# 	if classif_omc_search_type == "OMC":
# 		edited_colnames = ["Fold"] + ["Sample_index"] + [num_all_features] + list(range(2, (max_complexity_of_tested_MCs + 1)))  # old naming cols coming with including allfts model in estimations of mcc for omc mdls
# 	else:
# 		edited_colnames = ["Fold"] + ["Sample_index"] + list(range(2, (max_complexity_of_tested_MCs + 1)))
# 	OMC_quest_allfts_vs_MCs_wide.columns = edited_colnames
# 	# <<<<<<<<<<<<END OF PREDICTIONS MAKING
# 	# >>>>>>>>>>>ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
# 	print(bcolors.WARNING + " << - OL2 is still at fold", j, ": Choosing OMC among MCs of IL1" + bcolors.ENDC)
# 	# ------------Median MCCs calculations as the criteria to elect omc on
# 	dict_of_mcc_values_by_mdl_to_update = {}
# 	# to calculate MCCs, restrict the OMC_quest_wide to the mdl_column for the corresponding preds...
# 	frame_of_all_mdls_called_predictions = OMC_quest_allfts_vs_MCs_wide.iloc[:, 2:len(OMC_quest_allfts_vs_MCs_wide.columns)]
# 	# index_start_space_of_cols_as_mdls = 2 and endplus1_space_of_cols_as_mdls = len(endplus1_space_of_cols_as_mdls.columns)
# 	# space_of_cols_indexes_to_consider_as_mdls = 2: or if native = :
# 	# ...and mock up a same length response column from ol2_train_y with each row of col of indexes having the corresponding response
# 	frame_of_all_mdls_implicated_samples_observations = ol2_train_y.loc[OMC_quest_allfts_vs_MCs_wide["Sample_index"].tolist(), :]  ##!!! last_stop_check if the 2 last frames works after the columns renaming
# 	dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, frame_of_all_mdls_called_predictions, frame_of_all_mdls_implicated_samples_observations, Resp_col_name, RespClasses)
# 	# ~~~~~~~~preceed with this restriction to use singular mcc calculation
# 	# for l in il1_folds:
# 	# 	restricted_to_samples_in_test = restriction_of_MCCs_wide(OMC_quest_allfts_vs_MCs_wide, l, il1_test_data.index.tolist())
# 	#   dict_of_mcc_values_by_mdl_to_update = calculate_mcc_w_storing(dict_of_mcc_values_by_mdl_to_update, restricted_to_samples_in_test, 2, len(restricted_to_samples_in_test.columns), il1_test_y, Resp_col_name, RespClasses)
# 	# ~~~~~~~~
# 	# ---choosing the best mc....
# 	omc_il1, omc_il1_mcc = OMC_founder_in_dict_MCs_MCCs(dict_of_mcc_values_by_mdl_to_update)
# 	# and...keeping it away...
# 	ol2_col_of_omc_il1.append(omc_il1)  # 2 lists are used to also take the mcc with us alongside the omc until the end
# 	ol2_col_of_omc_il1_mcc.append(omc_il1_mcc)
# 	# ...contains the best_fts_mc at each fold of LOO of the outer loop2 ie 38 values
# 	# <<<<<<<<<<<<<<<END OF ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
# 	# >>>>>>>>>>>>>START OF OMC PREDICTIONS MAKING
# 	print(bcolors.WARNING + " << - OL2 is still at fold", j, ": Getting for OMC of IL1 its predictions on OL2 training data" + bcolors.ENDC)
# 	# ---process of estimating fts for fts ranking
# 	# ---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (z dict of feat: p_val )
# 	ol2_fts_ranking = ranker_by_pval_v2(ol2_train_x, ol2_train_y, featuretype, Resp_col_name)
# 	# ---get on ol2_data the prediction of the omc
# 	if classif_omc_search_type == "OMC":
# 		model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + ", " + str(num_all_features) + "]"
# 		if omc_il1 == num_all_features:  # ------ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS # basically its if nfts = all_var gives pred_prob of all_model otherwise use the function for ranking fts with p_values
# 			# topfeats_in_ol2_fts_ranking = ol2_train_x.columns.tolist() #line not needed for the prediction here because its the all fts model so just taking the whole ol2trainset
# 			# ---introduce the model
# 			mdl_allfts = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x, ol2_train_y, aseed)  ## class weighting with classwt #to be check if not inversed
# 			# ---train the model on train_x,train_y
# 			mdl_allfts_fitted = classifier_model_training(mdl_allfts, ol2_train_x, ol2_train_y)
# 			# -- mk test for each fold
# 			ol2_pred_2probs_w_omc = classifier_model_prediction(mdl_allfts_fitted, ol2_test_x, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 			ol2_pred_call_w_omc = []  # last_stop_end_we
# 			ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, mdl_allfts_fitted.classes_, separate_value_for_short_random_actions)
# 			# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
# 			# ---adding raw predictions to a keep tracker of predictions in folds
# 			print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, j, [j], mdl_allfts_fitted, print_pred_col, ol2_pred_call_w_omc, aseed)
# 			# ---adding the call to a collector for stats or metrics
# 			ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, [j], ol2_col_pred_call_1by_seed_w_omc)
# 		else:  # ------A MODEL SMALLER IN SIZE THAN ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS
# 			# restrict train and test to fts of omc
# 			topfeats_in_ol2_fts_ranking = ol2_fts_ranking[1][0:omc_il1]
# 			ol2_train_x_w_omc_il1 = ol2_train_x.loc[:, topfeats_in_ol2_fts_ranking]  # train_x restricted to MC fts
# 			ol2_test_x_w_omc_il1 = ol2_test_x.loc[:, topfeats_in_ol2_fts_ranking]  # test_x restricted to MC fts
# 			np.random.seed(aseed)
# 			# ---introduce the model
# 			rf3 = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, aseed)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
# 			# ---train the model on train_x,train_y
# 			rf3_fitted = classifier_model_training(rf3, ol2_train_x_w_omc_il1, ol2_train_y)
# 			# -- mk test for each fold
# 			ol2_pred_2probs_w_omc = classifier_model_prediction(rf3_fitted, ol2_test_x_w_omc_il1, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 			ol2_pred_call_w_omc = []
# 			ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, rf3_fitted.classes_, separate_value_for_short_random_actions)
# 			# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
# 			# ---adding raw predictions to a keep tracker of predictions in folds
# 			print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, j, [j], rf3_fitted, print_pred_col, ol2_pred_call_w_omc, aseed)
# 			# ---adding the call to a collector for stats or metrics
# 			ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, [j], ol2_col_pred_call_1by_seed_w_omc)
# 	##*******deal with this line later
# 	# il2_col_il2_pred_call_mc.append(ol2_pred_call_w_omc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
# 	# *******deal with this line later
# 	else:  # ------A MODEL SMALLER IN SIZE THAN ALL FEATURES MODEL AS OMC, LETS GETS ITS PREDICTIONS
# 		model_complexities_explored_report = "[" + str(min_complexity_of_tested_MCs) + ":" + str(max_complexity_of_tested_MCs) + "]"
# 		# restrict train and test to fts of omc
# 		topfeats_in_ol2_fts_ranking = ol2_fts_ranking[1][0:omc_il1]
# 		ol2_train_x_w_omc_il1 = ol2_train_x.loc[:, topfeats_in_ol2_fts_ranking]  # train_x restricted to MC fts
# 		ol2_test_x_w_omc_il1 = ol2_test_x.loc[:, topfeats_in_ol2_fts_ranking]  # test_x restricted to MC fts
# 		np.random.seed(aseed)
# 		# ---introduce the model
# 		rf3 = classifier_introduction(tag_alg, tag_alg_mark, ol2_train_x_w_omc_il1, ol2_train_y, aseed)  # train_y does not need to be restricted ## class weighting with classwt #to be check if not inversed
# 		# ---train the model on train_x,train_y
# 		rf3_fitted = classifier_model_training(rf3, ol2_train_x_w_omc_il1, ol2_train_y)
# 		# -- mk test for each fold
# 		ol2_pred_2probs_w_omc = classifier_model_prediction(rf3_fitted, ol2_test_x_w_omc_il1, "prob")  # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
# 		ol2_pred_call_w_omc = []
# 		ol2_pred_call_w_omc = prediction_calling(ol2_pred_2probs_w_omc, ol2_pred_call_w_omc, classif_thres, rf3_fitted.classes_, separate_value_for_short_random_actions)
# 		# ~~~adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
# 		# ---adding raw predictions to a keep tracker of predictions in folds
# 		print_pred_col = raw_predictions_pusher(ol2_pred_2probs_w_omc, j, [j], rf3_fitted, print_pred_col, ol2_pred_call_w_omc, aseed)
# 		# ---adding the call to a collector for stats or metrics
# 		ol2_col_pred_call_1by_seed_w_omc = called_predictions_pusher(ol2_pred_call_w_omc, [j], ol2_col_pred_call_1by_seed_w_omc)
# 		##*******deal with this line later
# 		# il2_col_il2_pred_call_mc.append(ol2_pred_call_w_omc)  # stash the prediction # in a list of x preds (for x samples in test) being added into list #
# 		# *******deal with this line later
# 	# end of judging the feature ranking of the omc in all folds of the training data
# 	# >>>>>>>>>> END OF OMC PREDICTIONS MAKING
# 	# end of outer loop 2 (on each fold for LOOCV on all training set, predictions are reaped)
# 	print(bcolors.WARNING + " << END OF OUTER LOOP 2 (OL2): LOOCV on all data ie on", len(ol2_folds), "folds" + bcolors.ENDC)
# 	# >>>>>>>>>>>>METRICS computations FOR EACH SEED
# 	# ~~~~~~>(FOR THE OMC MODEL)
# 	print(bcolors.OKGREEN + " > - # OL1 is still at seed", aseed, ": Metrics of the OMC model on this one seed alone..." + bcolors.ENDC)
# 	# 1: Create the requirements...
# 	# ~~~~~(finishing touches for outer loop 1 collectors)-A
# 	# lets add a median of all the OMCs from OL2, one from each fold on the training set (getting one OMC estimation by seed) # P1 (P1U)
# 	med_ol2_col_omc = np.median(ol2_col_of_omc_il1)
# 	ol1_col_med_ol2_col_omc.append(med_ol2_col_omc)  # P1U # a list that collect the # of fts found in at least 50% at each seed analysis
# 	# ~~~~~(finishing touches for outer loop 1 collectors)-B-omc model
# 	#### lets make for the 2 models (omc and allfts) the df of the preds called collected to ready it for their mcc calculations
# 	df_from_ol2_col_pred_call_1by_seed_w_omc = pd.DataFrame(ol2_col_pred_call_1by_seed_w_omc)
# 	# lets rename the columns properly to what they correspond
# 	edited_colnames_for_ol2_col_pred_call_1by_seed_w_omc = ["Sample_index"] + ["OMC_called_preds"]
# 	df_from_ol2_col_pred_call_1by_seed_w_omc.columns = edited_colnames_for_ol2_col_pred_call_1by_seed_w_omc
# 	# on the full dataset LOO folds, to calculate the MCC and others metrics for the OMC, make a df of the mdl corresponding preds and the responses...
# 	frame_of_omc_mdl_called_predictions_on_all_data = df_from_ol2_col_pred_call_1by_seed_w_omc.iloc[:, [1]]
# 	frame_of_omc_mdl_implicated_samples_observations_on_all_data = dataBin.loc[df_from_ol2_col_pred_call_1by_seed_w_omc["Sample_index"].tolist(), :]
# 	# ~~~~~(finishing touches for outer loop 1 collectors)-C-omc model
# 	# lets report the predictions with each seed in a file from the temp dataframe ## no need to write, juste make df of it # used for the auc computations and roc curve plot
# 	df_from_print_pred_col = pd.concat(print_pred_col)
# 	all_seeds_col_of_df_from_print_pred_col.append(df_from_print_pred_col)  # stock it from all seeds version creation
# 	# on the full dataset LOO folds, to calculate the AUC and make the roc curve, make 2 arrays of the mdl corresponding probs of preds and the responses...
# 	array_of_omc_mdl_predictions_2probs_on_all_data = np.array(df_from_print_pred_col.loc[:, [df_from_print_pred_col.columns[4]]])  # df_from_print_pred_col.columns[4] is same as the older RespClasses[1]
# 	array_of_omc_mdl_implicated_samples_observations_considering_predictions_2probs_on_all_data = np.array(dataBin.loc[df_from_print_pred_col[df_from_print_pred_col.columns[2]].tolist(), :])  # df_from_print_pred_col.columns[2] is the same as the older "Test_sample_index"
# 	# ~~~~~~~~uncomment this for indivudal computation of the mcc
# 	# mcc_w_omc_as_dict_of_one_entry = {}
# 	# mcc_w_omc_as_dict_of_one_entry = calculate_mcc_wo_storing(frame_of_omc_mdl_called_predictions_on_all_data, frame_of_omc_mdl_implicated_samples_observations_on_all_data, Resp_col_name, RespClasses).values()[0]
# 	# ~~~~~~~~
# 	# 2 : ...calculate the 8 metrics...
# 	metrics_mdl_omc = pd_ml_classif_report_on_cm_binary(frame_of_omc_mdl_implicated_samples_observations_on_all_data, frame_of_omc_mdl_called_predictions_on_all_data, RespClasses)
# 	acc_w_omc = metrics_mdl_omc[0]


# #======================trials for dnn code=========
# ###---------------------IMPORTS FOR CLASSIFICATION
# import numpy as np # for np.shape(x) giving n rows and k cols of a array as (n,k)
# import pandas as pd # for pushing dataframes functions in order to report or keep track
# import locale
# #----------a random seed initialisation to fixate some randomness due to intialisation of librairies or ibjects
# np.random.seed(0) # first intention was to minimize variations in neural networks
# from tensorflow import set_random_seed # as TensorFlow backend is used by Keras DNNs and TensorFlow has its own random number generator, that must also be seeded
# set_random_seed(2)
# # ---random forest installation
# from sklearn.ensemble import RandomForestClassifier
# from math import sqrt # for mtry computation
# from enginesV3.fs_engine import length_features_list # for numbers of fts computation in mtry
# #--- XGBoost installation
# from xgboost import XGBClassifier
# #--- GBM installation
# from sklearn.ensemble import GradientBoostingClassifier
# #----DNN implementation
# from keras import optimizers # for the optimizers
# from keras.regularizers import l2 # for the weight decay as L2 regularization
# from keras.layers import Dropout # from the dropout regularization essay
# from keras.models import Sequential # for the model structure
# from keras.layers import Dense # for each input layer mostly hidden layers
# from keras.layers import Activation # for activation functions
# from sklearn.preprocessing import StandardScaler # a scaling function
# from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
# from keras.layers.normalization import BatchNormalization # for batch normlisation betwee X.w and batchnormalised (X.w) + b
# # in case
# # from keras.constraints import maxnorm
# # from keras.optimizers import SGD
# # from keras.wrappers.scikit_learn import KerasClassifier
# # from sklearn.model_selection import cross_val_score
# # from sklearn.model_selection import StratifiedKFold
# # from sklearn.pipeline import Pipeline
#
#
# #-------------------------------functions for RF
# def classifier_introduction(tag_alg,tag_alg_mark, trainframe_x, trainframe_y,aseed):
# 	np.random.seed(aseed)  # initiated again as a mesure of security so that even if random state not given, the none value make it catch this as a random seed that stays the same acrross repetitons of the same experiement
# 	if tag_alg == "RF": # case of RF as alg
# 		if tag_alg_mark == "Mark1": #codename = RF_default (see Sl_algs_descriptor.txt for params value)
# 			model = RandomForestClassifier(random_state=aseed)
#
# 		elif tag_alg_mark == "Mark2":  ##!!!## new linh params # codename = RF_Linh_params
# 			# ntrees = 10 # testing
# 			ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
# 		elif tag_alg_mark == "Mark3":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs100
# 			ntrees = 100  # testing
# 			# ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
# 		elif tag_alg_mark == "Mark4":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs10
# 			ntrees = 10  # testing
# 			# ntrees = 1000
# 			mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
# 			sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))
# 			class0 = sorted_unique_classes_in_y[0]
# 			class1 = sorted_unique_classes_in_y[1]
# 			class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
# 			class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
# 			prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
# 			prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
# 			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
# 		# tune with this under :
# 		# - randomstate is none by dflt so it pick up the seed in np.random.seed()
# 		# - mtry is "max_features" so max_features = mtry
# 		# - n_estimators is ntrees
# 		# the class_weight is 2 classes and each one attached the other class proportion
# 		# the importance is by default because the gini criterion already by default
# 		## test the classweight auto
# 		# randomForest(ol2_train_x, ol2_train_y, mtry=mtry_mdl_allfts, importance=TRUE, ntree=ntree, classwt=c(Res=prop.sens_mdl_allfts, Sen=prop.res_mdl_allfts))
# 	elif tag_alg == "XGB": # case of XGBoost as alg
# 		num_cores_heavy = 38 ##!! to check how it fits in with out multiprocessing
# 		num_cores_light = 2
# 		if tag_alg_mark == "Mark1":  # codename = XGB_default (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(seed=aseed)
# 		if tag_alg_mark == "Mark2":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
# 														 colsample_bytree=0.8, silent=1,
# 														 nthread=num_cores_heavy, seed=aseed)
# 		if tag_alg_mark == "Mark2nonthread":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
# 			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
# 														 colsample_bytree=0.8, silent=1,
# 														 seed=aseed) # nthread=num_cores_heavy not used because it risks undeertaking the allocated cores to il1 xproc
# 	elif tag_alg == "GBM":  # case of GBM as alg
# 		if tag_alg_mark == "Mark1":  # codename = GBM_default (see Sl_algs_descriptor.txt for params value)
# 			model = GradientBoostingClassifier(random_state=aseed)
# 	elif tag_alg == "DNN":  # case of DNN as alg
# 		if tag_alg_mark == "Mark1":  # codename = DNN_default (see Sl_algs_descriptor.txt for params value)
# 			print("in_dev")
# 		if tag_alg_mark == "Mark2":  # codename = DNN_recommended_zhou2019_v1 (see Sl_algs_descriptor.txt for params value)
# 			# create larger model
# 			model = Sequential()
# 			# # optional to use dropout on the input layer
# 			# model.add(Dropout(0.6, input_shape=(trainframe_x.shape[1],)))
# 			# model.add(Dense(512, kernel_initializer='normal', activation='relu'))
# 			model.add(Dense(512, input_dim=trainframe_x.shape[1], kernel_initializer='normal',kernel_regularizer=l2(1e-05)))
# 			model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 			# epsilon = 1e-05 to satisfy Theano demands for portability, mode = 0 feature-wise normalization,
# 			# axis is left to default instead of 1 for samples (for better management )
# 			# by default if weights=None:
# 			# beta_init = 'zero', gamma_init = 'one'
# 			# beta_regularizer=None,gamma_regularizer=None
# 			# beta_constraint = None, gamma_constraint = None
# 			# moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones'
# 			model.add(Activation('relu'))
# 			model.add(Dropout(0.6))
# 			model.add(Dense(256, kernel_initializer='normal', kernel_regularizer=l2(1e-05))) # replaced by a line : activation='relu'
# 			model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 			model.add(Activation('relu'))
# 			model.add(Dropout(0.6))
# 			model.add(Dense(64, kernel_initializer='normal', kernel_regularizer=l2(1e-05)))
# 			model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 			model.add(Activation('relu'))
# 			model.add(Dropout(0.6))
# 			model.add(Dense(1, kernel_initializer='normal'))
# 			model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 			model.add(Activation('sigmoid'))
# 			# Compile model
# 			adagrad_mark1_as_opt =optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0)
# 			model.compile(loss='binary_crossentropy', optimizer=adagrad_mark1_as_opt)
# 	return model
#
# def classifier_model_training(model, trainframe_x, trainframe_y,tag_alg,tag_alg_mark,featuretype, classif_list_cat_fts,aseed,RespClasses):
# 	np.random.seed(aseed)
# 	if tag_alg == "DNN": # requires some hyperparameters to be passed into the fit function
# 		if tag_alg_mark == "Mark1":  # codename = DNN_default (see Sl_algs_descriptor.txt for params value)
# 			print("in_dev")
# 		elif tag_alg_mark == "Mark2": # codename = DNN_recommended_zhou2019_v1 (see Sl_algs_descriptor.txt for params value)
# 			# following type of profile, scale values of features (if not categrical values) or not
# 			if featuretype not in classif_list_cat_fts:
# 				train_x_as_array = StandardScaler().fit_transform(trainframe_x)
# 			else:
# 				train_x_as_array = np.array(trainframe_x)
# 			# for the responses classes, the values have to be encoded
# 			le = LabelEncoder()  # the encoder
# 			le.fit(RespClasses)  # encode the classes to memorize
# 			# le.classes_ gives an array of the classes
# 			train_y_as_array = np.array(le.transform(trainframe_y.iloc[:, 0].tolist())) # get array from frames of thruths with values encoded
# 			# make the training
# 			model.fit(train_x_as_array, train_y_as_array, epochs=100, batch_size=100, verbose=0) # giving this to a variable only stock a history object
# 			# do this to store the model fitted
# 			model_fitted = model
# 	else: # a normal fit function used
# 		# the fit function in the case of most algs take 2 arrays in the next specifications otherwise error can be sent :
# 		# (the array_x is multi_dimensional ie shape(n_samples,#fts) and the array_y is 1_dimensional ie shape(n_samples,)
# 		train_x_as_array = np.array(trainframe_x) # a multi_d array is needed
# 		train_y_as_array = np.array(trainframe_y.iloc[:, 0].tolist())  # the array of train_y has to be 1_d ie created from a list made from the response column info
# 		model_fitted = model.fit(train_x_as_array, train_y_as_array)  # fit the method to the training data to get a classifier and predict with it later
# 	return model_fitted
#
# def classifier_model_prediction(model_fitted,tag_alg, testframe_x, prediction_type,aseed,RespClasses):
# 	np.random.seed(aseed)
# 	# same as in training the function used here ask for an array_x that is multi_dimensional ie shape(n_samples,#fts)
# 	test_x_as_array = np.array(testframe_x)
# 	if tag_alg == "DNN":  # requires some hyperparameters to be passed into the fit function
# 		if prediction_type == "prob":
# 			prob_class_pos = model_fitted.predict(test_x_as_array)[0][0] # model_fitted.predict(test_x_as_array) is a (1,1) shape array
# 			prob_class_neg = 1 - prob_class_pos
# 			model_prediction = np.array([[prob_class_neg, prob_class_pos]]) # based on a = np.array([[1, 1], [2, 2], [3, 3]]) is a (3,2) shape array
# 		elif prediction_type == "pred":
# 			code_pred_class = model_fitted.predict_classes(test_x_as_array)[0][0]  # model_fitted.predict(test_x_as_array) is a (1,1) shape array
# 			le = LabelEncoder()  # the encoder
# 			le.fit(RespClasses)  # encode the classes to memorize
# 			model_prediction = le.inverse_transform([code_pred_class]) #if to be used for future exploitation compare and make in same shape than regular output of model.predict for most models
# 	else:
# 		if prediction_type == "prob":
# 			model_prediction = model_fitted.predict_proba(test_x_as_array) # putting brackts aroung fix the need to reshpe error if working with series
# 		elif prediction_type == "pred":
# 			model_prediction = model_fitted.predict(test_x_as_array)  # putting brackts aroung fix the need to reshpe error
# 	return model_prediction



################################################
# # a new way to do DNN multiprocessing (parallelize batches but not the fit and predict)
# if tag_alg_mark == "Mark4Vseq":  # codename = DNN_recommended_zhou2019_v1 with session management for parallelism (see Sl_algs_descriptor.txt for params value)
# 	# ~~~~~~~~~~~~# functions for a new way to do DNN multiprocessing
# 	from keras.utils import Sequence
# 	from keras.utils import to_categorical
#
#
# 	class TrainDataGenerator(Sequence):
# 		# Sequence is a safer way to do multiprocessing.
# 		# This structure guarantees that the network will only train once on each sample per epoch
# 		# which is not the case with generators.
#
# 		# 'Generates data for Keras'
# 		def __init__(self, x_set, y_set, batch_size=5, n_classes=2, shuffle=True):
# 			# # following type of profile, scale values of features (if not categrical values) or not
# 			# if featuretype not in classif_list_cat_fts:
# 			# 	x_set = StandardScaler().fit_transform(x_set)
# 			# else:
# 			# 	x_set = np.array(x_set)
# 			# # for the responses classes, the values have to be encoded (using here an encoding done just after data_management)
# 			# # binary_classes_le.classes_ gives an array of the classes
# 			# y_set = np.array(binary_classes_le.transform(y_set.iloc[:, 0].tolist()))  # get array from frames of thruths with values encoded
#
# 			# 'Initialization'
# 			self.x = x_set
# 			self.y = y_set
# 			self.batch_size = batch_size
# 			self.n_classes = n_classes
# 			self.shuffle = shuffle
# 			self.on_epoch_end()
#
# 		# 'Updates indexes after each epoch'
# 		def on_epoch_end(self):
# 			# If the shuffle parameter is set to True, we will get a
# 			# new order of exploration at each pass (or just keep a linear
# 			# exploration scheme otherwise)"
# 			# " Shuffling the order in which examples are fed to the classifier is
# 			# helpful so that batches between epochs do not look alike.
# 			# Doing so will eventually make our model more robust."
# 			self.indexes = np.arange(len(self.x))
# 			if self.shuffle == True:
# 				np.random.shuffle(self.indexes)
#
# 		# ----Now comes the part where we build up these components together :
#
# 		# 1-Denotes the number of batches per epoch'
# 		def __len__(self):
# 			# Each call requests a batch index between 0 and the total number of batches,
# 			# where the latter is specified in the __len__ method.
# 			# A common practice is to set this value to : " samples / batch_size
# 			# (so that the model sees the training samples at most once per epoch)
# 			# ---- Also : each example being only fed once  is guaranteed by the fact that
# 			# we specified an appropriate number of batches per epoch in the __len__ method.
# 			# Keras takes care of the rest so the value of steps_per_epoch is not needed here
# 			# seem logical that step_per_epoch overrides __len__ when used to know when on_epoch_end
# 			# should be called (the name implies it)
# 			return int(np.floor(len(self.x) / float(self.batch_size)))
#
# 		# 2- Now, when the batch corresponding to a given index is called, the generator
# 		# executes the __getitem__ method to generate it.
# 		def __getitem__(self, index):
# 			# 'Generate one batch of data using indexes of the batch'
# 			batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
# 			batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
# 			batch_y_categ = to_categorical(batch_y, num_classes=self.n_classes)
# 			# return 2 arrays also called X, y
# 			return np.array(batch_x), batch_y_categ
#
#
# 	# Parameters
# 	params = {'batch_size': 5,
# 		'n_classes': 2,
# 		'shuffle': True}
# 	# Generators
# 	training_generator = TrainDataGenerator(trainframe_x, trainframe_y, **params)
# 	# testing_generator = TestDataGenerator(testframe_x, **params)
# 	# ~~~~~~~~~~~~Introduce the model
# 	# create large model
# 	model = Sequential()
# 	# # optional to use dropout on the input layer
# 	# model.add(Dropout(0.6, input_shape=(trainframe_x.shape[1],)))
# 	# model.add(Dense(512, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(512, input_dim=trainframe_x.shape[1], kernel_initializer='normal', kernel_regularizer=l2(1e-05)))
# 	model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 	# epsilon = 1e-05 to satisfy Theano demands for portability, mode = 0 feature-wise normalization,
# 	# axis is left to default instead of 1 for samples (for better management )
# 	# by default if weights=None:
# 	# beta_init = 'zero', gamma_init = 'one'
# 	# beta_regularizer=None,gamma_regularizer=None
# 	# beta_constraint = None, gamma_constraint = None
# 	# moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones'
# 	model.add(Activation('relu'))
# 	model.add(Dropout(0.6))
# 	model.add(Dense(256, kernel_initializer='normal', kernel_regularizer=l2(1e-05)))  # replaced by a line : activation='relu'
# 	model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(0.6))
# 	model.add(Dense(64, kernel_initializer='normal', kernel_regularizer=l2(1e-05)))
# 	model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(0.6))
# 	model.add(Dense(1, kernel_initializer='normal'))
# 	model.add(BatchNormalization(epsilon=1e-05, mode=0, momentum=0.9, weights=None))
# 	model.add(Activation('sigmoid'))
# 	# Compile model
# 	adagrad_mark1_as_opt = optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0)
# 	model.compile(loss='binary_crossentropy', optimizer=adagrad_mark1_as_opt)
# 	# ~~~~~~~~~~~~Train with the model
# 	model.fit_generator(generator=training_generator,
# 						use_multiprocessing=True,
# 						epochs=100,
# 						workers=6,
# 						verbose=1)
# 	# do this to store the model fitted
# 	model_fitted = model
# 	# predict with the trained model
# 	model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
################################################################################################################################################

# ##### trying this (not working
# import concurrent.futures
# import numpy as np
#
# import keras.backend as K
# from keras.layers import Dense
# from keras.models import Sequential
#
# import tensorflow as tf
# from tensorflow.python.client import device_lib
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     # return [x.name for x in local_device_protos if x.device_type == 'GPU']
# 	return [x.name for x in local_device_protos if ((x.device_type == 'CPU') | (x.device_type == 'XLA_CPU') )]
#
# xdata = np.random.randn(100, 8)
# ytrue = np.random.randint(0, 2, 100)
#
# def fit(gpu):
#     with tf.Session(graph=tf.Graph()) as sess:
#         K.set_session(sess)
#         with tf.device(gpu):
#             model = Sequential()
#             model.add(Dense(12, input_dim=8, activation='relu'))
#             model.add(Dense(8, activation='relu'))
#             model.add(Dense(1, activation='sigmoid'))
#
#             model.compile(loss='binary_crossentropy', optimizer='adam')
#             model.fit(xdata, ytrue, verbose=0)
#
#             return model.evaluate(xdata, ytrue, verbose=0)
#
# gpus = get_available_gpus()
# with concurrent.futures.ThreadPoolExecutor(len(gpus)) as executor:
#     results = [x for x in executor.map(fit, gpus)]
# print('results: ', results)

######copies kept from old dnn with keras  classifier
# # testing this
# # *********DNN using Keras imports (done here and not on top to avoid multiprocessing issues
# # ----------a random seed initialisation to fixate some randomness due to intialisation of librairies or ibjects
# np.random.seed(aseed)  # first intention was to minimize variations in neural networks ## use 0 if not working
# # import tensorflow as tf # testing this : previously was : import tensorflow.compat.v1 as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# # from tensorflow import set_random_seed # as TensorFlow backend is used by Keras DNNs and TensorFlow has its own random number generator, that must also be seeded
# tf.set_random_seed(aseed)  ## use 0 if not working
# # ----DNN implementation
# from keras import optimizers  # for the optimizers
# from keras.regularizers import l2  # for the weight decay as L2 regularization
# from keras.layers import Dropout  # from the dropout regularization essay
# from keras.models import Sequential  # for the model structure
# from keras.layers import Dense  # for each input layer mostly hidden layers
# from keras.layers import Activation  # for activation functions
# # from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
# from sklearn.preprocessing import StandardScaler # to scale again the real values features
# from keras.layers.normalization import BatchNormalization  # for batch normlisation between X.w and batchnormalised (X.w) + b
# from keras.backend import tensorflow_backend as K # testing this : previously was : from keras import backend as K
# # testing this
##### functions use examples
# # build model for training
# model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
# 					kernel_regularizer_decay_val=1e-05,
# 					bn=True, bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None,
# 					input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
# 					add_dropout=True, dropout_val=0.6,
# 					loss_function='binary_crossentropy', optimizer_lr=0.1, optimizer_epsilon=None, optimizer_decay=0.0)
# # extract the trained model
# model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=1)
# # predict with the trained model
# model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
#****old managements :
# if tag_alg_mark == "Mark1Vseq":  # codename = DNN_recommended_zhou2019_v1 with session management for parallelism (see Sl_algs_descriptor.txt for params value)
	# 	# # ~~~~ the session management V1.1
	# 	# config_par_autodetect_log = tf.ConfigProto(
	# 	# 						allow_soft_placement=True,
	# 	# 						log_device_placement=True)
	# 	# session = tf.Session(config=config_par_autodetect_log)
	# 	# K.set_session(session)
	# 	# ~~~~ the session management V1.2
	# 	num_cores = 38
	#
	# 	from tensorflow.python.client import device_lib
	# 	local_device_protos = device_lib.list_local_devices()
	# 	if "GPU" in [x.device_type for x in local_device_protos]:
	# 		num_GPU = 1
	# 		num_CPU = 1
	# 	if "CPU" in [x.device_type for x in local_device_protos]:
	# 		num_CPU = 1
	# 		num_GPU = 0
	#
	# 	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
	# 							inter_op_parallelism_threads=num_cores,
	# 							allow_soft_placement=True,
	# 							device_count={'CPU': num_CPU,
	# 								'GPU': num_GPU}
	# 							)
	# 	session = tf.Session(config=config)
	# 	K.set_session(session)
	# 	# build model for training
	# 	model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
	# 								  kernel_regularizer_decay_val=1e-05,
	# 								  bn=True, bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None,
	# 								  input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
	# 								  add_dropout=True, dropout_val=0.6,
	# 								  loss_function='binary_crossentropy', optimizer_lr=0.1, optimizer_epsilon=None, optimizer_decay=0.0)
	# 	# extract the trained model
	# 	model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=1)
	# 	# predict with the trained model
	# 	model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
	# 	K.clear_session()  # ~~~~end of session management
	# if tag_alg_mark == "Mark2Vseq":  # codename = DNN_recommended_zhou2019_v1 with session management for sequential (see Sl_algs_descriptor.txt for params value)
	# 	# # ~~~~ the session management V2.1
	# 	# config_seq_log_autodetect = tf.ConfigProto(intra_op_parallelism_threads=1,
	# 	# 						inter_op_parallelism_threads=1,
	# 	# 						log_device_placement=True,
	# 	# 						allow_soft_placement=True)
	# 	# session = tf.Session(config=config_seq_log_autodetect)
	# 	# K.set_session(session)
	# 	# ~~~~ the session management V2.2 num cores = 10
	# 	num_cores = 10
	#
	# 	from tensorflow.python.client import device_lib
	# 	local_device_protos = device_lib.list_local_devices()
	# 	if "GPU" in [x.device_type for x in local_device_protos]:
	# 		num_GPU = 1
	# 		num_CPU = 1
	# 	if "CPU" in [x.device_type for x in local_device_protos]:
	# 		num_CPU = 1
	# 		num_GPU = 0
	#
	# 	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
	# 							inter_op_parallelism_threads=num_cores,
	# 							allow_soft_placement=True,
	# 							device_count={'CPU': num_CPU,
	# 								'GPU': num_GPU}
	# 							)
	# 	session = tf.Session(config=config)
	# 	K.set_session(session)
	# 	# build model for training
	# 	model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
	# 								  kernel_regularizer_decay_val=1e-05,
	# 								  bn=True, bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None,
	# 								  input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
	# 								  add_dropout=True, dropout_val=0.6,
	# 								  loss_function='binary_crossentropy', optimizer_lr=0.1, optimizer_epsilon=None, optimizer_decay=0.0)
	# 	# extract the trained model
	# 	model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=1)
	# 	# predict with the trained model
	# 	model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
	# 	K.clear_session() #~~~~end of session management
	# if tag_alg_mark == "Mark3Vseq":  # codename = DNN_recommended_zhou2019_v1 with session management for parallelism (see Sl_algs_descriptor.txt for params value)
	# 	# # ~~~~ the session management V3.1 :  no log or soft placement to reduce times of steps at max through the epochs
	# 	# config_par_noautodetect_nolog = tf.ConfigProto()
	# 	# session = tf.Session(config=config_par_noautodetect_nolog)
	# 	# K.set_session(session)
	# 	# # ~~~~ the session management V3.2 num cores = 10 verbose=0 # testing this
	# 	# num_cores = 10
	# 	#
	# 	# from tensorflow.python.client import device_lib
	# 	# local_device_protos = device_lib.list_local_devices()
	# 	# if "GPU" in [x.device_type for x in local_device_protos]:
	# 	# 	num_GPU = 1
	# 	# 	num_CPU = 1
	# 	# if "CPU" in [x.device_type for x in local_device_protos]:
	# 	# 	num_CPU = 1
	# 	# 	num_GPU = 0
	# 	#
	# 	# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
	# 	# 						inter_op_parallelism_threads=num_cores,
	# 	# 						allow_soft_placement=True,
	# 	# 						device_count={'CPU': num_CPU,
	# 	# 							'GPU': num_GPU}
	# 	# 						)
	# 	# session = tf.Session(config=config)
	# 	# K.set_session(session)
	# 	# ~~~~ the session management V3.3 num cores = 10 verbose=0 # testing this
	# 	num_cores = 30
	#
	# 	with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_cores)) as sess:
	# 		K.set_session(sess)
	# 		# build model for training # testing this
	# 		model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
	# 									  kernel_regularizer_decay_val=1e-05,
	# 									  bn=True, bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None,
	# 									  input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
	# 									  add_dropout=True, dropout_val=0.6,
	# 									  loss_function='binary_crossentropy', optimizer_lr=0.1, optimizer_epsilon=None, optimizer_decay=0.0)
	# 		# extract the trained model
	# 		model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=0)
	# 		# predict with the trained model
	# 		model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
	# 		# K.clear_session()  # ~~~~end of session management # testing this
	#####ON TRIAL#####
# # # ~~~~ the session management V4.1 num cores = 10 verbose=0
# # num_cores = 1
# #
# # from tensorflow.python.client import device_lib
# # local_device_protos = device_lib.list_local_devices()
# # if "GPU" in [x.device_type for x in local_device_protos]:
# # 	num_GPU = 1
# # 	num_CPU = 1
# # if "CPU" in [x.device_type for x in local_device_protos]:
# # 	num_CPU = 1
# # 	num_GPU = 0
# #
# # config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
# # 						inter_op_parallelism_threads=num_cores,
# # 						allow_soft_placement=True,
# # 						device_count={'CPU': num_CPU,
# # 							'GPU': num_GPU}
# # 						)
# # session = tf.Session(config=config)
# # K.set_session(session)
# # testing this
# testing this
# K.clear_session()  # ~~~~end of session management
# testing this
####last system management developped
# # ~~~~ the session management V4.1 num cores = 10 verbose=0
		# num_cores = 10
		#
		# from tensorflow.python.client import device_lib
		# local_device_protos = device_lib.list_local_devices()
		# if "GPU" in [x.device_type for x in local_device_protos]:
		# 	num_GPU = 1
		# 	num_CPU = 1
		# if "CPU" in [x.device_type for x in local_device_protos]:
		# 	num_CPU = 1
		# 	num_GPU = 0
		#
		# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
		# 						inter_op_parallelism_threads=num_cores,
		# 						allow_soft_placement=True,
		# 						device_count={'CPU': num_CPU,
		# 							'GPU': num_GPU}
		# 						)
		# session = tf.Session(config=config)
		# K.set_session(session)
		# # testing this
		# # ~~~~ the session management V3.3 num cores = 10 verbose=0 # testing this
		# num_cores = 30
		#
		# with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_cores)) as sess:
		# 	K.set_session(sess)
		# 	# build model for training # testing this
		# 	model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
		# 								  kernel_regularizer_decay_val=1e-05,
		# 								  bn=True, bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None,
		# 								  input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
		# 								  add_dropout=True, dropout_val=0.6,
		# 								  loss_function='binary_crossentropy', optimizer_lr=0.1, optimizer_epsilon=None, optimizer_decay=0.0)
		# 	# extract the trained model
		# 	model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=0)
		# 	# predict with the trained model
		# 	model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
		# # testing this

# #### the separated in three functions dnn classifier
# def classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark,trainframe_x, trainframe_y,featuretype, classif_list_cat_fts,binary_classes_le,testframe_x, prediction_type,aseed):
# 	# ~~~~~~~~~~~~ function for creating model with given values of hyperparams
# 	# optional to use dropout on the input layer
# 	# epsilon = 1e-05 to satisfy Theano demands for portability, mode = 0 feature-wise normalization,
# 	# axis is left to default instead of 1 for samples (for better management )
# 	# by default if weights=None:
# 	# beta_init = 'zero', gamma_init = 'one'
# 	# beta_regularizer=None,gamma_regularizer=None
# 	# beta_constraint = None, gamma_constraint = None
# 	# moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones'
# 	def build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=20000, kernel_initializer_type='normal',
# 							  kernel_regularizer_decay_val=1e-05,
# 							  bn=True,
# 							  input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
# 							  add_dropout=True, dropout_val=0.6,
# 							  loss_function='binary_crossentropy', optimizer_lr=0.1):
# 		#--- modified hyperparams values :
# 		# eliminate mode for tf.keras and simplify params because unknown territory (bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None, # before)
# 		# eliminate and simplify some params because unknown territory (, optimizer_epsilon=None, optimizer_decay=0.0 # before)
# 		#---- necessary imports and random env init :
# 		np.random.seed(aseed)
# 		import tensorflow as tf
# 		tf.set_random_seed(aseed)
# 		from tensorflow.python.keras import optimizers
# 		from tensorflow.python.keras.regularizers import l2
# 		from tensorflow.python.keras.layers import Dropout
# 		from tensorflow.python.keras.models import Sequential
# 		from tensorflow.python.keras.layers import Dense
# 		from tensorflow.python.keras.layers import Activation
# 		from tensorflow.python.keras.layers.normalization import BatchNormalization
# 		#------- Introduce the model
# 		model = Sequential()
# 		# the input layer
# 		model.add(Dense(layers[0], input_dim=input_layer_dim, kernel_initializer=kernel_initializer_type, kernel_regularizer=l2(kernel_regularizer_decay_val)))
# 		if bn:
# 			model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
# 		model.add(Activation(input_hidden_layers_activation_type))
# 		if add_dropout:
# 			model.add(Dropout(dropout_val))
# 		# the hidden layers
# 		for i in layers[1:]:
# 			model.add(Dense(i, kernel_initializer=kernel_initializer_type, kernel_regularizer=l2(kernel_regularizer_decay_val)))
# 			if bn:
# 				model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
# 			model.add(Activation(input_hidden_layers_activation_type))
# 			if add_dropout:
# 				model.add(Dropout(dropout_val))
# 		# the output layer
# 		model.add((Dense(1, kernel_initializer=kernel_initializer_type)))
# 		if bn:
# 			model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
# 		model.add(Activation(output_layer_activation_type))
# 		# Compile the model
# 		adagrad_mark1_as_opt = optimizers.Adagrad(lr=optimizer_lr)  # eliminate and simplify some params because unknown territory (, epsilon=optimizer_epsilon, decay=optimizer_decay # before)
# 		model.compile(loss=loss_function, optimizer=adagrad_mark1_as_opt)
# 		# testing this
# 		print("1 dnn model architecture made : ")
# 		# testing this
# 		return model
# 	# ~~~~~~~~~~~~  function for training model with given values of hyperparams
# 	def train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=0):
# 		# #testing this
# 		from sklearn.preprocessing import StandardScaler  # to scale again the real values features
# 		# # testing this
# 		# ~~~~~~~~~~~~Train with the model
# 		# following type of profile, scale values of features (if not categrical values) or not
# 		if featuretype not in classif_list_cat_fts:
# 			train_x_as_array = StandardScaler().fit_transform(trainframe_x)
# 		else:
# 			train_x_as_array = np.array(trainframe_x)
# 		# for the responses classes, the values have to be encoded (using here an encoding done just after data_management)
# 		# binary_classes_le.classes_ gives an array of the classes
# 		train_y_as_array = np.array(binary_classes_le.transform(trainframe_y.iloc[:, 0].tolist()))  # get array from frames of thruths with values encoded
# 		# choose the batch size
# 		if batch_size_strategy == "half_pop":
# 			chosen_batch_size = int(np.floor(trainframe_x.shape[0] / 2))
# 		else:
# 			chosen_batch_size = batch_size_default_val
# 		model.fit(train_x_as_array, train_y_as_array, epochs=epochs_num, batch_size=chosen_batch_size, verbose=verbose_in_fit)  # giving this to a variable only stock a history object
# 		# do this to store the model fitted
# 		model_fitted = model
# 		# testing this
# 		print("1 dnn model trained : ")
# 		# testing this
# 		return model_fitted
# 	# ~~~~~~~~~ function to predict with model fitted
# 	def predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le):
# 		# ~~~~~~~~~~~~Predict with the model
# 		# same as in training the function used here ask for an array_x that is multi_dimensional ie shape(n_samples,#fts)
# 		test_x_as_array = np.array(testframe_x)
# 		if prediction_type == "prob":
# 			prob_class_pos = model_fitted.predict(test_x_as_array)[0][0]  # model_fitted.predict(test_x_as_array) is a (1,1) shape array
# 			prob_class_neg = 1 - prob_class_pos
# 			model_prediction = np.array([[prob_class_neg, prob_class_pos]])  # based on a = np.array([[1, 1], [2, 2], [3, 3]]) is a (3,2) shape array
# 		elif prediction_type == "pred":
# 			code_pred_class = model_fitted.predict_classes(test_x_as_array)[0][0]  # model_fitted.predict(test_x_as_array) is a (1,1) shape array
# 			# binary_classes_le is an encoding done just after data_management
# 			model_prediction = binary_classes_le.inverse_transform([code_pred_class])  # if to be used for future exploitation compare and make in same shape than regular output of model.predict for most models
# 		# testing this
# 		print("1 dnn prediction done : ",model_prediction )
# 		# print(model_prediction)
# 		# testing this
# 		return model_prediction
# 	if tag_alg_mark == "Mark4Vseq":
# 		# build model for training
# 		model = build_keras_dnn_model(layers=(512, 256, 64), input_layer_dim=trainframe_x.shape[1], kernel_initializer_type='normal',
# 							kernel_regularizer_decay_val=1e-05,
# 							bn=True,
# 							input_hidden_layers_activation_type='relu', output_layer_activation_type='sigmoid',
# 							add_dropout=True, dropout_val=0.6,
# 							loss_function='binary_crossentropy', optimizer_lr=0.1)
# 		# eliminate mode for tf.keras and simplify params because unknown territory (bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None, # before)
# 		# eliminate and simplify some params because unknown territory (, optimizer_epsilon=None, optimizer_decay=0.0 # before)
# 		# extract the trained model
# 		model_fitted = train_keras_dnn_model(model, trainframe_x, trainframe_y, featuretype, classif_list_cat_fts, binary_classes_le, epochs_num=100, batch_size_strategy="half_pop", batch_size_default_val=100, verbose_in_fit=0)
# 		# predict with the trained model
# 		model_prediction = predicting_w_keras_dnn_model(model_fitted, testframe_x, prediction_type, binary_classes_le)
# 		# K.clear_session()  # ~~~~end of session management # testing this
# 	return model_prediction

#****added after extension of profiles analysed
data_loadout_right_cnwgex = []
data_loadout_right_gexwcn = []
data_loadout_right_snvwcna = []
data_loadout_right_cnawsnv = []
data_loadout_right_snvwcnawgexa = []
data_loadout_right_snvwgexawcna = []
data_loadout_right_cnawsnvwgexa = []
data_loadout_right_cnawgexawsnv = []
data_loadout_right_gexawsnvwcna = []
data_loadout_right_gexawcnawsnv = []

data_loadout_right_cnwgex, data_loadout_right_gexwcn, \
	data_loadout_right_snvwcna, data_loadout_right_cnawsnv,\
	data_loadout_right_snvwcnawgexa, data_loadout_right_snvwgexawcna, \
	data_loadout_right_cnawsnvwgexa,data_loadout_right_cnawgexawsnv,
data_loadout_right_gexawsnvwcna,data_loadout_right_gexawcnawsnv
#----origin files loadouts
data_loadouts_origin_files_for_cnwgex = []
data_loadouts_origin_files_for_gexwcn = []
data_loadouts_origin_files_for_snvwcna = []
data_loadouts_origin_files_for_cnawsnv = []
data_loadouts_origin_files_for_snvwcnawgexa = []
data_loadouts_origin_files_for_snvwgexawcna = []
data_loadouts_origin_files_for_cnawsnvwgexa = []
data_loadouts_origin_files_for_cnawgexawsnv = []
data_loadouts_origin_files_for_gexawsnvwcna = []
data_loadouts_origin_files_for_gexawcnawsnv = []


data_loadouts_origin_files_for_cnwgex, data_loadouts_origin_files_for_gexwcn,
data_loadouts_origin_files_for_snvwcna, data_loadouts_origin_files_for_cnawsnv,
data_loadouts_origin_files_for_snvwcnawgexa, data_loadouts_origin_files_for_snvwgexawcna,
data_loadouts_origin_files_for_cnawsnvwgexa, data_loadouts_origin_files_for_cnawgexawsnv,
data_loadouts_origin_files_for_gexawsnvwcna, data_loadouts_origin_files_for_gexawcnawsnv

#****
"_SNVwCNA",
"_CNAwSNV",
"_CNwGEX",
"_GEXwCN",
"_SNVwCNAwGEXA",
"_SNVwGEXAwCNA",
"_CNAwSNVwGEXA",
"_CNAwGEXAwSNV",
"_GEXAwSNVwCNA",
"_GEXAwCNAwSNV" ##! make a rule later to estimate if profile is categ or not

aseed = 1

import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
# skf = StratifiedKFold(n_splits=2)
skf = StratifiedKFold(n_splits=2,shuffle=True,random_state=aseed)
print(skf)
for train_index, test_index in skf.split(X, y):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


dframe_x = dframe[list(dframe)[5:]]
dframe_x = np.array(dframe_x)
dframe_y = dframe.loc[:,[Resp_col_name]]
dframe_y = np.array(dframe_y)

for train_index, test_index in skf.split(dframe_x, dframe_y):
	print("TRAIN:", train_index, "TEST:", test_index)
	dframeX_train, dframeX_test = dframe_x[train_index], dframe_x[test_index]
	dframey_train, dframey_test = dframe_y[train_index], dframe_y[test_index]

stratKfolds_making
stratKfolds_making(10,0,dframe,Resp_col_name,True)
stratKfolds_making(10,aseed,dframe,Resp_col_name,True)

ol2_folds = stratKfolds_making(10,aseed,dframe,Resp_col_name,True)
ol2_train_data = dframe.loc[j[1],:] # Set the training set (for a row j of data, select all rows of data without the row of index 1 (here the frame start index at 1)
ol2_test_data = dframe.loc[j[2],:]

Classif_cross_validation_folds_number
Regr_cross_validation_folds_number
classif_CV_folds_number