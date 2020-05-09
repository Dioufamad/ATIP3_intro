##### script for the datamanagement to create across ctypes datasets :
import pandas as pd # for dataframes manipulation
import os
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataBis/pData.28"): # change here folder of prof in ctype1
	for file in filenames:
		# print(file)
		filename1 = file
		filename1_corrected = filename1.replace('.', '_')
		elts_of_filename1 = filename1_corrected.split("_")
		ctype1 = elts_of_filename1[0]
		drugid1 = elts_of_filename1[1]
		response_type1 = elts_of_filename1[2]
		profile1 = elts_of_filename1[3]
		c1_df_cn = pd.read_csv(os.path.join(root, file))
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataBis/pData.32"): # change here folder of prof in ctype2
	for file in filenames:
		# print(file)
		filename2 = file
		filename2_corrected = filename2.replace('.', '_')
		elts_of_filename2 = filename2_corrected.split("_")
		ctype2 = elts_of_filename2[0]
		drugid2 = elts_of_filename2[1]
		response_type2 = elts_of_filename2[2]
		profile2 = elts_of_filename2[3]
		c2_df_cn = pd.read_csv(os.path.join(root, file))
# regrouping both frames in a list to parse their list of columns for unique and aftzerwards find the common columns
frames = []
frames.append(c1_df_cn)
frames.append(c2_df_cn)
common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
one_unik_df_from_c1cn_c2cn = pd.concat([df[common_cols] for df in frames], ignore_index=True)
# sort the columns before moving cols of samples and resp in their right place to have after wards fts in order
one_unik_df_from_c1cn_c2cn = one_unik_df_from_c1cn_c2cn.reindex(sorted(one_unik_df_from_c1cn_c2cn.columns), axis=1)
# put col of samples in right place
column_samples_to_move = one_unik_df_from_c1cn_c2cn['Model']
one_unik_df_from_c1cn_c2cn.drop(labels=['Model'], axis=1, inplace=True)
one_unik_df_from_c1cn_c2cn.insert(one_unik_df_from_c1cn_c2cn.shape[1], 'Model', column_samples_to_move)
# put col of response in right place
column_resp_to_move = one_unik_df_from_c1cn_c2cn['BestResCategory']
one_unik_df_from_c1cn_c2cn.drop(labels=['BestResCategory'], axis=1, inplace=True)
one_unik_df_from_c1cn_c2cn.insert(0, 'BestResCategory', column_resp_to_move)
# making of filename
def consensus_maker_on_elts(elt1,elt2):
	if elt1 == elt2:
		consensus = elt1
	else:
		consensus = 'w'.join([elt1,elt2])
	return consensus
ctype_final = consensus_maker_on_elts(ctype1,ctype2)
drugid_final = consensus_maker_on_elts(drugid1,drugid2)
response_type_final = consensus_maker_on_elts(response_type1,response_type2)
profile_final = consensus_maker_on_elts(profile1,profile2)
filename_final = ctype_final + "_" + drugid_final + "_" + response_type_final + "_" + profile_final + ".csv"
# save df as a .csv file ready for analysis
one_unik_df_from_c1cn_c2cn.to_csv(filename_final, index=None, header=True)