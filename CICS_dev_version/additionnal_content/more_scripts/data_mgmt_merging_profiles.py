##### script for the datamanagement to create across ctypes datasets :
import pandas as pd # for dataframes manipulation
import os
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataBis/pData.63"): # change here folder of prof in ctype1
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
for root, directories, filenames in os.walk("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/PDX__data/processedDataBis/pData.64"): # change here folder of prof in ctype2
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
# regrouping both frames in a list to parse their list of samples (rows) for unique and aftzerwards find the common samples (rows)
frames = []
frames.append(c1_df_cn)
frames.append(c2_df_cn)
# common_cols = list(set.intersection(*(set(df.columns) for df in frames))) # we need common rows for this one
common_samples = list(set.intersection(*(set(df["Model"]) for df in frames)))
# lets put a reminder of original profile on each column of each df
profile1_prefix = ''.join([profile1,"of"])
profile2_prefix = ''.join([profile2,"of"])
c1_df_cn_tagged = c1_df_cn.add_prefix(profile1_prefix)
c2_df_cn_tagged = c1_df_cn.add_prefix(profile2_prefix)
# joining the two frames
sample_colname_joined_on_for_c1_df = ''.join([profile1_prefix,"Model"])
sample_colname_joined_on_for_c2_df = ''.join([profile2_prefix,"Model"])
frames_joined = pd.merge(c1_df_cn_tagged, c2_df_cn_tagged, how="inner", left_on=sample_colname_joined_on_for_c1_df, right_on=sample_colname_joined_on_for_c2_df)
# lets drop the rows that show 2 values of response that are different
resp_colname_from_c1 = ''.join([profile1_prefix,"BestResCategory"])
resp_colname_from_c2 = ''.join([profile2_prefix,"BestResCategory"])
frames_joined = frames_joined[frames_joined[resp_colname_from_c1] == frames_joined[resp_colname_from_c2]] # or use df = df.query("S != T")
# lets drop one in each duo of cols used to join
frames_joined.drop(labels=[sample_colname_joined_on_for_c2_df,resp_colname_from_c2], axis=1, inplace=True)
# lets rename the duo of samples and response cols remaining with the right names
frames_joined = frames_joined.rename(columns={sample_colname_joined_on_for_c1_df: "Model"})
frames_joined = frames_joined.rename(columns={resp_colname_from_c1: "BestResCategory"})
# sort the columns before moving cols of samples and resp in their right place to have after wards fts in order
frames_joined = frames_joined.reindex(sorted(frames_joined.columns), axis=1)
# put col of samples in right place
column_samples_to_move = frames_joined['Model']
frames_joined.drop(labels=['Model'], axis=1, inplace=True)
frames_joined.insert(frames_joined.shape[1], 'Model', column_samples_to_move)
# put col of response in right place
column_resp_to_move = frames_joined['BestResCategory']
frames_joined.drop(labels=['BestResCategory'], axis=1, inplace=True)
frames_joined.insert(0, 'BestResCategory', column_resp_to_move)
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
frames_joined.to_csv(filename_final, index=None, header=True)
