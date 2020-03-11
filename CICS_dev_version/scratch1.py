# separators
# >>>>>>>>>>>>>>>>>>>>>>>>>>>IMPORTS
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<END OF IMPORTS

#======================================##! 881
import os
os.system("ls")
curpath = os.getcwd()
curpath
#======================================
import os
d = os.path.dirname(__file__) # directory of script
p = r'{}/results/graphs'.format(d) # path to be created

try:
    os.makedirs(p)
except OSError:
    pass
#======================================


#
# # meth
# test_df =df_joined[df_joined.columns[:10]]
# test_df = test_df.apply(lambda x: x.str.replace(',','.',regex=True))
# testdf1 = df_joined
#
# testdf1[feat_cols] = testdf1[feat_cols].stack().str.replace(',','.').unstack()
# testdf1[feat_cols] = testdf1[feat_cols].astype('float64')
#
# feat_cols = test_df.columns[2:]
# test_df[feat_cols] = test_df[feat_cols].stack().str.replace(',','.').unstack()
# test_df[feat_cols] = test_df[feat_cols].astype('float64')
#
# test_df[feat_cols] = test_df[feat_cols].transform(lambda x: x.str.replace(',','.'))
# test_df.dtypes
# test_df[feat_cols] = test_df[feat_cols].astype('float64')
# test_df.dtypes
#
#
# for bad_col in test_df.columns:
#     test_df[bad_col].str.replace(",",".")
#
# # meth
# cols = df_joined.columns[df_joined.dtypes.eq(object)]
# cols10 = cols[:10]
# df_joined[cols10] = df_joined[cols10].apply(pd.to_numeric, errors='coerce')
#
#
# for a_bad_col in cols10:
#     df_joined[a_bad_col] = df_joined[a_bad_col].convert_objects(convert_numeric=True)
#
# index_of_1st_ft = 2
# for col_as_ft_going_float in list(df_joined)[index_of_1st_ft:]:
#     if df_joined[col_as_ft_going_float].dtypes != "float64":
#         # print("The feature ", col_as_ft_going_float,  "was found to not be float and will be forced into it...")
# 	    df_joined[col_as_ft_going_float] = df_joined[col_as_ft_going_float].astype('float64')
# print("All features columns changed into float64 dtype...")
#
#
# # meth
# cols = df_joined.columns[df_joined.dtypes.eq(object)]
# df_joined[cols] = df_joined[cols].apply(pd.to_numeric, errors='coerce')
#
#
#
# # meth
# ser = pd.Series(df_joined[feat_cols])
#
# df_joined[df_joined.columns[2:]] = pd.to_numeric(pd.Series([df_joined.columns[2:]]),errors="coerce")
#
# # meth
# ter = pd.Series(df_joined)
#
#
# index_of_1st_ft = 2
# for col_as_ft_going_float in list(df_joined)[index_of_1st_ft:]:
#     if df_joined[col_as_ft_going_float].dtypes != "float64":
#         # print("The feature ", col_as_ft_going_float,  "was found to not be float and will be forced into it...")
# 	    df_joined[col_as_ft_going_float] = df_joined[col_as_ft_going_float].astype('float64')
# print("All features columns changed into float64 dtype...")
#
# feat_cols = df_joined.columns[2:]
# df_joined[feat_cols] = df_joined[feat_cols].convert_objects(convert_numeric=True)
#
# index_of_1st_ft = 2
# for col_as_ft_going_float in list(df_joined)[index_of_1st_ft:]:
# 	df_joined[col_as_ft_going_float] = pd.to_numeric(df_joined[col_as_ft_going_float], errors="coerce")
#
# df_joined[list(df_joined)[:10]].info()

####=============================================================####
#  df_joined[feat_cols] = df_joined[feat_cols].astype('float64') # slow
# df_joined[feat_cols] = df_joined[feat_cols].apply(pd.to_numeric, errors="coerce") # ok
# # df_joined[df_joined.columns[:10]].dtypes # use this to get a peak at the dtypes in the final dataframe
#
# #meth1
# cols_done = 0
# for col2change in feat_cols:
# 	df_joined[col2change] = df_joined[col2change].apply(pd.to_numeric, errors="coerce")
# 	cols_done +=1
# 	print(cols_done, "done in",len(feat_cols),".")
#
#
# # meth 2
# colz_done = 0
# for col_as_ft_going_float in list(df_joined)[2:]:
# 	if df_joined[col_as_ft_going_float].dtypes != "float64":
# 		# print("The feature ", col_as_ft_going_float,  "was found to not be float and will be forced into it...")
# 		df_joined[col_as_ft_going_float] = df_joined[col_as_ft_going_float].astype('float64')
# 		colz_done +=1
# 		print(colz_done, "done in",len(list(df_joined)[2:]),".")
# print("All features columns changed into float64 dtype...")
#
# # meth 3
# df2 = df_joined
# df2[df2.columns[2:10]] = df2[df2.columns[2:10]].astype(np.float64)
# df2[feat_cols] = df2[feat_cols].astype(np.float64)
#
# #meth 4 ##! this destroys the columns name so keep the columns name aside and reapply them after ward
# df3 = df_joined
# df3[df3.columns[2:10]] = pd.DataFrame(df3[df3.columns[2:10]].values.astype(np.float64))
# df_joined[feat_cols] = pd.DataFrame(df_joined[feat_cols].values.astype(np.float64))
#
# #meth 5 ##! this destroys the columns name so keep the columns name aside and reapply them after ward
# df4 = df_joined
# colz_done = 0
# for col2do in list(df4)[2:]:
# 	# real_name_col = col2do
# 	# real_index_col = list(df4)[2:].index(real_name_col)
# 	if df4[col2do].dtypes != "float64":
# 		df4[col2do] = pd.DataFrame(df4[col2do].values.astype(np.float64))
# 		# df4.columns[real_index_col] = real_name_col
# 		colz_done +=1
# 		print(colz_done, "done in",len(list(df4)[2:]),".")
# print("All features columns changed into float64 dtype...")
#
# # meth 6 used to change all fts values into floats
# df5 = df_joined
# df_fts = df5[list(df5)[2:]]
# old_col_names = df_fts.columns
# df_fts_changed_as_series = df_fts.values.astype(np.float64)
# df_fts_changed_as_df = pd.DataFrame(df_fts_changed_as_series)
#
# df_fts_changed_as_df.columns = old_col_names
# df5[list(df5)[2:]] = df_fts_changed_as_df
# # or also try this
# df_fts_changed_as_df.insert(0, df5.columns[0], df5[df5.columns[0]])
# df_fts_changed_as_df.insert(1, df5.columns[1], df5[df5.columns[1]])
#
#
#
#
#
# df3[df3.columns[2:10]] = pd.DataFrame(df3[df3.columns[2:10]].values.astype(np.float64))
# df_joined[feat_cols] = pd.DataFrame(df_joined[feat_cols].values.astype(np.float64))
