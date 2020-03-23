# separators
# >>>>>>>>>>>>>>>>>>>>>>>>>>>IMPORTS
##! last_stop
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<END OF IMPORTS
# #cleaning out the coerced values and reporting on the loss due to formatting the fts dtypes
# samples_b4_coercing = len(dframe.axes[0])
# fts_b4_coercing = len(dframe.axes[1])-2
# dframe.dropna(axis='index',inplace = True) # removing null values to avoid errors
# samples_aft_coercing = len(dframe.axes[0])
# fts_aft_coercing = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
# lost_samples = samples_b4_coercing - samples_aft_coercing
# lost_fts = fts_b4_coercing - fts_aft_coercing
# print("Report on the losses during the formatting of the features data types : ")
# if lost_samples==0:
# 	print("No samples has been lost")
# else:
# 	print(lost_samples,"samples has been lost")
# if lost_fts==0:
# 	print("No features has been lost")
# else:
# 	print(lost_fts,"samples has been lost")
#======================================##! 881

import os
print('[initial directory]')
print('getcwd:      ', os.getcwd())
print('[change directory]')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('getcwd:      ', os.getcwd())


# import os
# os.system("ls")
# curpath = os.getcwd()
# curpath
# #======================================
# import os
# d = os.path.dirname(__file__) # directory of script
# d
# p = r'{}/results/graphs'.format(d) # path to be created
#
# try:
#     os.makedirs(p)
# except OSError:
#     pass
# #======================================


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


# #===================
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>INITIAL DATA ANALYSIS OPERATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print("The final dataframe (dframe) is supplied ! Now onto the proper data analysis...")
# print("We will try to have a complete cycle of the data analysis including :  ")
# print("- visuals on data.")
# print("- univariates analysis")
# print("- multivariate analysis")
# print("- machine learning analysis")
# #---> Let's do statistics on our datasets variables
# # check if in each column, the values stay in a reasonable scale and what they are :
# # (missing values?,min, max, quartils for scale, mean, sd, etc. )
# # - removing null values to avoid errors
# # dframe.dropna(axis='columns',inplace=True) # (already done)
# # - percentile list
# # perc =[.25, .50, .75, .90]  ##! can be choosed later with an arguement
# # - list of dtypes to include
# # include =['object', 'float', 'int']    ##! to be choosed by operator   (1 argument)
# # - stash the describe result
# # desc = sdss_df.describe(percentiles = perc, include = include)  # used by the args for perc values and the types
# desc = dframe[dframe.columns[:10]].describe() ##! to reuse with list of candidates genes in order to see how there values are
# ## default include=None only takes into acount the numeric as dtypes columns (very good compromise to easily capture except for a few prticular cases of samples names in numeric
# # default  percentile are 0.25, 0.50 and 0.75 so quite okay to join it with min and max and see the range the values of a feature are staying
# # calling describe method (optional) ##! keep a figure of 5 1st features to see
# print(desc)
# ##! idea : for each class, extract the 3rd quantile of each ft and it is the value having under it the 75% of the population
# # count that num of samples that are less than the 3rd quartile value, in each class
# # make a table with row as ft, x cols for x classes and for each class the % of that class samples under the 3rd quartile
# # add two columns, for most and least bags
# # make a table of 4 cols, each col is a class and the genes that are over or under expressed
# # sort the cols that are
# # this gives for each ft, between both classes, where the overpression of the ft lies more
#
# #=======> 1st DATA FILTERING
# print("DATA FILTERING...")
# # ---> discarding features that we consider non informative (nif) (a unique value in a col while the response changed across samples)
# if len(RespClasses) == 0 :
# 	print("Only 1 class exist in your population. Electing non informative features is not possible.")
# 	dframe_wo_nif = dframe
# else:
# 	dframe_w_nif_cols = list(dframe)   # show the features in drection of dropping the mostly non related to class ## (see also df_cols = sdss_df.columns.values )
# 	dframe_wo_nif = dframe.drop(dframe.columns[dframe.apply(lambda col: col.nunique(dropna=True) == 1)], axis=1) # dropping the cols with num unique values = 1
# 	dframe_wo_nif_cols = list(dframe_wo_nif)
# 	nif_cols = [x for x in dframe_w_nif_cols if x not in dframe_wo_nif_cols] ##! output a list of this
# 	dframe_of_nif = dframe[nif_cols] # keep this
# 	# report on the remaining fts and samples
# 	remaining_samples = len(dframe.axes[0])
# 	remaining_feats = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
# 	if len(nif_cols) ==0:
# 		print("No feature have been taken out of the final frame due to non informativeness.")
# 	else:
# 		print(len(nif_cols),"features have been taken out of the final frame due to non informativeness.")
# 	print("In the resulting final frame",remaining_feats,"features remaining describing",remaining_samples,"samples.")
#
#
# # ##! volontarely dropping columns the operator knows not informative by prior knowledge
# # make a for loop and going through the list provided by the op, if in the dframe_wo_nif_cols we drop it with (axis=1, inplace=True)
# # ex : sdss_df_vfo.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)
#
# # dframe_wo_nif.head(1) # (optional) " only to see what 1 line of the dframe looks like now # not possible if too many fts
#
# #==========> Univariate Analysis
# # Methodology :
# # - we have 2 steps here :
# # 1) focus some 2 or 3 features
# # 2) for those features, examine the distribution of the values, deduce the range comprising the most values for each class,
# # and then knowing the role of the feature from previous knowledge, conclude that this class of the studied phenomena comes
# # with this feature role in those values (eg : a feature D is an estimate for the distance, classes are the state of blur due to the distance
# # within this list [not blurred-A, abitblurred-B, blurred-C,veryblurred-D]
# # nb :  the case where the distribution are close to same across the classes mean that this feature does not have enough classifying power for this phenomena
# # - the distplot tells us how most of each class behave for the fts D
# # - and then we can order the classes for each ft (which class of car are further, then after them which one, then after them which one)
# # - this is called distinguishing the classes just based on a column
# #-----> Way 1 : histograms for a ft for each class
# # make distribution of a variable in a class
# # formula : f(feature,class label) -> ditributuion of features's attributes in each class
# # objective : To guess a variable possible contribution in a model visualy. Also to link variablme significance to the distribution
# # (this portion of the data of class x is in k interval so it relates to this aspect in real life)
# foi = feat_cols[0] # ft of interest " chosen here as the 1st ft just as an example ##! get from the operator
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
# ax = sns.distplot(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[0]][foi], bins = 30, ax = axes[0], kde = False)
# ax.set_title(encoded_classes[0])
# ax = sns.distplot(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[1]][foi], bins = 30, ax = axes[1], kde = False)
# ax.set_title(encoded_classes[1])
# ##! add a saving option to save this figure
# ##! using a for loop, this can be produced for all features and kept away
# ##! such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests, so do it after the univariate tests rankings
# # - a table summarizing this (where are 75% of the values by class for each feature ?) using the univariate tests rankings
#
# # -------> Way 2 : LVplot (letter value plot)
# # Another way is to do univariate is that, for each ft, boxing the values ny bags and comparing the bags size
# # (base length = number of values and height is the range where the values are)
# # Interpretation 1 : similar boxing across 2 classes shows same behaviour in regards to the ft analysed
# # Interpretation 2 : ###! last stop
# fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
# ax = sns.lvplot(x=dframe_wo_nif[Resp_col_name_left], y=dframe_wo_nif[foi], palette='coolwarm')
# ax.set_title(foi)
# ##! add a saving option to save this figure
# ##! such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests, so do it after the univariate tests rankings
#
# # ----> Way 3 : We have a high number of features. Let's rank them to examine more the top ones
#
#
#
# # ======> Multivariate Analysis :
# # 1) Heatmaps for correlation between features
# # This to just get a feeling of features that separate from the rest in correlation
#
# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
# fig.set_dpi(100)
# ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[0]][dframe_wo_nif_cols[2:10]].corr(), ax = axes[0], cmap='coolwarm')
# ax.set_title(encoded_classes[0])
# ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[1]][dframe_wo_nif_cols[2:10]].corr(), ax = axes[1], cmap='coolwarm')
# ax.set_title(encoded_classes[1])
# # des correlations pas très fortes mais toujours présentes existent entre certains features qui semblent proches en denomination
# ##! Le grd nombre de feature invit à répéter cette opération après le ranking avec des univariates tests
# # Interpretation awaited 1 : a correlation space means that the features in that space in regards to their role are
# # more invested in the interplay leading to the phonemenon and sould be examinated more if the phenomenan is to be understood.
# # Interpretation awaited 1 : also, if the correlation space is th same for every class, the feature behave the same towards the classes, hence has little to no classifying power
# # therefore the feature has little to nothing to do with the interplay leading to the phenomenon
#
# # 2) multivariate for a duo of features : Plotting fetaures 2 by 2 (##! usefull after a ranked list of features is obtained)
# # Interpretation  awaited 1 : see first if the values of the couple of fts does differ between the classes.
# # this is the same than studying the classes distribution over the values of one ft in the univariate analysis, except now it is done for for 2 fts at the same time.
# ##! modify it to be for 3 fts using a surface (4th ft canot be because colour is already used to differentiate clsses)
# # The whole idea is to see if the classes generally differ accross the values of the a group of classes (two or three)
# # this is to see that if an enclosed phenomenon is controled by these 2/3 fts, the globally inquired phenomenon is not
# # varying following that enclosed one.
# ##! make a function of this and call it anytime
# foi1 = feat_cols[0]
# foi2 = feat_cols[1]
# sns.lmplot(x=foi1, y=foi2, data=dframe_wo_nif, hue=Resp_col_name_left, fit_reg=False, palette='coolwarm', size=6, aspect=2)
# plt.title('Equatorial coordinates')
#
# #====================