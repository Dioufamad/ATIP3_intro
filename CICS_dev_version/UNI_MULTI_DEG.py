# =============================== DIFFERENTLY EXPRESSED GENES INQUIRY  ========================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>INITIAL DATA ANALYSIS OPERATIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import os
import pandas as pd # for dataframes manipulation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # to change the Response values from string to classes 0 and 1 # not needed at the moment
import numpy as np # linear algebra and exploit arrays faster and easier computations
import os #for bash command lines in python
import locale
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline
from pathlib import Path # to manage paths as into arguments
from slate_engines.fs_engine import ranker_by_pval_v2
#>>>>>>>>>>>>>>>>>>>>>>>>>>> Variables to initialise------------------------------------------
print("Initialising environnement variables...")
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
# -----needed to save the output dataset
basedir = str(Path()) #for setting the working directory to create the paths to the location of the output dataset
#---how to launch the script
# tag_decision_launching_way = "cmd_line"
tag_decision_launching_way = "line_by_line"
# ----for the location of the datasets
command_center = "Gustave_Roussy"
# command_center = "Home"
# ----for the cohort choice
cohort_used = "REMAGUS02"
# cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
# -----for the plots
SMALL_SIZE = 10
MEDIUM_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150
print("All imports and settings are successfully placed")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> README <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Necessary libraries imported.")
print("Environnement variables initialised.")
print("The final dataframe (dframe) is supplied ! Now onto the data analysis...")
print("This tool can perform these following data analysis :  ")
print("- visuals on data.")
print("- univariates analysis")
print("- multivariate analysis")
print("- machine learning analysis")
print("- please read documentation joined to properly launch other possible tasks.")

#>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESSING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Reading the dataframe to analyse...")
# stock the file path and its separator
if command_center == "Gustave_Roussy" :
	file_path = "/home/amad/PycharmProjects/ATIP3_in_GR/CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R02/BRCA_Treatment11_REMAGUS02xNACx221x54675_GEX.csv" # @ GR
	sep_in_file = ","
	supporting_file_path = "/home/amad/PycharmProjects/ATIP3_in_GR/CICS/CICS_dev_version/atip3_material/table_of_treatments_details/treatments_details_source.csv" # @ GR
	# sep_in_file_sup = "," # use the one for dataset
else :  # command_center == "Home"
	file_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R02/BRCA_Treatment11_REMAGUS02xNACx221x54675_GEX.csv" # @ home
	sep_in_file = ","
	supporting_file_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/atip3_material/table_of_treatments_details/treatments_details_source.csv" # @ home
	# sep_in_file_sup = "," # use the one for dataset

##! a mock up of df making # replace later
dframe = pd.read_csv(file_path,sep_in_file) ##! add skiprows=0 to skip lines 0 lines here, default is None
dframe_sup = pd.read_csv(supporting_file_path,sep_in_file)
# needed columns : sample col, response col
Resp_col_name_left = "BestResCategory"
Samples_col_name_right = "Model" # for the right table
# displaying a report on the classes
RespBin = dframe.loc[:,[Resp_col_name_left]]  # get the 1st column of data ... # anciently it was dframe[Resp_col_name] but gives a series instead of a df
RespClasses = sorted(RespBin.iloc[:, 0].unique())
binary_classes_le = LabelEncoder()  # the encoder
binary_classes_le.fit(RespClasses)  # encode the classes to memorize
encoded_classes = binary_classes_le.classes_ ##! change it into a list to access it directly (list of cols of array, same as getting the cols of a df)
del RespBin # clear mem
print("DESCRIBING THE OBTAINED FINAL SAMPLES-FEATURES-RESPONSE FRAME...")
total_samples = len(dframe.axes[0])
total_feats = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
print("The frame to analyse has ", total_samples,"samples and ",total_feats ,"features")
print("Among",total_samples,"samples,",len(encoded_classes),"classes has been detected as being : {}.".format(' and '.join(str(class_value) for class_value in encoded_classes)))
for class_value in encoded_classes:
	class_size = dframe.iloc[:, list(dframe).index(Resp_col_name_left)].value_counts()[encoded_classes[list(encoded_classes).index(class_value)]] # before it was using dframe.iloc[:, 0]
	class_size_perc = (class_size / total_samples)*100
	print("The class value",class_value,"is found on",class_size,"samples counting for",'{:.3f}'.format(class_size_perc),"% of the samples")
# -------step 17 : giving names to the each of the 3 groups of columns to manipulate them in group using a name
# sampl_col = df_joined.columns[0] is already Samples_col_name_right
# resp_col = df_joined.columns[1] is already Resp_col_name_left
# strategy of saomples col pos : samples moved to last col or not
tag_decision_move_samples_col_at_last_pos = "yes"
# tag_decision_move_samples_col_at_last_pos = "no"
if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]: # decide where are the fts
	index_ft_one = 1
	# index_ft_last = -1
	feat_cols = dframe.columns[index_ft_one:-1]
else:
	index_ft_one = 2
	# index_ft_last = len(dframe.columns)-1
	feat_cols = dframe.columns[index_ft_one:]

##! last_stop

#---> Let's do statistics on our datasets variables
# check if in each column, the values stay in a reasonable scale and what they are :
# (missing values?,min, max, quartils for scale, mean, sd, etc. )
# - removing null values to avoid errors
# dframe.dropna(axis='columns',inplace=True) # (already done)
# - percentile list
# perc =[.25, .50, .75, .90]  ##! can be choosed later with an arguement
# - list of dtypes to include
# include =['object', 'float', 'int']    ##! to be choosed by operator   (1 argument)
# - stash the describe result
# desc = sdss_df.describe(percentiles = perc, include = include)  # used by the args for perc values and the types
desc = dframe[dframe.columns[:10]].describe() ##! to reuse with list of candidates genes in order to see how there values are
## default include=None only takes into acount the numeric as dtypes columns (very good compromise to easily capture except for a few prticular cases of samples names in numeric
# default  percentile are 0.25, 0.50 and 0.75 so quite okay to join it with min and max and see the range the values of a feature are staying
# calling describe method (optional) ##! keep a figure of 5 1st features to see
print("This is a description of the first 10 features values")
print(desc)
##! idea : for each class, extract the 3rd quantile of each ft and it is the value having under it the 75% of the population
# count that num of samples that are less than the 3rd quartile value, in each class
# make a table with row as ft, x cols for x classes and for each class the % of that class samples under the 3rd quartile
# add two columns, for most and least bags
# make a table of 4 cols, each col is a class and the genes that are over or under expressed
# sort the cols that are
# this gives for each ft, between both classes, where the overpression of the ft lies more

#=======> 1st DATA FILTERING
print("DATA FILTERING...")
# ---> discarding features that we consider non informative (nif) (a unique value in a col while the response changed across samples)
if len(RespClasses) == 0 :
	print("Only 1 class exist in your population. Electing non informative features is not possible.")
	dframe_wo_nif = dframe
else:
	dframe_w_nif_cols = list(dframe)   # show the features in drection of dropping the mostly non related to class ## (see also df_cols = sdss_df.columns.values )
	dframe_wo_nif = dframe.drop(dframe.columns[dframe.apply(lambda col: col.nunique(dropna=True) == 1)], axis=1) # dropping the cols with num unique values = 1
	dframe_wo_nif_cols = list(dframe_wo_nif)
	nif_cols = [x for x in dframe_w_nif_cols if x not in dframe_wo_nif_cols] ##! output a list of this
	dframe_of_nif = dframe[nif_cols] # keep this
	# report on the remaining fts and samples
	remaining_samples = len(dframe.axes[0])
	remaining_feats = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
	if len(nif_cols) ==0:
		print("No feature have been taken out of the final frame due to non informativeness.")
	else:
		print(len(nif_cols),"features have been taken out of the final frame due to non informativeness.")
	print("In the resulting final frame",remaining_feats,"features remaining describing",remaining_samples,"samples.")


# ##! volontarely dropping columns the operator knows not informative by prior knowledge
# make a for loop and going through the list provided by the op, if in the dframe_wo_nif_cols we drop it with (axis=1, inplace=True)
# ex : sdss_df_vfo.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

# dframe_wo_nif.head(1) # (optional) " only to see what 1 line of the dframe looks like now # not possible if too many fts

#==========> Univariate Analysis
print("UNIVARIATE ANALYSIS")
# Methodology :
# - we have 2 steps here :
# 1) focus some 2 or 3 features
# 2) for those features, examine the distribution of the values, deduce the range comprising the most values for each class,
# and then knowing the role of the feature from previous knowledge, conclude that this class of the studied phenomena comes
# with this feature role in those values (eg : a feature D is an estimate for the distance, classes are the state of blur due to the distance
# within this list [not blurred-A, abitblurred-B, blurred-C,veryblurred-D]
# nb :  the case where the distribution are close to same across the classes mean that this feature does not have enough classifying power for this phenomena
# - the distplot tells us how most of each class behave for the fts D
# - and then we can order the classes for each ft (which class of car are further, then after them which one, then after them which one)
# - this is called distinguishing the classes just based on a column
#-----> Way 1 : histograms for a ft for each class
# make distribution of a variable in a class
# formula : f(feature,class label) -> ditributuion of features's attributes in each class
# objective : To guess a variable possible contribution in a model visualy. Also to link variablme significance to the distribution
# (this portion of the data of class x is in k interval so it relates to this aspect in real life)
print("Univariate Analysis : histograms for the ditribution of a features's attributes in each class (Figure 1)")
foi = feat_cols[0] # ft of interest " chosen here as the 1st ft just as an example ##! get from the operator
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
ax = sns.distplot(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[0]][foi], bins = 30, ax = axes[0], kde = False)
ax.set_title(encoded_classes[0])
ax = sns.distplot(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[1]][foi], bins = 30, ax = axes[1], kde = False)
ax.set_title(encoded_classes[1])
##! add a saving option to save this figure
##! using a for loop, this can be produced for all features and kept away
##! such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests, so do it after the univariate tests rankings
# - a table summarizing this (where are 75% of the values by class for each feature ?) using the univariate tests rankings

# -------> Way 2 : LVplot (letter value plot)
# Another way is to do univariate is that, for each ft, boxing the values ny bags and comparing the bags size
# (base length = number of values and height is the range where the values are)
# Interpretation 1 : similar boxing across 2 classes shows same behaviour in regards to the ft analysed
# Interpretation 2 : ###! last stop
print("Univariate Analysis : LVplot (letter value plot) for classes behaviour in regards to a feature (Figure 2)")
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
ax = sns.lvplot(x=dframe_wo_nif[Resp_col_name_left], y=dframe_wo_nif[foi], palette='coolwarm')
ax.set_title(foi)
##! add a saving option to save this figure
##! such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests, so do it after the univariate tests rankings

# ----> Way 3 : We have a high number of features. Let's rank them to examine more the top ones



# ======> Multivariate Analysis :
print("MULTIVARIATE ANALYSIS")
# 1) Heatmaps for correlation between features
# This to just get a feeling of features that separate from the rest in correlation
print("Multivariate Analysis : Heatmaps for correlation between features (Figure 3)")
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[0]][dframe_wo_nif_cols[index_ft_one:10]].corr(), ax = axes[0], cmap='coolwarm') #1 or 2 if samples col not at end or not
ax.set_title(encoded_classes[0])
ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[1]][dframe_wo_nif_cols[index_ft_one:10]].corr(), ax = axes[1], cmap='coolwarm')
ax.set_title(encoded_classes[1])
# des correlations pas très fortes mais toujours présentes existent entre certains features qui semblent proches en denomination
##! Le grd nombre de feature invit à répéter cette opération après le ranking avec des univariates tests
# Interpretation awaited 1 : a correlation space means that the features in that space in regards to their role are
# more invested in the interplay leading to the phonemenon and sould be examinated more if the phenomenan is to be understood.
# Interpretation awaited 1 : also, if the correlation space is th same for every class, the feature behave the same towards the classes, hence has little to no classifying power
# therefore the feature has little to nothing to do with the interplay leading to the phenomenon

# 2) multivariate for a duo of features : Plotting fetaures 2 by 2 (##! usefull after a ranked list of features is obtained)
# Interpretation  awaited 1 : see first if the values of the couple of fts does differ between the classes.
# this is the same than studying the classes distribution over the values of one ft in the univariate analysis, except now it is done for for 2 fts at the same time.
##! modify it to be for 3 fts using a surface (4th ft canot be because colour is already used to differentiate clsses)
# The whole idea is to see if the classes generally differ accross the values of the a group of classes (two or three)
# this is to see that if an enclosed phenomenon is controled by these 2/3 fts, the globally inquired phenomenon is not
# varying following that enclosed one.
##! make a function of this and call it anytime
print("Multivariate Analysis : Equatorial coordinates to see if the values of a couple of features differ between the classes (Figure 4)")
foi1 = feat_cols[0]
foi2 = feat_cols[1]
sns.lmplot(x=foi1, y=foi2, data=dframe_wo_nif, hue=Resp_col_name_left, fit_reg=False, palette='coolwarm', size=6, aspect=2)
plt.title('Equatorial coordinates')

# 3) FS
print("Multivariate Analysis : features ranking with Student t-test p-values (Table 1)")
if tag_decision_move_samples_col_at_last_pos in ["yes", "y"]:
	il1_train_x = dframe_wo_nif[list(dframe_wo_nif)[index_ft_one:-1]]
else:
	il1_train_x = dframe_wo_nif[list(dframe_wo_nif)[index_ft_one:]]
il1_train_y = dframe_wo_nif.loc[:, [Resp_col_name_left]]
feature_val_type = "real"
Resp_col_name = Resp_col_name_left
# encoded_classes # already exists
# the ranking
il1_fts_ranking = ranker_by_pval_v2(il1_train_x, il1_train_y, feature_val_type, Resp_col_name,encoded_classes)




# restrict the material to build the metric not train and test here but the list of selected fts for the omc in the ranking

# -----lets try to document the fts that had good enough pvalue to be part of the FS
max_omc_il1 = int(np.ceil(float(len(dframe_wo_nif)) / 2))
pval_last_feat_of_omc = il1_fts_ranking[2][(max_omc_il1-1)]
list_pvals_for_fts_correlated_enough_to_response = [a_pval for a_pval in il1_fts_ranking[2] if a_pval <= pval_last_feat_of_omc]
list_fts_correlated_enough_to_response = il1_fts_ranking[1][:len(list_pvals_for_fts_correlated_enough_to_response)]
# # add the list of fts ranked to a collector to later compute the persistent accross seeds and the non persistent
# all_seeds_col_of_list_fts_correlated_enough_to_response.append(list_fts_correlated_enough_to_response)
# - creating the df to receive ["Seed","pval","feat ranked"] 2nd col content before the 1st col because the 1st col uses it to be created
df_of_fts_correlated_enough_to_response_by_seed = pd.DataFrame()
# content_Seed_column = np.repeat(aseed, len(list_pvals_for_fts_correlated_enough_to_response))  # the longest column is the indexes in the fold column so start from it
content_Pvals_column = list_pvals_for_fts_correlated_enough_to_response # pretty straight forward as it is just the pvals
content_Pvals_column_as_str = [str(val) for val in content_Pvals_column]
content_FeatsRanked_column = list_fts_correlated_enough_to_response # it is the list of the feats but if it did not make the FS cut mark it as such
content_rank_column = list(range(1, (max_omc_il1 + 1)))

df_ranking4FS = pd.DataFrame(list(zip(content_rank_column, content_FeatsRanked_column, content_Pvals_column_as_str)), columns =['Rang','Features_ranked', 'Pval_of_univ_stat'])

# -------step 20 : Save a copy of the final dframe  ##! last_stop this was okay and working, make it plush and hot launchable in CICS
print("The feature ranking final dataframe (dframe) is ready ! Lets save it in a .csv file...")
tag_ctype = "BRCA"
tag_drugname = "REMAGUS02_NAC" # manually recordd in the treatments details files
tag_drugID = "Treatment11"
tag_respType = str(cohort_used) + "x" + "NAC"+ "x" + str(total_samples) + "x" + str(total_feats) + "xFSranking"
tag_profilename = "GEX"
# the output path has 3 parts : the root until the ouput folder, the output folder, and the filename
# - lets make the file name
output_filename_for_final_dframe = tag_ctype + "_" + tag_drugID + "_" + tag_respType + "_" + tag_profilename + ".csv"
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
df_ranking4FS.to_csv(fullname, index=None, header=True)
print("File saved !")
print(cohort_used,"dataset formatting for CICS analysis is done!")


#==================








# #...building the last dataframe (across all seeds) and appending it to the col of the dfs for across the seeds extended lists of feats selected
# df_of_persistent_or_not_feats_accross_seeds = pd.DataFrame({'Seed': pd.Series(accross_seeds_frame_content_allsseeds_column), 'Features_ranked': pd.Series(accross_seeds_frame_content_persistent_column), 'Pval_of_univ_stat': pd.Series(accross_seeds_frame_content_nonpersistent_column)})
# all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed.append(df_of_persistent_or_not_feats_accross_seeds)
# # the df for all extended feats lists across the seeds
# df_of_fts_correlated_enough_to_response_across_all_seeds = pd.concat(all_seeds_col_of_df_of_fts_correlated_enough_to_response_by_seed)
# # lets make up a filename for the extended FS and then use it to create the .csv file
# output_filename_for_ExtFS_omc_mdl = basedir + "/" + "outputs" + "/" + "Output_" + tag_task_type + "_" + tag_alg + "-" + models_compared[0] + "_" + tag_ctype + "-" + tag_drugname + "-" + tag_profilename + "_" + tag_num_trial + "_ExtFS.csv"
# df_of_fts_correlated_enough_to_response_across_all_seeds.to_csv(output_filename_for_ExtFS_omc_mdl, index=None, header=True)