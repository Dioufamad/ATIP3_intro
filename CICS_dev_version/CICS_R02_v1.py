# =============================== DIFFERENTLY EXPRESSED GENES INQUIRY  ========================================
#>>>>>>>>>>>>>>>>>>>>>>>>>>>INITIAL DATA ANALYSIS OPERATIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#>>>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#>>>>>>>>>>>>>>>>>>>>>>>>>>> README <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Welcome in Case Implicated Candidates Search (CICS)")
print("We suppose you have done the querying of the database and you have separated values files (csv,tsv,etc.).")
print("Such values tables describe samples over multiples features, rows samples and features as columns or vice-versa.")
print("We will try to realise a search of the features that are differently varying following a response.")
print("Such features are the candidates we search for...")
print("Importing necessary libraries...")
#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

# ---------------------Variables to initialise------------------------------------------
print("Initialising environnement variables...")
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
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
#>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PREPROCESSING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("Preprocessing...")
# the file we are given a table with the intent to represent values, each one corresponding to a variable and a sample
# the vision is one of these 2 representations : variables as columns and rows as samples, or vice-versa

# 1st issue : the file might not be in a supported encoding so we have to reencode it in UTF-8

# file -i REMAGUS02-Données\ genomique_226x54676\ totales.txt
# iconv -f UTF-16LE -t UTF-8//IGNORE REMAGUS02-Données\ genomique_226x54676\ totales.txt > output2.tsv

# stock the file and its separator
if command_center == "Gustave_Roussy" :
	file_path = "/home/amad/PycharmProjects/ATIP3_in_GR/CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS02_Donnees_genomiques_226x54676_totales.tsv" # @ GR
	sep_in_file = "\t"
	supporting_file_path = "/home/amad/PycharmProjects/ATIP3_in_GR/CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS02-Données cliniques.xls" # @ GR
	sheet_id = "extractionCNahmias"
else :  # command_center == "Home"
	file_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/atip3_material/3c_data_trial1/tsv/REMAGUS02_Donnees_genomiques_226x54676_totales.tsv" # @ home
	sep_in_file = "\t"
	supporting_file_path = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/atip3_material/3c_data_trial1/support/REMAGUS02-Données cliniques.xls" # @ home
	sheet_id = "extractionCNahmias"

#>>>>>-----start of everything that is data fil specific in the preprocessing
if cohort_used == "REMAGUS02" :
	#make a df out of the file (# ---> how to stock a dataset)
	df_file = pd.read_csv(file_path,sep_in_file) ##! add skiprows=0 to skip lines 0 lines here, default is None
	df_sup_file = pd.read_excel(supporting_file_path,sheet_id)
	# restricting the support info table to only the needed columns
	# needed columns : sample col, response col
	# NB : the 2 samples columns must have different names to join them later
	old_Samples_col_name_left = "CLETRI" # given cols name by the user ##! both got from the arguments
	Samples_col_name_left = "Sample_id_bis" # for the left table
	old_Resp_col_name_left = "RCH"
	Resp_col_name_left = "Resp_Class" # set cols name by the tool for future dealings
	old_Samples_col_name_right = "cletri"
	Samples_col_name_right = "Sample_id" # for the right table
	common_samples_id_prefix = "CLETRI"
	# Objective : joining both tables (the one with the features and the one with the response)
	# -----> put in form the content of each table

	# - put in form the content of response table (LEFT)
	df_sup_file = df_sup_file[[old_Samples_col_name_left,old_Resp_col_name_left]]
	df_sup_file.rename(columns={old_Samples_col_name_left: Samples_col_name_left, old_Resp_col_name_left : Resp_col_name_left}, inplace=True)
	df_sup_file.dropna(axis='index', inplace=True) # lets make sure the load out going to the left is without nan
	df_sup_file[Samples_col_name_left] = common_samples_id_prefix + "_" + df_sup_file[Samples_col_name_left].astype(str) # sample_name is a string ColumnName_IdInColumn
	# df_sup_file[Resp_col_name_left] = df_sup_file[Resp_col_name_left].astype(int) ##! remove bcuz not needed, iz done later on after the join

	# - put in form the content of features table (RIGHT)
	df_file = df_file.transpose() # changes columns into rows
	# make the index (presently being the sample names) as an index
	df_file = df_file.reset_index() # reset the index in a way to get the older index as a column
	df_file.columns = df_file.iloc[0] # take the first line and use it as titles of the columns
	df_file = df_file.drop(df_file.index[0]) # drop the first line because it is now the titles of the columns
	df_file = df_file.reset_index(drop=True) # the index is missing now a the 1st line and is starting by 1 instead of 0. reset it in a way to not get a new column
	# dropping columns that are not necessary
	list_of_unecessary_cols_2_drop = ["CLETRI"]
	df_file.drop(labels=list_of_unecessary_cols_2_drop, axis=1, inplace=True) # dropping a column that is just a repetiton of the sample names col
	# renaming the sample column
	df_file.rename(columns={old_Samples_col_name_right: Samples_col_name_right}, inplace=True)
	df_file[Samples_col_name_right] = common_samples_id_prefix + "_" + df_file[Samples_col_name_right].astype(str) # sample_name is a string ColumnName_IdInColumn

	# - time for the joining of the features and responses table (joined on the sample name columns)
	df_joined = pd.merge(df_sup_file, df_file, how="inner", left_on=Samples_col_name_left, right_on=Samples_col_name_right)
elif cohort_used == "REMAGUS04" :
	print("dataset treatment to add")
elif cohort_used == "MDAnderson" :
	print("dataset treatment to add")
else :
	print("no dataset known added for treatment")
#<<<<<<<----end of all that is data file specific about the preprocessing

# >>>>>---------formatting every group of columns in the final frame
# the awaited configuration is :
# - 1 column of dtype object (samples names)
# - 1 column that can have anything as dtype (the response) and that is why we encode it
# - and a bunch that is int64/float64/object but the one and the same type that we previously formatted in bools or floats

# dropping the extra sample column
df_joined.drop(labels=[Samples_col_name_left], axis=1, inplace=True)
del df_file # clear memory
del df_sup_file # clear memory
# store the resp col that is last, drop it from the df and then insert it again at the 2nd position of the df
Resp_col_to_move = df_joined[Resp_col_name_left]
df_joined.drop(labels=[Resp_col_name_left], axis=1, inplace=True)
df_joined.insert(1, Resp_col_name_left, Resp_col_to_move)
del Resp_col_to_move # clear memory
# drop all rows with nan (in R02, b4 : 221x54677 aft : 221x54677)
# df_joined = df_joined.dropna(axis='index') # less efficient
df_joined.dropna(axis='index',inplace = True) ##! choose if you lose samples or fts
# keep a copy of the raw final df
# df_joined_res = df_joined
# use this to get a peak at the dtypes in the final dataframe
# df_joined[df_joined.columns[:10]].dtypes


# giving names to the each of the 3 groups of columns to manipulate them in group using a name
# sampl_col = df_joined.columns[0] is already Samples_col_name_right
# resp_col = df_joined.columns[1] is already Resp_col_name_left
feat_cols = df_joined.columns[2:]
# formatting the dtypes of each group of columns
df_joined[Samples_col_name_right] = df_joined[Samples_col_name_right].astype(str) ##! not needed if already formatted in df_left # strings dtype is object
df_joined[Resp_col_name_left] = df_joined[Resp_col_name_left].astype(int) ##! not needed if already formatted in df_left  # or df_joined["RCH"] = df_joined["RCH"].astype("int")
# df_joined[feat_cols] = df_joined[feat_cols].transform(lambda x: x.str.replace(',','.')) # slow and complex # replace the commas blocking the conversion of objects in floats
df_joined[feat_cols] = df_joined[feat_cols].replace(",", ".", regex=True)
# meth 6 used to change all fts values into floats
old_fts_col_names = df_joined[feat_cols].columns
df_fts_as_series = df_joined[feat_cols].values.astype(np.float64)
df_fts_back_as_df = pd.DataFrame(df_fts_as_series)
df_fts_back_as_df.columns = old_fts_col_names
# instead of putting the galerie of df_fts back in the df_joined, just take the 2 remaining cols and add them to it
df_fts_back_as_df.insert(0, Samples_col_name_right, df_joined[Samples_col_name_right])
df_fts_back_as_df.insert(1, Resp_col_name_left, df_joined[Resp_col_name_left])
dframe = df_fts_back_as_df
del df_fts_back_as_df
# df_fts_changed_as_df[df_fts_changed_as_df.columns[:10]].dtypes
dframe.info() # for an overall summary of remaining dtypes

#cleaning out the coerced values and reporting on the loss due to formatting the fts dtypes
samples_b4_coercing = len(dframe.axes[0])
fts_b4_coercing = len(dframe.axes[1])-2
dframe.dropna(axis='index',inplace = True) # removing null values to avoid errors
samples_aft_coercing = len(dframe.axes[0])
fts_aft_coercing = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
lost_samples = samples_b4_coercing - samples_aft_coercing
lost_fts = fts_b4_coercing - fts_aft_coercing
print("Report on the losses during the formatting of the features data types : ")
if lost_samples==0:
	print("No samples has been lost")
else:
	print(lost_samples,"samples has been lost")
if lost_fts==0:
	print("No features has been lost")
else:
	print(lost_fts,"samples has been lost")

# formatting the response column, ordering it by class, displaying a report on the classes
RespBin = dframe.loc[:,[Resp_col_name_left]]  # get the 1st column of data ... # anciently it was dframe[Resp_col_name] but gives a series instead of a df
RespClasses = sorted(RespBin.iloc[:, 0].unique())
binary_classes_le = LabelEncoder()  # the encoder
binary_classes_le.fit(RespClasses)  # encode the classes to memorize
encoded_classes = binary_classes_le.classes_ ##! change it into a list to access it directly (list of cols of array, same as getting the cols of a df)
del RespBin # clear mem
# del RespClasses # clear mem
# sort the dataframe entries following response col values and remake a new index
# sort the df following the values of the resp column
dframe.sort_values(Resp_col_name_left, axis=0, ascending=True, inplace=True, kind='mergesort')
# after the precedent sort, the indexes are not in order. make a new order for them
dframe = dframe.reset_index(drop="True")
print("Describing the obtained final samples-features-response frame...")
total_samples = len(dframe.axes[0])
total_feats = len(dframe.axes[1])-2 # withdraw of the total the samples and the response col
print("The frame to analyse has ", total_samples,"samples and ",total_feats ,"features")
print("Among",total_samples,"samples,",len(encoded_classes),"classes has been detected as being : {}.".format(' and '.join(str(class_value) for class_value in encoded_classes)))
for class_value in encoded_classes:
	class_size = df_joined.iloc[:, 1].value_counts()[encoded_classes[list(encoded_classes).index(class_value)]]
	class_size_perc = class_size / total_samples
	print("The class value",class_value,"is found on ",class_size,"samples counting for ",class_size_perc,"of the lot")


print("The final dataframe (dframe) is ready ! Now onto the proper data analysis...")
##! also delete all the uneccesary variables got sooner

# ---> Let's take a first look at our dataset to see what we're working with!
# dframe[dframe.columns[:10]].head()
# ----> Let's find out about the data types we have accross columns :
# dframe.info()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>INITIAL DATA ANALYSIS OPERATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("We will try to have a complete cycle of the data analysis including :  ")
print("- visuals on data.")
print("- univariates analysis")
print("- multivariate analysis")
print("- machine learning analysis")
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
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
ax = sns.lvplot(x=dframe_wo_nif[Resp_col_name_left], y=dframe_wo_nif[foi], palette='coolwarm')
ax.set_title(foi)
##! add a saving option to save this figure
##! such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests, so do it after the univariate tests rankings

# ----> Way 3 : We have a high number of features. Let's rank them to examine more the top ones



# ======> Multivariate Analysis :
# 1) Heatmaps for correlation between features
# This to just get a feeling of features that separate from the rest in correlation

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[0]][dframe_wo_nif_cols[2:10]].corr(), ax = axes[0], cmap='coolwarm')
ax.set_title(encoded_classes[0])
ax = sns.heatmap(dframe_wo_nif[dframe_wo_nif[Resp_col_name_left]==encoded_classes[1]][dframe_wo_nif_cols[2:10]].corr(), ax = axes[1], cmap='coolwarm')
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
foi1 = feat_cols[0]
foi2 = feat_cols[1]
sns.lmplot(x=foi1, y=foi2, data=dframe_wo_nif, hue=Resp_col_name_left, fit_reg=False, palette='coolwarm', size=6, aspect=2)
plt.title('Equatorial coordinates')

