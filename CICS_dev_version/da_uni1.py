# =============================== DATA_ANALYSIS_UNIVARIATES_V1.0.0 ========================================
#>>>>>>>>>>>>>>>>>>>>>>>>>>>INITIAL DATA ANALYSIS OPERATIONS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print("We suppose you have done the querying the database and you have separated values files (csv,tsv,etc.).")
print("We will try to have a complete cycle of the data analysis including :  ")
print("- visuals on data.")
print("- univariates analysis")
print("It is followed usually by building machine learning models to predict for new data.")
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline

SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150
print("all imports and setings are successfully placed")

# ---> how to stock a dataset
sdss_df = pd.read_csv('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/SLATE/SLATE_dev_version/slate_data/datasets_to_process_folder/real_val_prof_test/BRCA_Treatment17_BestResCat_GEX.csv', skiprows=0)
# ---> Let's take a first look at our dataset to see what we're working with!
sdss_df.head()
# ----> Let's find out about the data types we have accross columns :
sdss_df.info()
# the awaited configuration is to have one column of type object that is ou response, rest is int64 or float64
# the samples names col can be discarded or be any type and we will just keep it like that

#---> Let's do statistics on our datasets variables (missing values?,min, max, quartils for scale, mean, sd, etc. )
# removing null values to avoid errors
sdss_df.dropna(axis='columns',inplace=True)
# percentile list
# perc =[.25, .50, .75, .90]  ##! can be choosed later with an arguement
# list of dtypes to include
# include =['object', 'float', 'int']    ##! to be choosed by operator   (1 argument)
# stash the describe result
# desc = sdss_df.describe(percentiles = perc, include = include)  # used by the args for perc values and the types
desc = sdss_df.describe()
## default include=None only takes into acount the numeric as dtypes columns (very good compromise to easily capture except for a few prticular cases of samples names in numeric
# default  percentile are 0.25, 0.50 and 0.75 so quite okay to join it with min and max and see the range the values of a feature are staying
# calling describe method (optional) ##! keep a figure of 5 1st features to see
desc
####testing the presence of nan (no need because of nan columns dropped)
# sdss_df_w_nan = sdss_df
# sdss_df_w_nan.iloc[5,5] = np.nan
# desc_w_nan = sdss_df_w_nan.describe(percentiles = perc, include = include)
# desc_w_nan
# we choosed to remove all nan values to avoid imputing
# we could check if in each column, the values stay in a reasonable scale (min, max and quartiles values in the same order)
# ----> Let's do a count of instances in each class
sdss_df[list(sdss_df)[0]].value_counts() ##! use the column name given by user as being the class and show something more nice with percentages of each class
#=======> 1st DATA FILTERING
# ---> discarding features that we consider non informative (a unique value in a col while the response changed across samples)
df_cols_w_uniques = list(sdss_df)   # show the features in drection of dropping the mostly non related to class ## (see also df_cols = sdss_df.columns.values )
sdss_df_wo_uniques = sdss_df.drop(sdss_df.columns[sdss_df.apply(lambda col: col.nunique(dropna=True) == 1)], axis=1)
df_cols_wo_uniques = list(sdss_df_wo_uniques)
cols_of_uniques = [x for x in df_cols_w_uniques if x not in df_cols_wo_uniques]
df_of_uniques = sdss_df[cols_of_uniques]
# sdss_df_vfo.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True) ##! change the inside list to fit the choice of the operator via argument

# sdss_df_wo_uniques.head(1) # (optional)

#==========> Univariate Analysis
# Methodology :
# - we have 2 steps here :
# 1) focus some 2 or 3 features
# 2) for those features, examine the distribution of the values, deduce the range comprising the most values for each class,
# and then knowing the role of the feature from previous knowledge, conclude that this class of the studied phenomena comes
# with this feature role in those values (eg : a feature D is a distance, a class is "the car is blurred or not", the blur occur
# while high values of D, the blur occurs when the car is far.
#-----> Way 1 : histograms
# make distribution of a variable in a class
# formula : f(feature,class label) -> ditributuion of features's attributes in each class
# objective : To guess a variable possible contribution in a model visualy. Also to link variablme significance to the distribution
# (this portion of the data of class x is in k interval so it relates to this aspect in real life)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
ax = sns.distplot(sdss_df_wo_uniques[sdss_df_wo_uniques['BestResCategory']=='Res'].A1BG, bins = 30, ax = axes[0], kde = False)
ax.set_title('Res')
ax = sns.distplot(sdss_df_wo_uniques[sdss_df_wo_uniques['BestResCategory']=='Sen'].A1BG, bins = 30, ax = axes[1], kde = False)
ax.set_title('Sen')
##! add a saving option
# the case where the distribution are close to same across the classes mean that this feature does not have enough classifying power for this phenomena
##! using a for loop, this can be produced for all features and kept away
##a table summarizing this (where are 75% of the values by class for each feature ?) but their is better : using the univariate tests rankings

# -------> Way 2 : LVplot (letter value plot)
# Another way is to do univariate boxing the values and comparing the boxes dimensions
# (base length = number of values and height is the range where the values are)
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
ax = sns.lvplot(x=sdss_df_wo_uniques['BestResCategory'], y=sdss_df_wo_uniques['A1BG'], palette='coolwarm')
ax.set_title('A1BG')
##! add a saving option
## once again, such a figure would be more interesting if done for features that are supposed to be the most contributing to the tests
## so do it after the univariate tests rankings
# ----> Way 3 : We have a high number of features. Let's rank them to examine more the top ones


# ======> Multivariate Analysis :
# 1) Heatmaps for correlation between features
# This to just get a feeling of features that separate from the rest in correlation

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sdss_df_wo_uniques[sdss_df_wo_uniques['BestResCategory']=='Res'][df_cols_wo_uniques[1:10]].corr(), ax = axes[0], cmap='coolwarm')
ax.set_title('Res')
ax = sns.heatmap(sdss_df_wo_uniques[sdss_df_wo_uniques['BestResCategory']=='Sen'][df_cols_wo_uniques[1:10]].corr(), ax = axes[1], cmap='coolwarm')
ax.set_title('Sen')
# des correlations pas très fortes mais toujours présentes existent entre certains features qui semblent proches en denomination
# Le grd nombre de feature invit à répéter cette opération après le ranking avec des univariates tests
# Interpretation awaited : a correlation space means that the features in that space in regards to their role are
# more invested in the interplay and sould be disturbd if the phenomenan is tested. If the correlation space is th same
# for every class, the feature behave the same towards the classes, hence has little to no clssifying power

# 2) multivariate for a duo of features : Plotting fetaures 2 by 2 (##! usefull after a ranked list of features is obtained)
sns.lmplot(x='A2M', y='A2ML1', data=sdss_df_wo_uniques, hue='BestResCategory', fit_reg=False, palette='coolwarm', size=6, aspect=2)
plt.title('Equatorial coordinates')








