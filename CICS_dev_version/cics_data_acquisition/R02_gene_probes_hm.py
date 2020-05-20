#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap # to wrap plot titles

#>>>>>>>>>>>>>>>>>>>Choosing the environnement values
# ----for the task to do
task_2do = "heatmap_w_all_probes_of_a_gene"
# ----for the response strategy
resp_used = "RCH3HSall"
resp_used_in_full = "All the samples with -defined or not RCH and 3 hormonals status, are kept"
# resp_used = "RCH3HSdefined"
# resp_used_in_full = "Only defined -RCH and the 3 hormonals status- samples are kept"
# resp_used = "RCHdefined"
# resp_used_in_full = "Only defined RCH samples are kept"
# resp_used = "TNBCdefined"
# resp_used_in_full = "Only defined TNBC samples are kept"
# resp_used = "RCHandTNBCdefined"
# resp_used_in_full = "Only defined RCH and TNBC samples are kept"
print("For population restriction, the response strategy chosen is",resp_used_in_full,"(",resp_used,")...")
# ----for the cohorts to implicate
cohort_used = "REMAGUS02"
# cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
#>>>>>>>>Setting up the environnement values
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
if command_center == "Gustave_Roussy":
	rest_of_abs_path_b4_content_root = "/home/amad/PycharmProjects/ATIP3_in_GR/"
else : # command_center = "Home"
	rest_of_abs_path_b4_content_root = "/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/"
print("command center used recognized...")
# ----making the dataset to use when the cohort(s) to manipulate is(are) known
#----paths to files of populations in cohorts
R02_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R02/"
R04_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/R04/"
MDA_ds_folder_path = rest_of_abs_path_b4_content_root + "CICS/CICS_dev_version/atip3_material/datasets_to_process_folder/MDA/"
if resp_used == "RCH3HSall":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx226Sx54675Fx4RasRCH3HSall_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx142Sx22277Fx4RasRCH3HSall_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx4RasRCH3HSall_GEX.csv"
elif resp_used == "RCH3HSdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx218Sx54675Fx4RasRCH3HSdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx4RasRCH3HSdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx129Sx22283Fx4RasRCH3HSdefined_GEX.csv"
elif resp_used == "RCHdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx221Sx54675Fx1RasRCHdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx1RasRCHdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx1RasRCHdefined_GEX.csv"
elif resp_used == "TNBCdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx226Sx54675Fx1RasTNBCdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx142Sx22277Fx1RasTNBCdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx1RasTNBCdefined_GEX.csv"
else:  # resp_used == "RCHandTNBCdefined":
	file_path_R02 = R02_ds_folder_path + "BRCA_Treatment11_REMAGUS02xNACx221Sx54675Fx2RasRCHandTNBCdefined_GEX.csv"
	file_path_R04 = R04_ds_folder_path + "BRCA_Treatment12_REMAGUS04xNACx139Sx22277Fx2RasRCHandTNBCdefined_GEX.csv"
	file_path_MDA = MDA_ds_folder_path + "BRCA_Treatment13_MDAndersonxNACx133Sx22283Fx2RasRCHandTNBCdefined_GEX.csv"
#---making the base dataset to use
sep_in_file = ","
if cohort_used == "REMAGUS02":
	df_file_R02 = pd.read_csv(file_path_R02, sep_in_file)
	df_file_R02.insert(len(list(df_file_R02.columns)), 'Cohort', 'Remagus02')
	# df_file_R02 = df_file_R02[list_of_cols_to_keep]
	df_file = df_file_R02
elif cohort_used == "REMAGUS04":
	df_file_R04 = pd.read_csv(file_path_R04, sep_in_file)
	df_file_R04.insert(len(list(df_file_R04.columns)), 'Cohort', 'Remagus04')
	# df_file_R04 = df_file_R04[list_of_cols_to_keep]
	df_file = df_file_R04
else : # cohort_used == "MDAnderson":
	df_file_MDA = pd.read_csv(file_path_MDA, sep_in_file)
	df_file_MDA.insert(len(list(df_file_MDA.columns)), 'Cohort', 'MDAnderson')
	# df_file_MDA = df_file_MDA[list_of_cols_to_keep]
	df_file = df_file_MDA
#---restrict the dataset to the columns of interest
# - Finding the columns for the probes of ATIP3
# they all have MTUS1 in the colname (they are 4 for R02 and 3 for R04 and MDA)
# a list of the column of the probes
full_list_of_cols = list(df_file.columns)
list_of_probes2keep_colnames = []
for colname in full_list_of_cols:
    if "MTUS1" in colname:
        list_of_probes2keep_colnames.append(colname)
sorted_list_of_unik_probes2keep_colnames = sorted(set(list_of_probes2keep_colnames))
sorted_list_of_unik_probes2keep_colnames = sorted(sorted_list_of_unik_probes2keep_colnames, key = lambda x: str(x.split("wPSIas")[1]))
print("the following columns have been found as probes for ATIP3 in",cohort_used," (response strategy is",resp_used,") : ")
for elt in sorted_list_of_unik_probes2keep_colnames:
    print("- ",elt)

# ----for the columns to keep in all 3 cohorts df
# we will keep the responses, the cohort, the samples, the probes
if resp_used == "RCH3HSall":
	list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2","Model", "Cohort"] + sorted_list_of_unik_probes2keep_colnames
	list_of_samplescol_and_1respcol = ["BestResCat_as_RCH","Model"]
	resp_of_interest = "BestResCat_as_RCH"
elif resp_used == "RCH3HSdefined":
	list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2","Model", "Cohort"] + sorted_list_of_unik_probes2keep_colnames
	list_of_samplescol_and_1respcol = ["BestResCat_as_RCH", "Model"]
	resp_of_interest = "BestResCat_as_RCH"
elif resp_used == "RCHdefined":
	list_of_cols_to_keep = ["BestResCat_as_RCH","Model", "Cohort"] + sorted_list_of_unik_probes2keep_colnames
	list_of_samplescol_and_1respcol = ["BestResCat_as_RCH", "Model"]
	resp_of_interest = "BestResCat_as_RCH"
elif resp_used == "TNBCdefined":
	list_of_cols_to_keep = ["BestResCat_as_TNBC","Model", "Cohort"] + sorted_list_of_unik_probes2keep_colnames
	list_of_samplescol_and_1respcol = ["BestResCat_as_TNBC", "Model"]
	resp_of_interest = "BestResCat_as_TNBC"
else : # resp_used == "RCHandTNBCdefined":
	list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_TNBC","Model", "Cohort"] + sorted_list_of_unik_probes2keep_colnames
	list_of_samplescol_and_1respcol = ["BestResCat_as_RCH", "Model"]
	resp_of_interest = "BestResCat_as_RCH"

# ---- making the df with full info to use
df_file_full_info = df_file[list_of_cols_to_keep]
df_file_probes_info1 = df_file_full_info.set_index("Model")
df_file_probes_info2 = df_file_probes_info1[sorted_list_of_unik_probes2keep_colnames]


#--------the clustering heatmapodel th for R02
# import seaborn as sns; done already
sns.set(color_codes=True)
series_samples_resp = df_file_probes_info1.pop(resp_of_interest) # isolate in a series the samples names and the resp to mark the samples later with a color following the resp they have
# - heatmap to build options
# - Add colored labels to identify observations:
dict_1sample_1color = dict(zip(series_samples_resp.unique(), "rbg"))
samples_colors_for_resp = series_samples_resp.map(dict_1sample_1color)
#-the hm
g = sns.clustermap(df_file_probes_info2, figsize=(18, 12),
                   row_cluster=True,col_cluster=False, method="average", metric="correlation", row_colors=samples_colors_for_resp,
				   z_score=1, standard_scale=None,robust=True,
				   )
# dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4), cmap = "vlag"/"mako", cmap = "coolwarm",cmap="viridis"
# robust=True #use outlier detection (False to not use it)

# plt.xlabel("Clustering heatmap for the sample-to-sample distances using all four probes of MTUS1 gene (proxy for ATIP3) expression values", fontsize=6)
# plt.ylabel("Numbers in bins", fontsize=18)
# plt.title("\n".join(wrap("Clustering heatmap for the sample-to-sample distances using all four probes of MTUS1 gene (proxy for ATIP3) expression values")), fontsize=12)
# plt.legend(title="Cohorts exlored", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/amad/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/ClusteringHeatmapThR02.png')


# >>>>>>>>>>>method of drawing clustering heatmaps (Plot a matrix dataset as a hierarchically-clustered heatmap)
## see : https://seaborn.pydata.org/generated/seaborn.clustermap.html
# - Plot a clustered heatmap:
# import seaborn as sns; done already
sns.set(color_codes=True)
iris = sns.load_dataset("iris")
species = iris.pop("species") # removes the columns not used like the categorical ones
g = sns.clustermap(iris)
# - Change the size and layout of the figure:
g = sns.clustermap(iris,
                   figsize=(7, 5),
                   row_cluster=False,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(0, .2, .03, .4))
# figsize=(7, 5), # (width,height of overall figure)
# row_cluster=False, # (row,col)_cluster is used to activate the clustering on the rows or cols
# dendrogram_ratio=(.1, .2), # {dendrogram,colors}_ratio: float, or pair of floats. Proportion of the fig size devoted to the two marginal elements. If a pair is given, correspond to (row, col) ratios.
# cbar_pos=(0, .2, .03, .4)) # (left, bottom, width, height).Position of the colorbar axes in the figure. None will disable the colorbar.
# - Add colored labels to identify observations:
lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)
g = sns.clustermap(iris, row_colors=row_colors)
# {row,col}_colors : list-like or pandas DataFrame/Series. # List of colors to label for either the rows or columns. Useful to evaluate whether samples within a group are clustered together.
# Can use nested lists or DataFrame for multiple color levels of labeling. ies. DataFrame/Series colors are also matched to the data by their index, ensuring colors are drawn in the correct order.

# -  Use a different colormap and adjust the limits of the color range:
g = sns.clustermap(iris, cmap="mako", vmin=0, vmax=10) # use this for example to diffentiate practical and theoritical outputs and compare easily

# - Use a different similarity metric:
g = sns.clustermap(iris, metric="correlation")  # method is used to choose the clustering linkage method, and default is average as our choice in the work

# - Standardize the data within the columns (properly a normalization):
g = sns.clustermap(iris, standard_scale=1) # # Either 0 (rows) or 1 (columns) # meaning for each row or column, subtract the minimum and divide each by its maximum # (not want we want)

# - Normalize the data within the rows (properly a standardization) :
g = sns.clustermap(iris, z_score=0, cmap="vlag") # Either 0 (rows) or 1 (columns). # Z scores are: z = (x - mean)/std, . This ensures that each row (column) has mean of 0 and variance of 1.

