#>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap # to wrap plot titles

#>>>>>>>>>>>>>>>>>>>Choosing the environnement values
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
# cohort_used = "REMAGUS02"
# cohort_used = "REMAGUS04"
# cohort_used = "MDAnderson"
cohort_used = "Allthree"
#>>>>>>>>Setting up the environnement values
# ----for the location of the datasets
# command_center = "Gustave_Roussy"
command_center = "Home"
if command_center == "Gustave_Roussy":
	rest_of_abs_path_b4_content_root = "/home/amad/PycharmProjects/ATIP3_in_GR/"
else : # command_center = "Home"
	rest_of_abs_path_b4_content_root = "/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/"
print("command center used recognized...")
# ----for the columns to keep in all 3 cohorts df
if resp_used == "RCH3HSall":
    list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2","GSasMTUS1wGBANasAL096842wPSIas212096_s_at","Model", "Cohort"]
elif resp_used == "RCH3HSdefined":
    list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_RO","BestResCat_as_RP", "BestResCat_as_HER2","GSasMTUS1wGBANasAL096842wPSIas212096_s_at","Model", "Cohort"]
elif resp_used == "RCHdefined":
    list_of_cols_to_keep = ["BestResCat_as_RCH", "GSasMTUS1wGBANasAL096842wPSIas212096_s_at","Model", "Cohort"]
elif resp_used == "TNBCdefined":
    list_of_cols_to_keep = ["BestResCat_as_TNBC", "GSasMTUS1wGBANasAL096842wPSIas212096_s_at","Model", "Cohort"]
else : # resp_used == "RCHandTNBCdefined":
    list_of_cols_to_keep = ["BestResCat_as_RCH", "BestResCat_as_TNBC", "GSasMTUS1wGBANasAL096842wPSIas212096_s_at","Model", "Cohort"]
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
#---making the dataset to use
sep_in_file = ","
if cohort_used == "REMAGUS02":
    df_file_R02 = pd.read_csv(file_path_R02, sep_in_file)
    df_file_R02.insert(len(list(df_file_R02.columns)), 'Cohort', 'Remagus02')
    df_file_R02 = df_file_R02[list_of_cols_to_keep]
    df_file = df_file_R02
elif cohort_used == "REMAGUS04":
    df_file_R04 = pd.read_csv(file_path_R04, sep_in_file)
    df_file_R04.insert(len(list(df_file_R04.columns)), 'Cohort', 'Remagus04')
    df_file_R04 = df_file_R04[list_of_cols_to_keep]
    df_file = df_file_R04
elif cohort_used == "MDAnderson":
    df_file_MDA = pd.read_csv(file_path_MDA, sep_in_file)
    df_file_MDA.insert(len(list(df_file_MDA.columns)), 'Cohort', 'MDAnderson')
    df_file_MDA = df_file_MDA[list_of_cols_to_keep]
    df_file = df_file_MDA
else: # cohort_used == "Allthree":
    #R02_part
    df_file_R02 = pd.read_csv(file_path_R02, sep_in_file)
    df_file_R02.insert(len(list(df_file_R02.columns)), 'Cohort', 'Remagus02')
    df_file_R02 = df_file_R02[list_of_cols_to_keep]
    #R04_part
    df_file_R04 = pd.read_csv(file_path_R04, sep_in_file)
    df_file_R04.insert(len(list(df_file_R04.columns)), 'Cohort', 'Remagus04')
    df_file_R04 = df_file_R04[list_of_cols_to_keep]
    #MDA_part
    df_file_MDA = pd.read_csv(file_path_MDA, sep_in_file)
    df_file_MDA.insert(len(list(df_file_MDA.columns)), 'Cohort', 'MDAnderson')
    df_file_MDA = df_file_MDA[list_of_cols_to_keep]
    # joining the dataframes by cohort into one df for all cohorts
    list_of_datasets_to_concat = [df_file_R02, df_file_R04, df_file_MDA]
    df_file_final = pd.concat(list_of_datasets_to_concat)
    df_file = df_file_final

#>>>>>>>>>>>>>>>>>>>>>Finding the colname of ATIP3 (done once to get the result below, no need to rerun)
# # df_file is obtained already so we use it (preferably when created from R02 with 2RasRCHandTNBCdefined
# # find the name of atip3 col
# list_colnames = list(df_file.columns)
# atip3_col_suspects = []
# for str_colname in list_colnames:
#     if "MTUS1" in str_colname:
#         atip3_col_suspects.append(str_colname)
# print("the found suspects gene symbols are:")
# for suspected_gs in atip3_col_suspects:
#     print("-",suspected_gs)
# print("We use this portal to find the most suitable GBAN for MTUS1. Portal is HUGO Gene Nomenclature Commitee")
# # source portal link is : https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:29789
# print("Among the ones in our data, the only GBAN to correspond to ATIP3 is AL096842.")
# print("(in portal go to Nucleotide resources and then to INSDC and the corresponding GBAN is the only one found with link to accession page as proof.")
# # link to accession page as of April 10th 2020 is : https://www.ncbi.nlm.nih.gov/nuccore/AL096842
name_of_atip3_col_to_use = "GSasMTUS1wGBANasAL096842wPSIas212096_s_at"

#>>>>>>>>>>>>>>>>>>>Finding the number of commons features across the 3 cohorts
if cohort_used == "Allthree":
    sep_in_file = ","
    # R02_part
    df_file_R02 = pd.read_csv(file_path_R02, sep_in_file)
    # R04_part
    df_file_R04 = pd.read_csv(file_path_R04, sep_in_file)
    # MDA_part
    df_file_MDA = pd.read_csv(file_path_MDA, sep_in_file)
    if resp_used == "RCH3HSall":
        num_resp_cols_from_sup = 4
        # the group of resp columns  for each cohort
        feat_cols_R02 = df_file_R02.columns[num_resp_cols_from_sup:-1]
        feat_cols_R04 = df_file_R04.columns[num_resp_cols_from_sup:-1]
        feat_cols_MDA = df_file_MDA.columns[num_resp_cols_from_sup:-1]
    elif resp_used == "RCH3HSdefined":
        num_resp_cols_from_sup = 4
        # the group of resp columns  for each cohort
        feat_cols_R02 = df_file_R02.columns[num_resp_cols_from_sup:-1]
        feat_cols_R04 = df_file_R04.columns[num_resp_cols_from_sup:-1]
        feat_cols_MDA = df_file_MDA.columns[num_resp_cols_from_sup:-1]
    elif resp_used == "RCHdefined":
        num_resp_cols_from_sup = 1
        # the group of resp columns  for each cohort
        feat_cols_R02 = df_file_R02.columns[num_resp_cols_from_sup:-1]
        feat_cols_R04 = df_file_R04.columns[num_resp_cols_from_sup:-1]
        feat_cols_MDA = df_file_MDA.columns[num_resp_cols_from_sup:-1]
    elif resp_used == "TNBCdefined":
        num_resp_cols_from_sup = 1
        # the group of resp columns  for each cohort
        feat_cols_R02 = df_file_R02.columns[num_resp_cols_from_sup:-1]
        feat_cols_R04 = df_file_R04.columns[num_resp_cols_from_sup:-1]
        feat_cols_MDA = df_file_MDA.columns[num_resp_cols_from_sup:-1]
    else:  # resp_used == "RCHandTNBCdefined":
        num_resp_cols_from_sup = 2
        # the group of resp columns  for each cohort
        feat_cols_R02 = df_file_R02.columns[num_resp_cols_from_sup:-1]
        feat_cols_R04 = df_file_R04.columns[num_resp_cols_from_sup:-1]
        feat_cols_MDA = df_file_MDA.columns[num_resp_cols_from_sup:-1]
    # get a list that is the intersection
    sorted_list_of_common_fts = sorted(list(set(list(feat_cols_R02)) & set(list(feat_cols_R04)) & set(list(feat_cols_MDA))))
    sorted_list_of_non_common_fts_in_R02 = sorted(list(set(list(feat_cols_R02)) - set(sorted_list_of_common_fts)))      # use this for the specific to 2 lists set([1, 2]).symmetric_difference(set([2, 3]))
    sorted_list_of_non_common_fts_in_R04 = sorted(list(set(list(feat_cols_R04)) - set(sorted_list_of_common_fts)))
    sorted_list_of_non_common_fts_in_MDA = sorted(list(set(list(feat_cols_MDA)) - set(sorted_list_of_common_fts)))
    sorted_list_of_fts_only_in_R02 = sorted(list(set(list(feat_cols_R02)) - set(list(feat_cols_R04)) - set(list(feat_cols_MDA))))
    sorted_list_of_fts_only_in_R04 = sorted(list(set(list(feat_cols_R04)) - set(list(feat_cols_R02)) - set(list(feat_cols_MDA))))
    sorted_list_of_fts_only_in_MDA = sorted(list(set(list(feat_cols_MDA)) - set(list(feat_cols_R02)) - set(list(feat_cols_R04))))
    print("Descrition of features representation across cohorts : ")
    print(" - Common features :",len(sorted_list_of_common_fts))
    print(" - Non common features in Remagus02 :", len(sorted_list_of_non_common_fts_in_R02))
    print(" - Non common features in Remagus04 :", len(sorted_list_of_non_common_fts_in_R04))
    print(" - Non common features in MDAnderson :", len(sorted_list_of_non_common_fts_in_MDA))
    print(" - Features only in Remagus02 :", len(sorted_list_of_fts_only_in_R02))
    print(" - Features only in Remagus04 :", len(sorted_list_of_fts_only_in_R04))
    print(" - Features only in MDAnderson :", len(sorted_list_of_fts_only_in_MDA))
    ##! make a table reporting present or not with 4cols (all features list, xist in R02, xist in R04, xist in MDA) with ft or Yes or No

#>>>>>>>>>>>>>>>>>>> An Histogramm representiing ATIP3 values distribution across the 3 cohorts
# step 1 one unique histrogram
fig, ax = plt.subplots()
sns.distplot(df_file[df_file['Cohort']=="Remagus02"][name_of_atip3_col_to_use], color="red", label="Remagus02") # df_file[df_file['cohort']==Remagus02] is the dataset resctricted to only the rows from R02
sns.distplot(df_file[df_file['Cohort']=="Remagus04"][name_of_atip3_col_to_use], color="skyblue", label="Remagus04")
sns.distplot(df_file[df_file['Cohort']=="MDAnderson"][name_of_atip3_col_to_use], color="olive", label="MDAnderson")
# ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Numbers in bins", fontsize=18)
plt.title("\n".join(wrap("Distribution of ATIP3 values in each cohort.")), fontsize=18)
plt.legend(title="Cohorts exlored", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/Hist_atip3values_3cohorts.png')

# step 2 one unique histrogram but only for R02 and R04
fig, ax = plt.subplots()
sns.distplot(df_file[df_file['Cohort']=="Remagus02"][name_of_atip3_col_to_use], color="red", label="Remagus02") # df_file[df_file['cohort']==Remagus02] is the dataset resctricted to only the rows from R02
sns.distplot(df_file[df_file['Cohort']=="Remagus04"][name_of_atip3_col_to_use], color="skyblue", label="Remagus04")
# sns.distplot(df_file[df_file['Cohort']=="MDAnderson"][name_of_atip3_col_to_use], color="olive", label="MDAnderson")
# ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Numbers in bins", fontsize=18)
plt.title("\n".join(wrap("Distribution of ATIP3 values in each cohort.")), fontsize=18)
plt.legend(title="Cohorts exlored", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/Hist_atip3values_2cohortsR02R04.png')

# step 3 one unique histrogram but only for MDA
fig, ax = plt.subplots()
# sns.distplot(df_file[df_file['Cohort']=="Remagus02"][name_of_atip3_col_to_use], color="red", label="Remagus02") # df_file[df_file['cohort']==Remagus02] is the dataset resctricted to only the rows from R02
# sns.distplot(df_file[df_file['Cohort']=="Remagus04"][name_of_atip3_col_to_use], color="skyblue", label="Remagus04")
sns.distplot(df_file[df_file['Cohort']=="MDAnderson"][name_of_atip3_col_to_use], color="olive", label="MDAnderson")
# ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Numbers in bins", fontsize=18)
plt.title("\n".join(wrap("Distribution of ATIP3 values by cohort.")), fontsize=18)
plt.legend(title="Cohorts exlored", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/Hist_atip3values_1cohortMDA.png')

#>>>>>>>>>>>>>>>>>>> A violin plot representiing ATIP3 values distribution across the 3 cohorts
# step 1-1 : the 3 cohorts in 1 plot, horizontal violin plots, not splited and with points as inner
plot = sns.violinplot(data=df_file, y="Cohort", x=name_of_atip3_col_to_use,
                      palette="muted",order=["Remagus02", "Remagus04","MDAnderson"],
                      split=False,scale_hue=True,
                      scale="count",inner="point",bw=0.2,cut=0) # hue="BestResCat_as_RCH" for split

#- setting up the figure
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Cohorts explored", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values by cohort.")), fontsize=18)
# plt.legend(title="BestResCat_as_RCH", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_3cohorts_hor_ns.png')

# step 1-2 : the 2 cohorts R02 and R04 in 1 plot, horizontal violin plots, not splited and with points as inner
df_data12 = df_file.loc[df_file['Cohort'].isin(["Remagus02", "Remagus04"])]
plot = sns.violinplot(data=df_data12, y="Cohort", x=name_of_atip3_col_to_use,
                      palette="muted",order=["Remagus02", "Remagus04"],
                      split=False,scale_hue=True,
                      scale="count",inner="point",bw=0.2,cut=0) # hue="BestResCat_as_RCH" for split

#- setting up the figure
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Cohorts explored", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values by cohort.")), fontsize=18)
# plt.legend(title="BestResCat_as_RCH", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_2cohortsR02R04_hor_ns.png')

# step 1-3 : 1 cohort (MDA) in 1 plot, horizontal violin plots, not splited and with points as inner
df_data13 = df_file.loc[df_file['Cohort'].isin(["MDAnderson"])]
plot = sns.violinplot(data=df_data13, y="Cohort", x=name_of_atip3_col_to_use,
                      palette="muted",order=["MDAnderson"],
                      split=False,scale_hue=True,
                      scale="count",inner="point",bw=0.2,cut=0) # hue="BestResCat_as_RCH" for split

#- setting up the figure
plt.xlabel("ATIP3 expression values", fontsize=18)
plt.ylabel("Cohorts explored", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values by cohort.")), fontsize=18)
# plt.legend(title="BestResCat_as_RCH", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_1cohortMDA_hor_ns.png')

# step 2-1 : all 3 cohorts in 1 plot, vertical violin plots, splited with RCH and with inner as quartile
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_RCH", data=df_file,
                      palette="muted", split=True,order=["Remagus02", "Remagus04","MDAnderson"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values by cohort and in each pCR response group")), fontsize=18)
plt.legend(title="pCR groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_3cohorts_ver_ns.png')

# step 2-2 : 2 cohorts (R02,R04), vertical violin plots, splited with RCH and with inner as quartile
df_data22 = df_file.loc[df_file['Cohort'].isin(["Remagus02", "Remagus04"])]
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_RCH", data=df_data22,
                      palette="muted", split=True,order=["Remagus02", "Remagus04"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values in 2 cohorts and in each pCR response group")), fontsize=18)
plt.legend(title="pCR groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_3cohorts_ver_s.png')

# step 2-3 : 1 cohorts MDA, vertical violin plots, splited with RCH and with inner as quartile
df_data23 = df_file.loc[df_file['Cohort'].isin(["MDAnderson"])]
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_RCH", data=df_data23,
                      palette="muted", split=True,order=["MDAnderson"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values in 2 cohorts and in each pCR response group")), fontsize=18)
plt.legend(title="pCR groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_1cohortsMDA_ver_s.png')






# step 3-1 : all 3 cohorts in 1 plot, vertical violin plots, splited with TNBC and with inner as quartile
df_file["BestResCat_as_TNBC"].replace(["Res", "Sen"], ["No", "Yes"], inplace=True)
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_TNBC", data=df_file,
                      palette="muted", split=True,order=["Remagus02", "Remagus04","MDAnderson"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values by cohort and in each pCR response group")), fontsize=18)
plt.legend(title="TNBC status groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_TNBC_3cohorts_ver_ns.png')

# step 3-2 : 2 cohorts (R02,R04), vertical violin plots, splited with TNBC and with inner as quartile
df_file["BestResCat_as_TNBC"].replace(["Res", "Sen"], ["No", "Yes"], inplace=True)
df_data32 = df_file.loc[df_file['Cohort'].isin(["Remagus02", "Remagus04"])]
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_TNBC", data=df_data32,
                      palette="muted", split=True,order=["Remagus02", "Remagus04"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values in 2 cohorts and in each pCR response group")), fontsize=18)
plt.legend(title="TNBC status groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_TNBC_2cohortsR02R04_ver_s.png')

# step 3-3 : 1 cohorts MDA, vertical violin plots, splited with TNBC and with inner as quartile
df_file["BestResCat_as_TNBC"].replace(["Res", "Sen"], ["No", "Yes"], inplace=True)
df_data33 = df_file.loc[df_file['Cohort'].isin(["MDAnderson"])]
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_TNBC", data=df_data33,
                      palette="muted", split=True,order=["MDAnderson"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("ATIP3 expression values in 2 cohorts and in each pCR response group")), fontsize=18)
plt.legend(title="TNBC status groups : ", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/1vp_atip3values_TNBC_1cohortsMDA_ver_s.png')


#========================================= END








#>>>>>>>>>>>>>>>>>>>>>>>>>>> PLOTS MOCKUPS
# >>> distribution of ATIP3 values over the 3 cohorts : plot for the splited vertical violin plots and the split is over RCH
plot = sns.violinplot(x="Cohort", y=name_of_atip3_col_to_use,
                      hue="BestResCat_as_RCH", data=df_file_final,
                      palette="muted", split=True,order=["Remagus02", "Remagus04","MDAnderson"],
                      scale="count",scale_hue=False,inner="quartile",bw=1,cut=0)
#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("Distribution of ATIP3 values over the cohorts and in each pCR response group")), fontsize=12)
plt.legend(title="BestResCat_as_RCH", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/Fig1.png')


# >>> distribution of ATIP3 values over the 3 cohorts : plot for the splited horizontal violin plots and the split is over RCH and with  points as inner
plot = sns.violinplot(data=df_file_final, y="Cohort", x=name_of_atip3_col_to_use,
                      palette="muted",order=["Remagus02", "Remagus04","MDAnderson"],
                      split=False,scale_hue=True,
                      scale="count",inner="point",bw=0.2,cut=0) # hue="BestResCat_as_RCH"

#- setting up the figure
plt.xlabel("Cohorts", fontsize=18)
plt.ylabel("ATIP3 expression values", fontsize=18)
plt.title("\n".join(wrap("Distribution of ATIP3 values over the cohorts and in each pCR response group")), fontsize=12)
plt.legend(title="BestResCat_as_RCH", loc=1, fontsize='small')
plt.show()
plt.savefig('/home/khamasiga/PALADIN_1/3CEREBRO/garage/projects/ATIP3/CICS/CICS_dev_version/outputs/Fig2.png')

#=========================training 1 : violin plots===================================================
# >>>>a single horizontal violinplot
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.violinplot(x=tips["total_bill"])
# >>>> Draw a vertical violinplot grouped by a categorical variable:
ax = sns.violinplot(x="day", y="total_bill", data=tips)
# >>>> Draw a violinplot with nested grouping by two categorical variables:
ax = sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")
# >>> Draw split violins to compare the across the hue variable :
ax = sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted", split=True)
# >>> Control violin order by passing an explicit order:
ax = sns.violinplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])
# >>>> Scale the violin width by the number of observations in each bin:
ax = sns.violinplot(x="day", y="total_bill", hue="sex",data=tips, palette="Set2", split=True,scale="count")
# Show each observation with a stick inside the violin:
ax = sns.violinplot(x="day", y="total_bill", hue="sex",data=tips, palette="Set2", split=True,scale="count", inner="quartile")
# Scale the density relative to the counts across all bins:
ax = sns.violinplot(x="day", y="total_bill", hue="sex",data=tips, palette="Set2", split=True, scale="count", inner="stick", scale_hue=False)
#Use a narrow bandwidth to reduce the amount of smoothing:
ax = sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, palette="Set2", split=True,scale="count", inner="stick", scale_hue=False, bw=.2)
# Draw horizontal violins:
planets = sns.load_dataset("planets")
ax = sns.violinplot(x="orbital_period", y="method",data=planets[planets.orbital_period < 1000],scale="width", palette="Set3")
# Don’t let density extend past extreme values in the data:
ax = sns.violinplot(x="orbital_period", y="method", data=planets[planets.orbital_period < 1000], cut=0, scale="width", palette="Set3")

#----training 2 : histograms
# Import library and dataset
# import seaborn as sns
df = sns.load_dataset('iris')

# Method 1: on the same Axis
fig, ax = plt.subplots()
sns.distplot(df["sepal_length"], color="skyblue", label="Sepal Length")
sns.distplot(df["sepal_width"], color="red", label="Sepal Width")
# ax.set(xlabel='common xlabel', ylabel='common ylabel')
plt.xlabel("Colors")
plt.ylabel("Values")
plt.title("Colors vs Values")
plt.legend()
plt.show()

# Method 1: on splited Axis
# plot
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot(df["sepal_length"], color="skyblue", ax=axes[0, 0])
sns.distplot(df["sepal_width"], color="olive", ax=axes[0, 1])
sns.distplot(df["petal_length"], color="gold", ax=axes[1, 0])
sns.distplot(df["petal_width"], color="teal", ax=axes[1, 1])
# no need for legend() and show() commands of plt

# NB : # or start with fig = plt.figure() # if used, set tiltes with fig.suptitle('test title', fontsize=20),
# plt.xlabel('xlabel', fontsize=18) plt.ylabel('ylabel', fontsize=16) and save with fig.savefig('test.jpg')