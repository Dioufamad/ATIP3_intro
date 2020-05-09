# # imports for plots
#
#
# #!/usr/bin/env python3
# #Louison Fresnais, CRCM, Ballester Team
# #Call ./plot_EF.py -d smina_results_0.01.tsv -c 1
# import sys
# import time
# import os
# from pandas import DataFrame
# import numpy as np
# from pandas import *
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pylab
# import math
#
# # ----plot boxplot 1 5sloocv to 1s10xcv
#
# sns.set(style="ticks", palette="pastel")
#
# # Load the example tips dataset
# tips = sns.load_dataset("tips")
#
# # Draw a nested boxplot to show bills by day and time
# sns.boxplot(x="day", y="total_bill",
#             hue="smoker", palette=["m", "g"],
#             data=tips)
# sns.despine(offset=10, trim=True)
# print(tips)
#
# # reproduction bfor boxplot runtime reduc
# # Load the example tips dataset
# df_runtime_reduc = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/runtime_reduc2.tsv", sep="\t")
# # print(df_runtime_reduc)
# ax = sns.boxplot(x='Test_RF', y='MCC', hue='Model', data=df_runtime_reduc, linewidth=1, fliersize=0)
# ax = sns.swarmplot(x='Test_RF', y='MCC', hue='Model', data=df_runtime_reduc, linewidth=1, dodge=True, color=".9", size=3)
# ax.grid(True)
# ax.set_ylabel('MCC')
# plt.show()
# ###########
# df8 = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/tab8.tsv", sep="\t")
# dfObj = df8
# df_sort_byMCC_only = dfObj.sort_values(by =['MCC'], ascending=False )
# basedir = os.getcwd() #for setting the working directory
# output_filename_for_FS_omc_mdl = basedir + "/" + "df_sort_byMCC_only.csv"
# df_sort_byMCC_only.to_csv(output_filename_for_FS_omc_mdl, index=None, header=True, sep='\t')
#
# df_sortMCC_all = dfObj.sort_values(by =["Algorithms",'MCC'], ascending=False )
# df_sortMCC1 = dfObj.sort_values(by ='MCC', ascending=False )
# df_sortMCC2 = df_sortMCC1.sort_values(by ='Algorithms', ascending=False )
# df_sortMCC3 = df_sortMCC2.sort_values(by ='Profile', ascending=False )
# ####rank on MCC top 10
# df2 = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/Rank_Perf_red3.tsv", sep="\t")
# dfObj2 = df2
# df_sort_byMCC_only2 = dfObj2.sort_values(by =['MCC'], ascending=False )
# last_index_rank = len(df_sort_byMCC_only2)+1
# myranks = list(range(1,last_index_rank))
# ranks_data = np.array(myranks)
# df_sort_byMCC_only2.insert(0, 'Rank', ranks_data)
# basedir = os.getcwd() #for setting the working directory
# output_filename_for_FS_omc_mdl = basedir + "/" + "df_sort_byMCC_only3.csv"
# df_sort_byMCC_only2.to_csv(output_filename_for_FS_omc_mdl, index=None, header=True, sep='\t')
#
#
#
# #### plot ranking of profiles by methods using best oMC value
#
# df_curves_perf_by_prof = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/df_sort_byMCC_only3_fig_perfs_1.csv", sep=",")
# # df_curves_perf_by_prof = read_csv(data, sep=",")
# # df_sort_byMCC_only3_fig_perfs
# #df_clean = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1)
# # plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_100_DUDE","HR1_100_DEKOIS2.0"], marker = "o")
# plot = df_curves_perf_by_prof.plot(kind='line', x="Profile", y=["RF","XGBoost","DNN","SVM-lk","SVM-rbf"], marker = "o") # y=["EF1_Stratified25_DEKOIS2.0","EF1_100_DEKOIS2.0"]
# #plot = df.plot(kind='line', x="Target", y=["HR1_100_SMINA","HR1_100_RF-Score-VS_V2"], marker = "o")
# #plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_Stratified25_DUDE","HR1_100_DUDE", "HR1_100_DUDE_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["EF1_100_SMINA","EF1_100_RF-Score-VS_V2", "EF1_100_RF-Score-VS_V2_TARGET_SPECIFC"], marker = "o")
#
# plot.set_ylabel('MCC')
# xticks = []
# plot.set_xticks(range(len(df_curves_perf_by_prof["Profile"])))
# plot.set_xticklabels(df_curves_perf_by_prof["Profile"])
#
# # Best models distribution across profiles
# # Profile(s) used by models
# x
# -0.2
# 5.2
# y - 1 1
# margin left plot 0.145
# #
# print(plot.get_xlim())
#
# plot.xlim(-0.5,7.5) #
# plot.margins(0.2) #
# plt.show() #
#
#
# #################rank algs following fs by profile
#
# df_curves_fs_by_prof = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/Rank_Perf_fs2-wfs_only.tsv", sep="\t")
# # sort done
# # # df_cn
# df_cn = df_curves_fs_by_prof.sort_values(by =['CN'], ascending=True )
# plot = df_cn.plot(kind='line', x="Algorithm", y="CN", marker = "o")
# # # df_cna
# # df_cna = df_curves_fs_by_prof.sort_values(by =['CNA'], ascending=True )
# # plot = df_cna.plot(kind='line', x="Algorithm", y="CNA", marker = "o")
# # # df_gex
# # df_gex = df_curves_fs_by_prof.sort_values(by =['GEX'], ascending=True )
# # plot = df_gex.plot(kind='line', x="Algorithm", y="GEX", marker = "o")
# # # # df_snv
# # df_snv = df_curves_fs_by_prof.sort_values(by =['SNV'], ascending=True )
# # plot = df_snv.plot(kind='line', x="Algorithm", y="SNV", marker = "o")
# # # df_reals
# # df_reals = df_curves_fs_by_prof.sort_values(by =['Reals'], ascending=True )
# # plot = df_reals.plot(kind='line', x="Algorithm", y="Reals", marker = "o")
# # # df_categ
# # df_categ = df_curves_fs_by_prof.sort_values(by =['Categ.'], ascending=True )
# # plot = df_categ.plot(kind='line', x="Algorithm", y="Categ.", marker = "o")
# ##replace each df in plot done
# plot.set_ylabel('FS size')
# plot.set_xlabel("Algorithms compared")
#
# # xticks = []
# # plot.set_xticks(range(len(df_curves_fs_by_prof["Profile"])))
# # plot.set_xticklabels(df_curves_fs_by_prof["Profile"])
#
# # Variation of feature selection size in CN profile following algorithms
# # Variation of feature selection size in CNA profile following algorithms
# # Variation of feature selection size in GEX profile following algorithms
# # Variation of feature selection size in SNV profile following algorithms
# # Variation of feature selection size in Reals profile following algorithms
# # Variation of feature selection size in Categ. profile following algorithms
# # Profile(s) used by models
# # x = Algorithms compared
# -0.2
# 5.2
# # y = FS size
# y - 1 1
# margin left plot 0.145
#
# ##############
# ####################### plot ranking of algorithms by runtimes following profiles
#
# df_curves_runtime_by_prof = read_csv("/home/diouf/ClassHD_work/actual_repo/xgb_basic27_2/support/plots/Rank_Perf_dur4.tsv", sep="\t")
# # df_curves_perf_by_prof = read_csv(data, sep=",")
# # df_sort_byMCC_only3_fig_perfs
# #df_clean = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1)
# # plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_100_DUDE","HR1_100_DEKOIS2.0"], marker = "o")
# plot = df_curves_runtime_by_prof.plot(kind='line', x="Profile", y=["RF","XGBoost","DNN","SVM-lk","SVM-rbf"], marker = "o") # y=["EF1_Stratified25_DEKOIS2.0","EF1_100_DEKOIS2.0"]
# #plot = df.plot(kind='line', x="Target", y=["HR1_100_SMINA","HR1_100_RF-Score-VS_V2"], marker = "o")
# #plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_Stratified25_DUDE","HR1_100_DUDE", "HR1_100_DUDE_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["EF1_100_SMINA","EF1_100_RF-Score-VS_V2", "EF1_100_RF-Score-VS_V2_TARGET_SPECIFC"], marker = "o")
#
# plot.set_ylabel('Duration (seconds)')
# xticks = []
# plot.set_xticks(range(len(df_curves_perf_by_prof["Profile"])))
# plot.set_xticklabels(df_curves_perf_by_prof["Profile"])
#
# # Best models distribution across profiles
# # Profile(s) used by models
# x
# -0.2
# 5.2
# y - 1 1
# margin left plot 0.145
#
#
#
# ##################"
#
#
# #----------plot1
#
# df = read_csv(data, sep="\t")
# #plot = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1).plot(marker='o')
# #plot.set_ylabel('EF1%')
# #To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
# #with a random first element(not displayed on the plot)
# # xlabels = []
# # xlabels.append('def')
# # for elem in df["Target"]:
# #     xlabels.append(elem)
# # plot.set_xticklabels(xlabels)
# # plt.suptitle('Docking Enrichment for the top 1% for 100% of the dataset \
# #              \n Docking poses rescored with RF-Score-VS V2')
# print(df)
# #Target(unique docked DUDE+DEKOIS2.0 actives)
# ax = sns.boxplot(x='Test', y='MCC', hue='Model', data=df, linewidth=1, fliersize = 0)
# ax = sns.swarmplot(x='Test', y='MCC', hue='Model', data=df, linewidth=1, dodge=True, color=".9", size=3)
# ax.grid(True)
# #ax.set_title('Boxplots of top 1% affinity for DEKOIS2.0 targets \nhaving EF1% greater than 0 rescored with RF-Score-VS V2')
# #ax.set_title('Boxplots of top 100 molecules for DEKOIS2.0 data-sets \nhaving EF1% greater than 0 scored RF-Score-VS V2 vs\nscored with target-specific RF-Score-VS V2 vs scored with SMINA')
# #ax.set_title('Boxplots of top 100 molecules for DUD-E data-sets \nhaving EF1% greater than 0 docked with SMINA')
# ax.set_ylabel('MCC')
# plt.show()
# #swarmplot allow to plot data points over boxes
# # ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
# #the following allow nested boxplot according to a variable
# #ax = sns.boxplot(x="day", y="total_bill", hue="time",data=tips, linewidth=2.5)
#
#
# # ------plot 2
# df = read_csv(data, sep=",")
# #df_clean = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1)
# # plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_100_DUDE","HR1_100_DEKOIS2.0"], marker = "o")
# plot = df.plot(kind='line', x="Target", y=["EF1_Stratified25_DEKOIS2.0","EF1_100_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["HR1_100_SMINA","HR1_100_RF-Score-VS_V2"], marker = "o")
# #plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_Stratified25_DUDE","HR1_100_DUDE", "HR1_100_DUDE_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["EF1_100_SMINA","EF1_100_RF-Score-VS_V2", "EF1_100_RF-Score-VS_V2_TARGET_SPECIFC"], marker = "o")
#
# plot.set_ylabel('EF1%')
# xticks = []
# plot.set_xticks(range(len(df["Target"])))
# plot.set_xticklabels(df["Target"])
# #
# # plot.set_ylabel('HR1%')
# # xticks = []
# # plot.set_xticks(range(len(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])))
# # plot.set_xticklabels(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])
#
# #Attempting to use two y axis but not a good idea for readibility
# # fig, ax1 = plt.subplots()
# # x = df["Target"]
# # y1 = df[["EF1_25_DEKOIS2.0","EF1_100_DEKOIS2.0"]]
# # y2 = df[["HR1_25_DEKOIS2.0","HR1_100_DEKOIS2.0"]]
#
# # ax2 = ax1.twinx()
# # ax1.plot(x, y1, marker = "o")
# # ax2.plot(x, y2, marker = "*")
# #
# # ax1.set_xlabel('X data')
#
# # ax2.set_ylabel('HR1%', color='green')
# #
# # plt.show
# #ax.plot(kind='line', x="Target", y=["EF1_25_DEKOIS2.0", "EF1_100_DEKOIS2.0"], data = df, marker = "o")
# #plot.set_xlim(-0.5,8)
# #To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
# #with a random first element(not displayed on the plot)
#
#
# #xlabels.append('def')
# # for elem in df["Target"]:
# #     xlabels.append(elem)
#
# #plt.suptitle('Enrichment Factor at top 100 for DEKOIS2.0 data-sets. \n SMINA poses scored by RF-Score-VS V2 scoring function \nvs scored by target-specific RF-Score-VS V2' )
# #plt.suptitle('Enrichment Factor at top 100 for DUD-E data-sets vs DEKOIS2.0 data-sets')
# print(plot.get_xlim())
# plt.xlim(-0.5,7.5)
# plot.margins(0.2)
# plt.show()
#
# ######model plot1 : box plots
# #####################################################################
# #   						INPUT	 								#
# #####################################################################
#
#
# df = read_csv(data, sep="\t")
# #plot = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1).plot(marker='o')
# #plot.set_ylabel('EF1%')
# #To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
# #with a random first element(not displayed on the plot)
# # xlabels = []
# # xlabels.append('def')
# # for elem in df["Target"]:
# #     xlabels.append(elem)
# # plot.set_xticklabels(xlabels)
# # plt.suptitle('Docking Enrichment for the top 1% for 100% of the dataset \
# #              \n Docking poses rescored with RF-Score-VS V2')
# print(df)
# #Target(unique docked DUDE+DEKOIS2.0 actives)
# ax = sns.boxplot(x='Target', y='-LOG10(VALUE)', hue='chunk', data=df, linewidth=1, fliersize = 0)
# ax = sns.swarmplot(x='Target', y='-LOG10(VALUE)', hue='chunk', data=df, linewidth=1, dodge=True, color=".9", size=3)
# ax.grid(True)
# #ax.set_title('Boxplots of top 1% affinity for DEKOIS2.0 targets \nhaving EF1% greater than 0 rescored with RF-Score-VS V2')
# #ax.set_title('Boxplots of top 100 molecules for DEKOIS2.0 data-sets \nhaving EF1% greater than 0 scored RF-Score-VS V2 vs\nscored with target-specific RF-Score-VS V2 vs scored with SMINA')
# #ax.set_title('Boxplots of top 100 molecules for DUD-E data-sets \nhaving EF1% greater than 0 docked with SMINA')
# ax.set_ylabel('-LOG10(measured affinity)')
# plt.show()
# #swarmplot allow to plot data points over boxes
# # ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
# #the following allow nested boxplot according to a variable
# #ax = sns.boxplot(x="day", y="total_bill", hue="time",data=tips, linewidth=2.5)
#
# ######model plot1 : sticks as curves
# df = read_csv(data, sep=",")
# #df_clean = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1)
# # plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_100_DUDE","HR1_100_DEKOIS2.0"], marker = "o")
# plot = df.plot(kind='line', x="Target", y=["EF1_Stratified25_DEKOIS2.0","EF1_100_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["HR1_100_SMINA","HR1_100_RF-Score-VS_V2"], marker = "o")
# #plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_Stratified25_DUDE","HR1_100_DUDE", "HR1_100_DUDE_DEKOIS2.0"], marker = "o")
# #plot = df.plot(kind='line', x="Target", y=["EF1_100_SMINA","EF1_100_RF-Score-VS_V2", "EF1_100_RF-Score-VS_V2_TARGET_SPECIFC"], marker = "o")
#
# plot.set_ylabel('EF1%')
# xticks = []
# plot.set_xticks(range(len(df["Target"])))
# plot.set_xticklabels(df["Target"])
# #
# # plot.set_ylabel('HR1%')
# # xticks = []
# # plot.set_xticks(range(len(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])))
# # plot.set_xticklabels(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])
#
# #Attempting to use two y axis but not a good idea for readibility
# # fig, ax1 = plt.subplots()
# # x = df["Target"]
# # y1 = df[["EF1_25_DEKOIS2.0","EF1_100_DEKOIS2.0"]]
# # y2 = df[["HR1_25_DEKOIS2.0","HR1_100_DEKOIS2.0"]]
#
# # ax2 = ax1.twinx()
# # ax1.plot(x, y1, marker = "o")
# # ax2.plot(x, y2, marker = "*")
# #
# # ax1.set_xlabel('X data')
#
# # ax2.set_ylabel('HR1%', color='green')
# #
# # plt.show
# #ax.plot(kind='line', x="Target", y=["EF1_25_DEKOIS2.0", "EF1_100_DEKOIS2.0"], data = df, marker = "o")
# #plot.set_xlim(-0.5,8)
# #To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
# #with a random first element(not displayed on the plot)
#
#
# #xlabels.append('def')
# # for elem in df["Target"]:
# #     xlabels.append(elem)
#
# #plt.suptitle('Enrichment Factor at top 100 for DEKOIS2.0 data-sets. \n SMINA poses scored by RF-Score-VS V2 scoring function \nvs scored by target-specific RF-Score-VS V2' )
# #plt.suptitle('Enrichment Factor at top 100 for DUD-E data-sets vs DEKOIS2.0 data-sets')
# print(plot.get_xlim())
# plt.xlim(-0.5,7.5)
# plot.margins(0.2)
# plt.show()