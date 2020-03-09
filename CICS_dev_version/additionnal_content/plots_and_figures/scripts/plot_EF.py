#!/usr/bin/env python3
#Louison Fresnais, CRCM, Ballester Team
#Call ./plot_EF.py -d smina_results_0.01.tsv -c 1
import sys
import time
import os
from pandas import DataFrame
from pandas import *
import matplotlib.pyplot as plt
import pylab
import math

#####################################################################
#   						INPUT	 								#
#####################################################################
def argument_parsing():

	import argparse #https://docs.python.org/3/library/argparse.html

	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description="SMINA launcher for a directory",
		epilog="""This script will launch a SMINA on every targets provided in a directory.

			Contact: fresnaislouison@gmail.com""")

	parser.add_argument('-d','--data', type=str, help='Enter a tsv file with your docking enrichment data"')
	args = parser.parse_args()

	return args
args = vars(argument_parsing())

try:
	data = str(args.get("data"))
	print(data)
except TypeError:
	data = str(input("Enter a tsv file with your docking enrichment data"))

df = read_csv(data, sep=",")
#df_clean = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1)
# plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_100_DUDE","HR1_100_DEKOIS2.0"], marker = "o")
plot = df.plot(kind='line', x="Target", y=["EF1_Stratified25_DEKOIS2.0","EF1_100_DEKOIS2.0"], marker = "o")
#plot = df.plot(kind='line', x="Target", y=["HR1_100_SMINA","HR1_100_RF-Score-VS_V2"], marker = "o")
#plot = df.plot(kind='line', x="Target(unique docked DUDE+DEKOIS2.0 actives)", y=["HR1_Stratified25_DUDE","HR1_100_DUDE", "HR1_100_DUDE_DEKOIS2.0"], marker = "o")
#plot = df.plot(kind='line', x="Target", y=["EF1_100_SMINA","EF1_100_RF-Score-VS_V2", "EF1_100_RF-Score-VS_V2_TARGET_SPECIFC"], marker = "o")

plot.set_ylabel('EF1%')
xticks = []
plot.set_xticks(range(len(df["Target"])))
plot.set_xticklabels(df["Target"])
#
# plot.set_ylabel('HR1%')
# xticks = []
# plot.set_xticks(range(len(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])))
# plot.set_xticklabels(df["Target(unique docked DUDE+DEKOIS2.0 actives)"])

#Attempting to use two y axis but not a good idea for readibility
# fig, ax1 = plt.subplots()
# x = df["Target"]
# y1 = df[["EF1_25_DEKOIS2.0","EF1_100_DEKOIS2.0"]]
# y2 = df[["HR1_25_DEKOIS2.0","HR1_100_DEKOIS2.0"]]

# ax2 = ax1.twinx()
# ax1.plot(x, y1, marker = "o")
# ax2.plot(x, y2, marker = "*")
#
# ax1.set_xlabel('X data')

# ax2.set_ylabel('HR1%', color='green')
#
# plt.show
#ax.plot(kind='line', x="Target", y=["EF1_25_DEKOIS2.0", "EF1_100_DEKOIS2.0"], data = df, marker = "o")
#plot.set_xlim(-0.5,8)
#To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
#with a random first element(not displayed on the plot)


#xlabels.append('def')
# for elem in df["Target"]:
#     xlabels.append(elem)

#plt.suptitle('Enrichment Factor at top 100 for DEKOIS2.0 data-sets. \n SMINA poses scored by RF-Score-VS V2 scoring function \nvs scored by target-specific RF-Score-VS V2' )
#plt.suptitle('Enrichment Factor at top 100 for DUD-E data-sets vs DEKOIS2.0 data-sets')
print(plot.get_xlim())
plt.xlim(-0.5,7.5)
plot.margins(0.2)
plt.show()


