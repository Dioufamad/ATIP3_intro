#!/usr/bin/env python3
#Louison Fresnais, CRCM, Ballester Team
#Call ./plot_EF.py -d smina_results_0.01.tsv -c 1
import sys
import time
import os
from pandas import DataFrame
from pandas import *
import seaborn as sns
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

df = read_csv(data, sep="\t")
#plot = df.drop(["EF1_25_RESCORE", "EF1_25_DUDE", "EF1_25_DEKOIS2.0", "EF1_100_DUDE"], axis=1).plot(marker='o')
#plot.set_ylabel('EF1%')
#To avoid xlabels being shifted due to first label not recgonized, we create a xtick_list
#with a random first element(not displayed on the plot)
# xlabels = []
# xlabels.append('def')
# for elem in df["Target"]:
#     xlabels.append(elem)
# plot.set_xticklabels(xlabels)
# plt.suptitle('Docking Enrichment for the top 1% for 100% of the dataset \
#              \n Docking poses rescored with RF-Score-VS V2')
print(df)
#Target(unique docked DUDE+DEKOIS2.0 actives)
ax = sns.boxplot(x='Target', y='-LOG10(VALUE)', hue='chunk', data=df, linewidth=1, fliersize = 0)
ax = sns.swarmplot(x='Target', y='-LOG10(VALUE)', hue='chunk', data=df, linewidth=1, dodge=True, color=".9", size=3)
ax.grid(True)
#ax.set_title('Boxplots of top 1% affinity for DEKOIS2.0 targets \nhaving EF1% greater than 0 rescored with RF-Score-VS V2')
#ax.set_title('Boxplots of top 100 molecules for DEKOIS2.0 data-sets \nhaving EF1% greater than 0 scored RF-Score-VS V2 vs\nscored with target-specific RF-Score-VS V2 vs scored with SMINA')
#ax.set_title('Boxplots of top 100 molecules for DUD-E data-sets \nhaving EF1% greater than 0 docked with SMINA')
ax.set_ylabel('-LOG10(measured affinity)')
plt.show()
#swarmplot allow to plot data points over boxes
# ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
#the following allow nested boxplot according to a variable
#ax = sns.boxplot(x="day", y="total_bill", hue="time",data=tips, linewidth=2.5)
