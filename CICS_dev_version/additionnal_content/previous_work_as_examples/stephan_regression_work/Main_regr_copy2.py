import datetime
import locale
import numpy as np
import os
import pandas as pd
import random
import scipy
import warnings
import xgboost as xgb
from collections import Counter
from old_scripts.Featureselection import feat_select
from old_scripts.Stratify import stratify
from random import shuffle

cv_standard = 5  # K-fold cross-validation
num_cores = 38  # Number of cores used

basedir = os.getcwd()

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
warnings.filterwarnings("ignore")
#style shebang
# ------------------THIS PART IS TO GET ALL THE VARIABLES OF THE PROBLEM------------------------------------------------
# + the cell lines (instances)
# + the drug used
# + the cancer type on which it have been used
# + the response of the drug used on that cancer type
# + the features :
# ++ SNV
# ++ CNA
# ++ RNA (GEx)
# ++ Meth
########################################
#         Cell line information        # # lets extract a table of 3 col : sample-cosmic id-cancer type
########################################
cell_lines = pd.read_excel(basedir + "/Data/TableS1E.xlsx", sheetname="TableS1E-CellLines", skiprows={0, 1, 3}, skip_footer=1, header=0, dtype=str)
cell_lines = cell_lines.drop('Unnamed: 0', 1)
cell_lines = cell_lines.drop(cell_lines.index[[-1]])
mapper = cell_lines[["Sample Name", "COSMIC identifier", "Cancer Type\n(matching TCGA label)"]]
mapper = mapper.rename(index=str, columns={"Cancer Type\n(matching TCGA label)": "TCGA"}) # got it

########################################
#         Drug name information        # # lets extract a dict with {identifier as index : [drugname]}
########################################
drugs = pd.read_excel(basedir + '/Data/TableS1F.xlsx', sheetname="TableS1F_ScreenedCompounds", skiprows={0, 1})
drugs = drugs[["Identifier", "Name"]].copy()
drugs["Identifier"] = drugs["Identifier"].apply(str)
mydrugs = drugs.set_index('Identifier').T.to_dict('list') # got it
drugs["Identifier"] = drugs["Identifier"].apply(str) # not necessary; just to give back identifier in drug its type str

########################################
#       Drug response information      # # get a dataframe of cells line ids-drugID-Log_IC50(resp)
########################################
drug_response = pd.read_excel(basedir + '/Data/TableS4A.xlsx', sheetname="TableS4A-IC50s", header=0, skiprows={0, 1, 2, 3, 5})
drug_response["Cell line cosmic identifiers"] = drug_response["Cell line cosmic identifiers"].apply(str)
drug_response = drug_response.drop('Sample Names', 1)
drug_response = pd.melt(drug_response, id_vars=['Cell line cosmic identifiers'], var_name=['DrugID']) # changing the table from wide to long (id = identifiers, col2 = previous 2: columns)
oldlen = len(drug_response) # initial number of cells lines that are studied (may or may not have treatment info)
drug_response["DrugID"] = drug_response["DrugID"].apply(str)
drug_response = drug_response.rename(index=str, columns={"value": "LOG_IC50"}) # rename col value to LOG_IC50
drug_response.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True) # got it # drop any column lacking info (identifiers, drug id,or LOG_IC50)
# only are left the cells lines with a drug tested and with the resp value to that drug known
# lets print the remaining cells line number shall we...

print "Removed", oldlen - len(drug_response), "drug_cell line pairs with NA values.", len(drug_response), "observations in", len(set(drug_response["Cell line cosmic identifiers"])), "cell lines remain."
# THIS IS THE FIRST MERGING OF DATAFRAMES # get a dataframe of cells line ids-drugID-Log_IC50(resp)-sample_name_TCGA
# stick together 2 dataframes and the 2 columns (one from a dataframe) to make their lines correspond i.e. the cosmic identifiers of the cell lines
drug_response = pd.merge(drug_response, mapper, how="inner", right_on="COSMIC identifier", left_on="Cell line cosmic identifiers") # got it
drug_response = drug_response[["Cell line cosmic identifiers", "DrugID", "LOG_IC50", "Sample Name", "TCGA"]] # get ride of the cosmic id from the 2nd dataframe invited in
drug_response["LOG_IC50"] = np.log10(np.exp(drug_response["LOG_IC50"]))  # Convert reported ln to log10
# WARNING : verify if the transformation in previous line is not already done and if it is necessary
drug_response["TCGA"].replace(to_replace="COAD/READ", value="COADREAD", inplace=True)

########################################
#               Add SNV                # # gettting a dataframe of the snv data (instances = cosmic id and features = genes_mut) will available and non available marked true and false
########################################
print "Reading SNV data"

# Odd behaviour happens when lines are not defined explicitly to skip
snvdata = pd.read_excel(basedir + '/Data/TableS2C.xlsx', sheetname="TableS2C-CellLineVariants", skiprows={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, usecols=[1, 2, 3, 4, 8])
snvdata["COSMIC_ID"] = snvdata["COSMIC_ID"].apply(str)
snvdata["Gene"] = snvdata["Gene"] + "_mut"
snvdata["Classification"] = True

snvdata = pd.pivot_table(snvdata, index=["COSMIC_ID", "SAMPLE"], columns='Gene', values='Classification')
snvdata.fillna(value=False, inplace=True)
snvdata = snvdata.reset_index()
del snvdata["SAMPLE"]

# 470 predicted cancer genes from Iorio et al (2016) are the only one kept
filtercolumns = ["COSMIC_ID", "FBXW7_mut", "KRAS_mut", "NF1_mut", "NRAS_mut", "PTEN_mut", "RB1_mut", "RPL5_mut", "TP53_mut", "FLT3_mut", "CNOT1_mut", "CNOT3_mut", "FGFR1_mut", "PTPN11_mut", "ACTG1_mut", "APC_mut", "ARID1A_mut", "BAP1_mut", "BRAF_mut", "CASP8_mut", "CDH1_mut", "CDKN1A_mut", "CDKN1B_mut", "CDKN2A_mut", "CIC_mut", "CTCF_mut", "CTNNB1_mut", "DDX5_mut", "ELF3_mut", "FAT1_mut", "HRAS_mut", "IDH1_mut", "KDM6A_mut", "KEAP1_mut", "MLL2_mut", "MLL3_mut", "NFE2L2_mut", "PHF6_mut", "RASA1_mut", "RHOA_mut", "RPSAP58_mut", "SETD2_mut", "SMAD4_mut", "STAG2_mut", "TBL1XR1_mut", "TXNIP_mut", "ACSL6_mut", "AFF4_mut", "ARHGAP26_mut", "BCOR_mut", "BLM_mut", "BRCA1_mut", "CDK12_mut", "CHEK2_mut", "CLTC_mut", "DICER1_mut", "EIF4A2_mut", "EP300_mut", "ERCC2_mut", "FAM123B_mut", "FGFR2_mut", "FGFR3_mut", "FUS_mut", "GNAS_mut", "GOLGA5_mut", "HSP90AA1_mut", "HSP90AB1_mut", "KLF6_mut", "MECOM_mut", "MED12_mut", "MET_mut", "MLH1_mut", "MYH11_mut", "NDRG1_mut", "NOTCH1_mut", "NUP98_mut", "SF3B1_mut", "SFPQ_mut", "SUZ12_mut", "TBX3_mut", "TGFBR2_mut", "THRAP3_mut", "TSC1_mut", "ZMYM2_mut", "ZNF814_mut", "ACTB_mut", "ADAM10_mut", "AHNAK_mut", "AHR_mut", "ANK3_mut", "AQR_mut", "ARFGAP1_mut", "ARFGEF2_mut", "ARHGAP35_mut", "ARID1B_mut", "ATR_mut", "BCLAF1_mut", "BMPR2_mut", "CAD_mut", "CARM1_mut", "CAST_mut", "CAT_mut", "CCAR1_mut", "CCT5_mut", "CEP290_mut", "CHD3_mut", "CHD9_mut", "CLASP2_mut", "CLSPN_mut", "COPS2_mut", "CSDE1_mut", "CUL2_mut", "DDX3X_mut", "DIS3_mut", "DLG1_mut", "EEF1B2_mut", "EIF2AK3_mut", "EIF4G1_mut", "ELF1_mut", "ERBB2IP_mut", "ERBB3_mut", "FKBP5_mut", "FN1_mut", "G3BP2_mut", "GPS2_mut", "HLA-A_mut", "HNRPDL_mut", "HSPA8_mut", "IREB2_mut", "IRS2_mut", "LIMA1_mut", "MAP3K4_mut", "MAP4K3_mut", "MED24_mut", "MGA_mut", "MTOR_mut", "MYH10_mut", "NAP1L1_mut", "NCF2_mut", "NCOR2_mut", "NUP107_mut", "PCDH18_mut", "PCSK6_mut", "PIK3CB_mut", "PIP5K1A_mut", "PTPRU_mut", "RAD21_mut", "RBM5_mut", "SETDB1_mut", "SF3A3_mut", "SMC1A_mut", "SOS1_mut", "SOS2_mut", "STAG1_mut", "STK4_mut", "TAF1_mut", "TAOK1_mut", "TAOK2_mut", "TNPO1_mut", "TP53BP1_mut", "TRIO_mut", "ZFP36L2_mut", "AKT1_mut", "ARID2_mut", "ATM_mut", "CBFB_mut", "CHD4_mut", "EGFR_mut", "EIF1AX_mut", "FOXA1_mut", "FUBP1_mut", "GATA3_mut", "KDM5C_mut", "LPHN2_mut", "MAP2K4_mut", "MYB_mut", "NCOR1_mut", "PBRM1_mut", "PIK3CA_mut", "PIK3R1_mut", "RUNX1_mut", "STK11_mut", "ZFP36L1_mut", "AKAP9_mut", "ATF1_mut", "ATIC_mut", "BRCA2_mut", "ERBB2_mut", "FOXP1_mut", "HLF_mut", "LCP1_mut", "MKL1_mut", "MLL_mut", "MLLT4_mut", "MYH9_mut", "NF2_mut", "NOTCH2_mut", "NSD1_mut", "PAX5_mut", "PRKAR1A_mut", "SMARCA4_mut", "TCF12_mut", "TCF7L2_mut", "ACO1_mut", "ACVR1B_mut", "ARID4B_mut", "ARNTL_mut", "ASH1L_mut", "ASPM_mut", "BNC2_mut", "BPTF_mut", "CSNK1G3_mut", "CUL1_mut", "DHX15_mut", "EIF2C3_mut", "FMR1_mut", "HCFC1_mut", "ITSN1_mut", "KALRN_mut", "KLF4_mut", "LRP6_mut", "MACF1_mut", "MAX_mut", "MED23_mut", "MSR1_mut", "MUC20_mut", "MYH14_mut", "NR4A2_mut", "PIK3R3_mut", "POLR2B_mut", "PRKCZ_mut", "PTGS1_mut", "RBBP7_mut", "RFC4_mut", "RHEB_mut", "RPGR_mut", "SEC24D_mut", "SPTAN1_mut", "SRGAP1_mut", "STIP1_mut", "SVEP1_mut", "TFDP1_mut", "TOM1_mut", "ATRX_mut", "MYD88_mut", "XPO1_mut", "ASXL1_mut", "CASP1_mut", "GNAI1_mut", "PLCB1_mut", "DNMT3A_mut", "IDH2_mut", "MAP2K1_mut", "SMAD2_mut", "SOX9_mut", "WT1_mut", "CDC73_mut", "CREBBP_mut", "PPP2R1A_mut", "SRGAP3_mut", "AXIN2_mut", "BRWD1_mut", "FXR1_mut", "NR2F2_mut", "NTN4_mut", "PCBP1_mut", "RBM10_mut", "RTN4_mut", "SYNCRIP_mut", "WIPF1_mut", "ZC3H11A_mut", "FBXO11_mut", "MYC_mut", "EEF1A1_mut", "EZH2_mut", "CAPN7_mut", "CUL3_mut", "HGF_mut", "HLA-B_mut", "NCKAP1_mut", "SHMT1_mut", "KDR_mut", "MEN1_mut", "ACAD8_mut", "ARHGEF6_mut", "CHD8_mut", "CLOCK_mut", "HDAC9_mut", "NEDD4L_mut", "NFATC4_mut", "PRPF8_mut", "SIN3A_mut", "TJP1_mut", "ACVR2A_mut", "B2M_mut", "CNOT4_mut", "CSNK2A1_mut", "EPHA2_mut", "RAC1_mut", "RPL22_mut", "SPOP_mut", "U2AF1_mut", "ZNF750_mut", "BCL11A_mut", "CIITA_mut", "COL1A1_mut", "CYLD_mut", "ELF4_mut", "PRRX1_mut", "PSIP1_mut", "WHSC1_mut", "APAF1_mut", "ATP6AP2_mut", "BAZ2B_mut", "FAT2_mut", "GPSM2_mut", "IRF6_mut", "LAMA2_mut", "MED17_mut", "MEF2C_mut", "MGMT_mut", "PABPC3_mut", "PPP2R5C_mut", "RASGRP1_mut", "TCF4_mut", "TFDP2_mut", "TJP2_mut", "TRIP10_mut", "PABPC1_mut", "VHL_mut", "WHSC1L1_mut", "CSDA_mut", "PSMA6_mut", "PSME3_mut", "CEBPA_mut", "KIT_mut", "NPM1_mut", "TET2_mut", "KAT6B_mut", "ZNF292_mut", "ACTG2_mut", "CHD1L_mut", "CRNKL1_mut", "EFTUD2_mut", "EPHA4_mut", "EPHB2_mut", "G3BP1_mut", "GNG2_mut", "LDHA_mut", "MAP4K1_mut", "MMP2_mut", "NCK1_mut", "NTRK2_mut", "PPP2R5A_mut", "PSMD11_mut", "RAD23B_mut", "SPRR3_mut", "UPF3B_mut", "ABL2_mut", "CTTN_mut", "DHX9_mut", "RGS3_mut", "SMO_mut", "PTCH1_mut", "FAM46C_mut", "ALK_mut", "MYCN_mut", "F8_mut", "HDAC3_mut", "YBX1_mut", "EPC1_mut", "FRG1_mut", "SCAI_mut", "FIP1L1_mut", "HNF1A_mut", "SMARCB1_mut", "ADCY1_mut", "ARFGAP3_mut", "CDC27_mut", "CNTNAP1_mut", "NKX3-1_mut", "PLXNA1_mut", "SMARCA1_mut", "WNT5A_mut", "ZFHX3_mut", "MNDA_mut", "CLCC1_mut", "PPP6C_mut", "ACSL3_mut", "C15orf55_mut", "CDK4_mut", "CRTC3_mut", "FAS_mut", "FCRL4_mut", "PER1_mut", "SYK_mut", "USP6_mut", "AHCTF1_mut", "ARHGAP29_mut", "ARHGEF2_mut", "CHD6_mut", "CYTH4_mut", "EIF4G3_mut", "FAF1_mut", "FANCI_mut", "IRF7_mut", "ITGA9_mut", "JMY_mut", "LNPEP_mut", "LRPPRC_mut", "MAGI2_mut", "MAP3K11_mut", "MAT2A_mut", "MCM3_mut", "MCM8_mut", "MFNG_mut", "PIK3C2B_mut", "POM121_mut", "RASA2_mut", "RHOT1_mut", "SMURF2_mut", "TRERF1_mut", "VIM_mut", "WASF3_mut", "WNK1_mut", "XRN1_mut", "ZNF638_mut", "GNA11_mut", "ARFGEF1_mut", "ARID4A_mut", "EPHA1_mut", "IRF2_mut", "PCSK5_mut", "PTPRF_mut", "STARD13_mut", "TNPO2_mut", "PPM1D_mut", "ARID5B_mut", "CCND1_mut", "ACACA_mut", "ARAP3_mut", "AXIN1_mut", "CTNND1_mut", "CUX1_mut", "DEPDC1B_mut", "DHX35_mut", "FOXA2_mut", "ING1_mut", "INPP4A_mut", "INPPL1_mut", "MLH3_mut", "NUP93_mut", "PGR_mut", "PLCG1_mut", "PLXNB2_mut", "ROBO2_mut", "SEC31A_mut", "SOX17_mut", "AMOT_mut","ASXL2_mut", "FTSJD1_mut", "LARP4B_mut", "MBD2_mut", "PHLPP1_mut", "RNF43_mut", "SACS_mut", "ZNRF3_mut"]
snvdata = snvdata[filtercolumns] # got it

########################################
#               Add CNA                # # get a dataframe {samples as instances, features = cna concerned regions} without counting the regions not related to the cancers
########################################
print "Reading CNA data"

cnagenes = pd.read_excel(basedir + '/Data/TableS2D.xlsx', sheetname="TableS2D-RACSs", skiprows={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, usecols=[1, 2, 3, 14], header=0)

# TableS2G - Copy Number Alterations found in cell lines i.e. we should eliminate from it any cna concerned region that doesnt figure in Iorio et al [new comment]
# RACs in cell lines, can subtract the 100+ cell lines that do not appear in the patient samples (old comment]
cnalist = pd.read_excel(basedir + '/Data/TableS2G.xlsx', sheetname="TableS2G-CellLinesRACSs_CNA", skiprows={0, 1, 2}, usecols=[1, 4, 6], header=0)

# TableS2F - Copy Number Alterations found in primary tumours i.e if a cna concerned region is not here we will take it out from out total of cna (cnalist)
# RACs in primary tumours, to use as filter to reduce number of columns to Iorio
cnafilterlist = pd.read_excel(basedir + '/Data/TableS2F.xlsx', sheetname="TableS2F-TumoursRACSs_CNA", skiprows={0, 1, 2}, usecols=[4], header=0)
cnafilterlist = set([x.split(" ")[0] for x in cnafilterlist["Region identifier"].values])

cnalist = cnalist.groupby("Sample")["Region identifier"].apply(",".join).reset_index() # join now with "," and split later with "," will separate joined but also words formed from factorisation while separating by ","
cnalist["Region identifier"] = cnalist["Region identifier"].apply(lambda x: str(x).split(","))
tempdf = cnalist["Region identifier"].apply(lambda x: '|'.join(pd.Series(x))).str.get_dummies()
cnadata = cnalist.join(tempdf)
del cnadata["Region identifier"]

for somecol in list(cnadata)[1:]:
	subfilter = somecol.split(" ")[0]
	if subfilter in cnafilterlist:
		pass
	else:
		del cnadata[somecol] # got it

########################################
#                Add RNA               # #get a dataframe {instances as cosmic ids, features = genes_expr limited to only the ones in Ioro et al}
########################################
print "Reading RNA data"
filtercolumns = [x.replace("_mut", "_expr") for x in filtercolumns]
convertme = {}
fin = open(basedir + "/Data/Mapping_biomart_HGNC.txt", "r")
for line in fin.read().split("\n"):
	try:
		convertme[line.split("\t")[0]] = line.split("\t")[1] + "_expr" # addition to dict
	except IndexError:
		pass
fin.close()

rnadata = pd.read_csv(basedir + '/Data/sanger1018_brainarray_ensemblgene_rma.txt', sep="\t")
rnadata = rnadata.transpose()
rnadata = rnadata.reset_index()

myheader = rnadata.iloc[0] # select a line (the title of the columns line here)
myheader["index"] = "COSMIC_ID"
rnadata = rnadata[1:]
rnadata = rnadata.rename(columns=myheader)
rnadata = rnadata.rename(columns=convertme, inplace=False)
rnadata = rnadata[[x for x in rnadata if x in filtercolumns]]  # got it # Apply filter  # to be continued...

########################################
#           Add methylation            # # eget a dataframe of only the informative sites with their methylation values
########################################
print "Reading methylation data"
informativeCPG = pd.read_excel(basedir + "/Data/TableS2J.xlsx", skiprows=[0, 1, 2])
informativeCPG = informativeCPG["HyperMethylated iCpG"].values
informativeCPG = ["COSMIC_ID"] + list(informativeCPG)

mapdic = {}
mapfile = pd.read_excel(basedir + "/Data/methSampleId_2_cosmicIds.xlsx", dtype=str)
for someindex in range(0, len(mapfile)): #to make a dict of {sentricID_sentricPosition : cosmic_id}
	mapdic[mapfile["Sentrix_ID"].values[someindex] + "_" + mapfile["Sentrix_Position"].values[someindex]] = mapfile["cosmic_id"].values[someindex]

methydata = pd.read_csv(basedir + "/Data/F2_METH_CELL_DATA.txt", sep="\t", dtype={"Unnamed: 0": str}) # a table whose colnames are the SentricID_sentricPositions

myheader = Counter([mapdic[x] for x in list(methydata)[1:]]) # loop on the the header of methydata (colnames), and for each colnames, get his corresponding cosmic id and stock it in a counter dict
errors = [k for k in mapdic if myheader[mapdic[k]] > 1] # creer une liste des erreurs et y collecter les entrees de mapdic if that entree (SentricID_sentricPositions) is counted more than once
for error in errors:
	del methydata[error] # delete the counted more than once frm methydata (needed to uniquely map)

methydata = methydata.rename(index=str, columns={"Unnamed: 0": "COSMIC_ID"})
methydata = methydata.transpose() # anytable can be shown in 2 dispositions. this function or .T change from one to another
methydata = methydata.reset_index() # rewrite o to infinite indexes infront of dataframe
newheader = methydata.iloc[0]
methydata = methydata[1:]
methydata = methydata.rename(columns=newheader)
methydata["COSMIC_ID"].replace(mapdic, inplace=True)
methydata = methydata[[x for x in list(methydata) if x in informativeCPG]] # got it # include in a new table a column when its title is in the informative list

########################################
#              Data merge              #
########################################
ctypes = set(drug_response["TCGA"])
# ctypes.add("PANCAN")  # Option to run PANCAN, not really computationally feasible

ctypes = {x for x in ctypes if x == x} # removes 'nan'
ctypes.remove("UNABLE TO CLASSIFY")
ctypes.remove("nan")
ctypes = sorted(ctypes) # we have the cancer types------>to loop on them later (loop 1)

# ----------an idea is to only launch a test analysis with the 5 ctypes from article :
# BRCA,COADREAD,LUAD,SCLC,SKCM
# a_given_list_of_ctypes = ["BRCA","COADREAD","LUAD","SCLC","SKCM"]
a_given_list_of_ctypes = ["BRCA"]  # use this for renewed test
ctypes = filter(lambda i: i in a_given_list_of_ctypes,ctypes)
#  -------------

# lets show the cancertypes we will run with
print "Analysing with these cancer types : " , ctypes

drugs = set(drug_response["DrugID"]) # we have the drugs ids
# ----------an idea is to only launch a test analysis with some drugs, use this k9 in the drugs loop insted of drugs (set) :
k9 = ['192']
# k9 = k9.append(list(drugs)[0])
# # list(drugs)[0]
print "Analysing with these drugs : " , k9



mylabels = ["snv", "cna", "rna", "methy"] # we have the profiles------>to loop on them later (loop 2)
mylabels = ["snv"] # use this for renewed test
print "Analysing with these profiles : " , mylabels


outfile = open(basedir + "/" + "/complexity_testing.txt", "w")
outfile.write("Seed" + "\t" + "Layer" + "\t" + "Ctype" + "\t" + "Drug" + "\t" + "Drug_Name" + "\t" + "Number_CellLines" + "\t" + "Train_set_size" + "\t" + "N/2" + "\t" "allFeat" + "\t" + "nOpt" + "\t" + "ValidationSet_Correlation" + "\t" + "Rs_test_OMC" + "\t" + "Rs_test_all" + "\t" + "Rs_test_OMC_Controlled" + "\t" + "RMSE_test_OMC" + "\t" + "RMSE_test_all" + "\t" + "R2_test_OMC" + "\t" + "R2_test_all" + "\t" + "Selected_Features" + "\t" + "Predictions_OMC" + "\t" + "Controlled_Preds_OMC" + "\t" + "Predictions_allFeat" + "\t" + "Predictions_observed" + "\n")
# we have the output file and the columns to fill


globalstart = datetime.datetime.now() #verify if its not better to move this time check closer to the analysis start


def r2(a, b):
	return 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)
# def r2(a, b):
#     metric1_r2 = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)
#     return metric1_r2

def rmse(a, b):
	return np.sqrt(np.sum((a - b) ** 2) / len(a))
# def rmse(a, b):
#     metric2_rmse = np.sqrt(np.sum((a - b) ** 2) / len(a))
# 	return metric2_rmse


print "Starting data analysis"
random.seed(1)
random_seeds = [1]

for ctype in ctypes: # loop in the ctypes and define each time a drug_response dataframe (cframe) by restricting to only actual considered ctype
	if ctype == "PANCAN":   #an option to work with all ctypes at the same time
		cframe = drug_response
	else:
		cframe = drug_response.loc[drug_response["TCGA"] == ctype] #take only rows of drug_response that are that ctype

	for drug in k9: # loop in the drugs (numbers here) and define each time a specific_cancer-drug_response dataframe (drugframe) by restricting to only actual considered ctype's and drug used
		drugframe = cframe.loc[cframe["DrugID"] == drug] #take only rows of drug_response that are that drug
		starttime = datetime.datetime.now() #verify if its not better to move this time check closer to the analysis start

		if len(drugframe) > 44: # verify if at least n=45 responses available # this if has no else does only the work or pass
			for featuretype in mylabels: # loop in the features and each time do the appending of the proper feature values to the specific_cancertype-drug_specific dataframe. in each case, the appending is based on both dtaframes containing the unique cosmic ids
				if featuretype == "snv": # snv condittion
					dframe = pd.merge(drugframe, snvdata, how="inner", left_on="Cell line cosmic identifiers", right_on="COSMIC_ID") # link with cosmic ids
					del dframe["COSMIC_ID"]
				else:
					if featuretype == "cna": # cna condittion
						dframe = pd.merge(drugframe, cnadata, how="inner", left_on="Sample Name", right_on="Sample") #link with sample names
						del dframe["Sample"]
						dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(bool) # change values 0 and 1 of cna into true and false
					else:
						if featuretype == "rna": # rna condittion
							dframe = pd.merge(drugframe, rnadata, how="inner", left_on="Cell line cosmic identifiers", right_on="COSMIC_ID")
							del dframe["COSMIC_ID"]
							dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(float) # change values of rna into floats

						else: # can only be meth condittion for sure because because only one non explored in mylabels (list of the profiles)
							dframe = pd.merge(drugframe, methydata, how="inner", left_on="Cell line cosmic identifiers", right_on="COSMIC_ID")
							del dframe["COSMIC_ID"]
							dframe[list(dframe)[5:]] = dframe[list(dframe)[5:]].astype(float) # change values of meth into floats

				dframe["LOG_IC50"] = dframe["LOG_IC50"].astype(float) # change values of response into floats

				try:
					if len(dframe) > 44: # verify if at least n=45 responses are still available
						seedscores = []
						for aseed in random_seeds:
							np.random.seed(aseed)
							myrun = []

							##########################################
							#          Train & test split            #
							##########################################

							dframe = dframe[list(dframe.loc[:, (dframe != 0).any(axis=0)])]
							dframe.sort_values("LOG_IC50", axis=0, ascending=True, inplace=True, kind='mergesort')
							dframe = dframe.reset_index(drop="True")

							indexer = float(len(dframe) - 2) / 9

							testselector = [1, int(np.ceil(indexer)), int(np.ceil(indexer * 2)), int(np.ceil(indexer * 3)), int(np.ceil(indexer * 4)), int(np.ceil(indexer * 5)), int(np.ceil(indexer * 6)), int(np.ceil(indexer * 7)), int(np.ceil(indexer * 8)), len(dframe) - 2]
							trainselector = sorted(set(range(len(dframe))) - set(testselector))

							# Training and test sets
							trainset = dframe.iloc[trainselector, :]
							trainset = trainset.reset_index(drop="True")
							testset = dframe.iloc[testselector, :]
							testset = testset.reset_index(drop="True")

							# Define maximal complexity of model
							maxfeatures = int(np.ceil(float(len(trainset)) / 2))
							complexities = range(2, maxfeatures) + [len(list(trainset[list(trainset)[5:]]))]
							complexities = list(sorted(set(complexities)))

							stratfolds = stratify(cv_standard, "LOG_IC50", trainset)

							##########################################
							#       Standard cross-validation        #
							##########################################

							OMC_estimator = {}
							OMC_pval = {}

							store_preds = {}
							store_obs = []

							for train, test in stratfolds:
								trainframe = trainset.iloc[train]
								trainframe.reset_index(drop=True)
								valframe = trainset.iloc[test]
								valframe = valframe.reset_index(drop=True)

								store_obs.append(valframe["LOG_IC50"].values)
								trainframe_feats = feat_select(featuretype, trainframe[list(trainframe)[5:]], trainframe["LOG_IC50"])

								sorter_trainframe = sorted(trainframe_feats, key=trainframe_feats.get, reverse=False)

								for complexity in complexities:
									ctrain = trainframe[sorter_trainframe[:complexity]]
									cval = valframe[sorter_trainframe[:complexity]]
									ctrain_y = trainframe["LOG_IC50"]
									cval_y = valframe["LOG_IC50"]

									model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700, colsample_bytree=0.8, silent=1, nthread=num_cores, seed=aseed)
									model.fit(ctrain, ctrain_y)

									try:
										store_preds[complexity].append(model.predict(cval))
									except KeyError:
										store_preds[complexity] = [model.predict(cval)]

									try:
										validation_OMC = scipy.stats.spearmanr(model.predict(cval), cval_y)
										try:
											OMC_estimator[complexity].append(validation_OMC[0])

										except KeyError:
											OMC_estimator[complexity] = [validation_OMC[0]]

										try:
											OMC_pval[complexity].append(validation_OMC[1])
										except KeyError:
											OMC_pval[complexity] = [validation_OMC[1]]
									except FloatingPointError:
										pass

							OMC = 0
							OMC_spearman = -1
							OMC_pvalue = 1

							for acomplexity in OMC_estimator:
								if np.min(OMC_estimator[acomplexity]) > 0.25:
									if len(OMC_estimator[acomplexity]) != 5:
										pass
									else:
										if np.median(OMC_estimator[acomplexity]) > OMC_spearman:
											OMC = acomplexity
											OMC_spearman = np.median(OMC_estimator[acomplexity])
											OMC_pvalue = np.median(OMC_pval[acomplexity])  # Disable when R2
										else:
											if np.median(OMC_estimator[acomplexity]) == OMC_spearman:
												if np.median(OMC_pval[acomplexity]) < OMC_pval:
													OMC = acomplexity
													OMC_spearman = np.median(OMC_estimator[acomplexity])
													OMC_pvalue = np.median(OMC_pval[acomplexity])
											else:
												pass
								else:
									pass

							if OMC == 0:
								pass
							else:
								store_preds = store_preds[OMC]
								store_preds = np.array([item for sublist in store_preds for item in sublist])
								store_obs = np.array([item for sublist in store_obs for item in sublist])
								slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(store_preds, store_obs)

								selected_feats = feat_select(featuretype, trainset[list(trainset)[5:]], trainset["LOG_IC50"])
								sorter_nopt = sorted(selected_feats, key=selected_feats.get, reverse=False)

								# OMC model training
								train_x = trainset[list(trainset)[5:]][sorter_nopt[:OMC]]
								final_train = xgb.DMatrix(train_x.values, label=trainset["LOG_IC50"].values, silent=True)

								OMCmodel = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700, colsample_bytree=0.8, silent=1, nthread=num_cores, seed=aseed)
								OMCmodel.fit(train_x, trainset["LOG_IC50"].values)

								# All features, no pre-ranking model training
								sorter_all = list(trainset)[5:]
								shuffle(sorter_all)
								train_x_noFS = trainset[list(trainset)[5:]][sorter_all]
								allfeat_train = xgb.DMatrix(train_x_noFS.values, label=trainset["LOG_IC50"].values, silent=True)

								allfeat_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700, colsample_bytree=0.8, silent=1, nthread=num_cores, seed=aseed)
								allfeat_model.fit(train_x_noFS, trainset["LOG_IC50"].values)

								test_x = testset[list(testset)[5:]][sorter_nopt[:OMC]]
								test_reg_y = testset["LOG_IC50"]
								test_x_noFS = testset[list(trainset)[5:]][sorter_all]

								OMCtest = xgb.DMatrix(test_x.values, label=test_reg_y.values)
								allfeattest = xgb.DMatrix(test_x_noFS.values, label=test_reg_y.values)

								# Evaluation
								try:  # Train, OMC
									spearman_train = scipy.stats.spearmanr(OMCmodel.predict(train_x), trainset["LOG_IC50"].values)
								except FloatingPointError:
									spearman_train = "FloatingPointError"

								try:  # Train, all features
									spearman_train_allFeat = scipy.stats.spearmanr(allfeat_model.predict(train_x_noFS), trainset["LOG_IC50"].values)
								except FloatingPointError:
									spearman_train_allFeat = "FloatingPointError"

								try:  # Test, OMC
									spearman_test = scipy.stats.spearmanr(OMCmodel.predict(test_x), test_reg_y.values)
								except FloatingPointError:
									spearman_test = "FloatingPointError"

								try:  # Test, all features
									spearman_test_allFeat = scipy.stats.spearmanr(allfeat_model.predict(test_x_noFS),
																				  test_reg_y.values)
								except FloatingPointError:
									spearman_test_allFeat = "FloatingPointError"

								controlled_preds = [(slope * x + intercept) for x in OMCmodel.predict(test_x)]

								try:
									spearman_test_controlled = \
										scipy.stats.spearmanr(controlled_preds, test_reg_y.values)[0]
								except FloatingPointError:
									spearman_test_controlled = "FloatingPointError"

								RMSE_OMC = rmse(OMCmodel.predict(test_x), test_reg_y.values)
								RMSE_all = rmse(allfeat_model.predict(test_x_noFS), test_reg_y.values)
								try:
									R2_OMC = r2(test_reg_y.values, OMCmodel.predict(test_x))
								except FloatingPointError:
									R2_OMC = "F"

								try:
									R2_all = r2(test_reg_y.values, allfeat_model.predict(test_x_noFS))
								except FloatingPointError:
									R2_all = "F"

								if OMC < 100:
									outfile.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(maxfeatures) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(OMC_estimator[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + " ".join(sorter_nopt[:OMC]) + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")
								else:
									outfile.write(str(aseed) + "\t" + featuretype + "\t" + ctype + "\t" + str(drug) + "\t" + "".join([str(x) for x in mydrugs[drug]]) + "\t" + str(len(dframe)) + "\t" + str(len(trainset)) + "\t" + str(maxfeatures) + "\t" + str(len(list(trainset[list(trainset)[5:]]))) + "\t" + str(OMC) + "\t" + str(np.median(OMC_estimator[OMC])) + "\t" + str(spearman_test[0]) + "\t" + str(spearman_test_allFeat[0]) + "\t" + str(spearman_test_controlled) + "\t" + str(RMSE_OMC) + "\t" + str(RMSE_all) + "\t" + str(R2_OMC) + "\t" + str(R2_all) + "\t" + ">100 features" + "\t" + str(OMCmodel.predict(test_x)).replace("\n", "") + "\t" + str(controlled_preds) + "\t" + str(allfeat_model.predict(test_x_noFS)).replace("\n", "") + "\t" + str(test_reg_y.values).replace("\n", "") + "\n")

						print featuretype, ctype, drug, "Done", datetime.datetime.now() - starttime
				except NameError:
					print ctype, drug, "Passed", datetime.datetime.now() - starttime
				del dframe
		del drugframe #get rid of drugframe
	del cframe # get rid of cframe

outfile.close()

globalend = datetime.datetime.now()
print "Time taken: ", globalend - globalstart

# okay fine testinng all now