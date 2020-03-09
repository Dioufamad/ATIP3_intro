from rpy2 import robjects
from rpy2.rinterface import RRuntimeError
import scipy
import locale
# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format

#  ---------------------------------------------------------------------------------------------------
# -------------------------------------- Function Definitions ---------------------------------------
# ---------------------------------------------------------------------------------------------------

# lets define a function taking as arg (presently computed featuretype, the part of the training folds frame with only the features columns, the part of the training folds frame with only the resp column)
def feat_select(feature_type, dmatrix_properties, dmatrix_logic50s):
    results_fs = {}  # Feature selection output

    if feature_type == "SNV" or feature_type == "CNA":  # t-test (for discrte values feats use Wilcoxonrank-sum test)
        for somecolumn in list(dmatrix_properties): # loop on the list of the feats (list of the columns names)
            mutindexes = dmatrix_properties[somecolumn].loc[dmatrix_properties[somecolumn] == True].index # give a list of indexes for the variant # strategy : for the whole frame get the lines having a given value; restrict that to the  frame but only cionsidering one column ; get indexes of the resulting lines (not needed to resptrict herte but it shows that in the spirit of the work only the resp column matter atthis point)
            wtindexes = dmatrix_properties[somecolumn].loc[dmatrix_properties[somecolumn] == False].index # give a list of indexes for the wt
            pval = 1

            try:  # NumPy wilcoxon not used due to it being nominal approximation
                wilcoxon2 = robjects.r['wilcox.test']
                v12 = robjects.FloatVector(dmatrix_logic50s[mutindexes].values) # the resp values for the mutated indexes
                v22 = robjects.FloatVector(dmatrix_logic50s[wtindexes].values) # the resp values for the wt indexes
                wilcox_result2 = wilcoxon2(v12, v22, conf_int=True)
                pval = wilcox_result2.rx2("p.value")[0]
            except RRuntimeError:
                pass  # If always mutated or not mutated, it is not single-handedly correlated with response
            results_fs[somecolumn] = pval  # Add p-values to dictionary

    else:  # spearman test  (for real values feats use corr coef of the spearman test) (does a test of correlation between 2 variable a and b : a is the presently considered column (feat.) and b is the resp column
        for somecolumn in list(dmatrix_properties):
            try:
                testresult = scipy.stats.spearmanr(dmatrix_properties[somecolumn].values, dmatrix_logic50s, axis=0, nan_policy='propagate')
                # in the var testresult, we have a tuple of [0] = correlation and [1] = pvalue.
                # lets extract the pvalue and add it as an entry of the dict resultfs
                results_fs[somecolumn] = testresult[1]
            except FloatingPointError:
                pass  # Same issue as for Wilcoxon, if always mutated or not, it alone is not correlated
    return results_fs
