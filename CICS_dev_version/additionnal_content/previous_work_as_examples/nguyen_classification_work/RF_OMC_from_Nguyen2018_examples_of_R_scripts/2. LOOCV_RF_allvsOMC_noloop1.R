#VALUES NEEDED THROUGHOUT THE SCRIPT
ntree = 1000 # number of trees (1out of 2 hyperparam of Rfs) !!!
mth = "RF" # tag on file/folders-names to hint for method used !!!
output.type = "BestResCat" ##!!! the best response (minimum delta volume) and best resp average (average responses set spanned by all t values, keep the minimum) -> make categories

library(randomForest) ##to get the method's library !!!

library(doParallel) ##!!!! needed to parallelize the foreach works (https://www.r-bloggers.com/lets-be-faster-and-more-parallel-in-r-with-doparallel-package/)
# detectCores() - 2
if(!exists("cl")) { # !!!
  cl <- makeCluster(detectCores() - 2)
  registerDoParallel(cl)
}

library(caret) # the Classification And REgression Training (CARET) package #!!!!

# create the folder for the workings #!!!!
if(all(dir("../" ) != paste(mth,".LOOCV2_predicted", sep = ""))) dir.create(paste("../",mth,".LOOCV2_predicted", sep = ""))
# create the folder for the results #!!!!
if(all(dir("../" ) != "results_selfeats")) dir.create("../results_selfeats/")

## LOADING DATA ABOUT PDXs AND THE TREATMENTS TESTED 
# get a table of 3 columns : [Treatment ID (not in order)],[treatment details usually name or combinaison used],[count.models as number of models for that treatment]
treatments = read.csv("../inp/pdxe_treatments50.csv", sep = ",")
# number of rows i.e of treatments registered in table treatments
ntreatments = nrow(treatments)
# table with 2 columns : [model ID], [cancer type that is considered]
cancer.type = read.csv("../inp/pdxe_cancertype.csv", sep= ",")

# starting time
start.time <- Sys.time()
## MODEL BUILDING
# loop on illnesses as blocked for noloop1
# for (cancertype in c("BRCA","CRC")){
for (cancertype in c("BRCA")){ #trial1
  start.time.cancertypeloop <- Sys.time()
  # cancertype = "BRCA" #noloop1 cond
  if(all(dir(paste("../",mth,".LOOCV2_predicted", sep = "")) != cancertype)) dir.create(paste("../",mth,".LOOCV2_predicted/",cancertype, sep = ""))
  # loop on the profiles as blocked for noloop2
  # type = c("SNV", "CN","CNA", "GEX")
  type = c("GEX") #trial1
  for (m in 1:length(type)){
    feat.type = type[m] # fixate the presently probed profile
    # feat.type = "GEX" #noloop2 cond
    # fixate selection method (fisher, ttest) following feature type is resp. categorical (SNV, CNA) or continuous (GEX,CN)
    sel.meth = ifelse(feat.type %in% c("SNV", "CNA"),"fisher","ttest")
    M2 = data.frame(matrix(NA, ncol = 38, nrow = ntreatments)) # a dataframe that will be used to collect this part results
    # loop on the treatments as blocked for noloop3
    # for (i in 1:ntreatments){
    for (i in c(12)){ # trial1
      # i = 12 #noloop3 cond 
      # get the treatment ID in the table treatments
      treatmentid = treatments$Treatment_ID[i]
      treatmentName = treatments$Treatment_Details[i]
      #create the template of the processed file's name #!!!!
      treatment_response_file = paste("../processedData.",feat.type,"/", cancertype, "_Treatment",treatmentid,"_",output.type,"_",feat.type,".csv",sep = "")
      # verify if file exists and if so get resp data and N/2 otherwise just show a red font message
      if(file.exists(treatment_response_file)){ #blocked as noloop4
        inpfile = treatment_response_file
        #  display the experience done : treatment ID, its name, the cancer type tested on, the profile probed
        cat(paste("Treatment ",treatmentid," - ",treatmentName, " has been tested on ",cancertype," and probed profile is ",feat.type, "\n",sep = ""))
        # read the processed file and store it in var data
        #(a 38 lines/models_studied/PDXs and 15234 columns/variables/genomic_regions_variation_state)
        data = read.csv(file = inpfile,sep = ",", header = TRUE)
        dataBin = data[,1] # get the 1st column of data
        # count and divise by 2 the numbers of values in the 2 classes (to have at least 2 datapoints and dodge a bit HD problem)
        count = length(unique(data$Model)) #count the lines of response columns without count the eventuals duplicated lines
        maxfeats = round(count/2)  ## Set Maxfeats = N/2
        #########################suite
        # verify if there is not class sparsity and if not so, create the directories to needed for the predictions :
        # if (sum(data[,1] %in% c("CR","PR","SD")) >= 5 & sum(data[,1] == "PD") >= 5){ # change this condittion because data shows Sen & Res as classes instead of CR PR SD (now Sen) and PD (as Res). Codename : deprecation1
        if (sum(data[,1] %in% c("Sen")) >= 5 & sum(data[,1] == "Res") >= 5){  # deprecation1 applied
          ##-for a main predictions dir>illness>treatment.profile (case of OMC) !!!
          if(all(dir(paste("../",mth,".LOOCV2_predicted/",cancertype, sep = "")) != paste(treatmentName,feat.type,sep = "."))) {
            dir.create(paste("../",mth,".LOOCV2_predicted/",cancertype,"/",paste(treatmentName,feat.type,sep = "."), sep = ""))}
          ##-for a main predictions dir>illness>treatment.profile.allfts (case of all features used) !!!
          if(all(dir(paste("../",mth,".LOOCV2_predicted/",cancertype, sep = "")) != paste(treatmentName,feat.type,"allfts",sep = "."))) {
            dir.create(paste("../",mth,".LOOCV2_predicted/",cancertype,"/",paste(treatmentName,feat.type,"allfts",sep = "."), sep = ""))}
          
          # create the vectors to collect the metrics of performance either for one seed and for being the one to be tranformed to give the collector of a metric for all seeds 
          # (this final is the reason why we init it here b4 the loop on seeds)
          il1_col_med_il2_col_omc = c() #P1I a list of the median of the best features at each seed; transmit a median of it; to self
          ol1_MCC_col_omc_for_1seed = c() #P2I a list of the mcc values of each mc; transmit a med of it; to med
          ol1_MCC_col_allfts_for_1seed = c() #P4I a list of the mcc values of each all_model; transmit a med of it; to med
          ol1_PREC_col_omc_for_1seed = c() #P6I transmit a median of it; to self
          ol1_RECALL_col_omc_for_1seed = c() #P7I transmit a median of it; to self
          ol1_SPEC_col_omc_for_1seed = c() #P8I transmit a median of it; to self
          ol1_F1_col_omc_for_1seed = c() #P9I transmit a median of it; to self
          ol1_col_num_pred_call_as_cPos_w_omc_by_1seed = c() #P10I transmit a sum of it; to self
          
          # loop on the seed values (10 seeds with the zero being 5678 and the rest well you know it...)
          # n = 1
          # seeds = c(5678,1111,2222,3333,4444,5555,6666,7777,8888,9999)
          seeds = c(5678,1111) # trial1
          for (n in 1:length(seeds)){ # loop on the seeds each is n #outer loop 1
            seed1 = seeds[n] # fixate the actual seed
            # init 3 : 
            #1--# a prelim for the best fts at each fold (LOO so many folds) # list of # of fts in each seed analysis to compute the median over it bcuz contains values in of samples in the LOO
            # best.feats.list = c() #P1_prelim_I ## test if really in use or needed ???
            #2---one collector by model, to stock the classes predicited for each fold of the val set of the (all model) (one class because its after prob thres is used) 
            ol2_col_pred_call_1by_seed_w_allfts = c() # (for all model)
            ol2_col_pred_call_1by_seed_w_omc = c() # (for omc)
            #3---#a dataframe by model, to keep track of predictions probs of classes in folds with only the header  : fold, pdxindex, res, sen (the two class in the same order they are given to the classwt) 
            print.pred = data.frame(matrix(NA, ncol = 4, nrow = 0)) 
            colnames(print.pred) = c("Fold", "PDXindex","Res","Sen") # (for the OMC)
            print.pred.all = data.frame(matrix(NA, ncol = 4, nrow = 0))
            colnames(print.pred.all) = c("Fold", "PDXindex","Res","Sen") # (for the all version)
            
            # folds creation 
            ol2_folds <- caret::createFolds(dataBin,k = nrow(data)) #(gives a list of numbers from 1 to num of folds to iterates on) # uses response col and # of folds wanted, here LOO so nrow of all_data
            # j = 1
            for(j in 1:nrow(data)){ #outer loop 2 (each fold for LOO on the training set 
              #------FULL DATASET MODEL : mk training and testing data for each fold of LOO (use j to def sets and get the predictions )
              #-- mk train for each fold (!!! training on all the data with LOOCV)
              # build the folds data 
              ol2_train_data = data[-ol2_folds[[j]],] #Set the training set (for a row j of data, select all rows of data without the row of index 1 (here the frame start index at 1)
              ol2_test_data = data[ol2_folds[[j]],] #Set the validation set (for a row j of data, select only the row j)
              #train_y, test_y of trainset_fold 
              ol2_train_y = ol2_train_data[,1] #keep the training set response column values (37 values) (for the row j of data analysed, the 1st value)
              ol2_test_y = ol2_test_data[,1]  #same (1 values)
              #train_x, test_x of trainset_fold
              ol2_train_x = ol2_train_data[,-c(1,ncol(ol2_train_data))] # keep only the features (index 1 to ncol) #here are dropped the model and the response #odd code tho because _c(i,j) drops cols i to j
              ol2_test_x = ol2_test_data[,-c(1,ncol(ol2_test_data))]
              
              #----intialise the classifier params ## find a suffix for each training equipement or even isolate
              mtry_mdl_allfts = round(sqrt(ncol(ol2_train_x))) #see docs as it is the default value
              # sens_mdl_allfts = length(which(ol2_train_y %in% c("CR","PR","SD"))) # deprecation1 applied
              sens_mdl_allfts = length(which(ol2_train_y %in% c("Sen"))) # deprecation1 applied # num of sens
              # res_mdl_allfts = length(which(ol2_train_y == "PD")) # deprecation1 applied
              res_mdl_allfts = length(which(ol2_train_y == "Res")) # deprecation1 applied # num of res
              prop.sens_mdl_allfts = sens_mdl_allfts/nrow(ol2_train_x) # proportion of class "sen"
              prop.res_mdl_allfts  = res_mdl_allfts/nrow(ol2_train_x) # proportion of class "res"
              #---train the model on train_x,train_y
              set.seed(seed1)
              rf.mdl_allfts_fitted = randomForest(ol2_train_x,ol2_train_y,mtry = mtry_mdl_allfts,importance = TRUE,ntree = ntree, classwt = c(Res = prop.sens_mdl_allfts,Sen = prop.res_mdl_allfts)) ## class weighting with classwt #to be check if not inversed
              #last_stop_we
              # -- mk test for each fold
              ol2_pred_2probs_mdl_allfts = predict(rf.mdl_allfts_fitted,ol2_test_x, type = "prob") # keep the prediction (a matrix of prob, one line for each sample in the test, one prob for each class in fashion of "what is prob of having each class")
              ol2_pred_call_mdl_allfts = c()
              thres = 0.5 # initialise at start of script
              for (r in 1:nrow(ol2_pred_2probs_mdl_allfts)){ ## solate in func of pos-neg calling following thres
                if(ol2_pred_2probs_mdl_allfts[r,1] > thres) { # 1 for class in col one and it is class cited 1st in classwt
                  ol2_pred_call_mdl_allfts[r] = "Res"           # predictions of all samples in the val set of the fold # affect class 1 if prob column 1 is > thres because col 1 is class 1 
                }else{
                  if(ol2_pred_2probs_mdl_allfts[r,1] == thres) { ## change it with an elif
                    set.seed(5678) #???
                    ol2_pred_call_mdl_allfts[r] = sample(c("Res","Sen"),1) # take one at random but restart the seed runner to hahve another chaznce to dodge this situation (will happens its dodged)
                  } else {
                    ol2_pred_call_mdl_allfts[r] = "Sen"} # only case left
                }
              }# end of calling pos or neg #last_stop
              #---adding to a collector for stats or metrics
              names(ol2_pred_call_mdl_allfts) = rownames(ol2_test_data) # name the elts of the pred of all samples in val set as the names of the rows in the test set (one sample pred get back the # of row it had)
              ol2_col_pred_call_1by_seed_w_allfts = c(ol2_col_pred_call_1by_seed_w_allfts,ol2_pred_call_mdl_allfts) # add the predictions of all samples ( one really) to a collector
              #---adding to keep tracker of predictions in folds ##isolate
              # stitch together 4 cols ("Fold", "PDXindex","Res","Sen") to form a table : 1-the fold x the # of lines in it; 2-the sample name; 3-2 cols that are the content of the predicrtions in the val set
              rownames(ol2_pred_2probs_mdl_allfts)= rownames(ol2_test_data) # get the names of samples attached to the predictions made
              ol2_pred_2probs_mdl_allfts = cbind(rep(j, nrow(ol2_pred_2probs_mdl_allfts)),rownames(ol2_pred_2probs_mdl_allfts),ol2_pred_2probs_mdl_allfts) # stich foldsxtimes # of samples and the predictions with their names before
              colnames(ol2_pred_2probs_mdl_allfts) = c("Fold", "PDXindex","Res","Sen") # name the cols
              print.pred.all = rbind(print.pred.all,ol2_pred_2probs_mdl_allfts) # push into a keep tracker of predictions probs in folds, these formatted results of prediction for a fold 
              # ----------------------end of all_model predictions making and stashing
              #------OMC MODEL
              #>>>>>>>>>>>MAKE # OF FOLDS FOR CV
              il1_folds <- caret::createFolds(ol2_train_y,k = nrow(ol2_train_data)) #create the folds for the LOO for the omc model
              
              # MCC.tst.inner = c() #??? # not used
              M3 = foreach(l = 1:nrow(ol2_train_data),.packages=c('randomForest','caret'),.combine = 'rbind', .inorder = TRUE) %dopar%{ # inner loop 1 #parallelise 37 jobs on each of on 37 folds/samples # the packages necessary are given # the results are combined in row after row fashion to form a matrix #and that in order of the samples
                #>>>>>>>>>>>>PRE-BUILDING THE SETS NEEDED
                #---1-making of trainframe and valframe
                il1_train_data = ol2_train_data[-il1_folds[[l]],] #Set the training set of 36 (by dropping one row/sample)
                il1_test_data = ol2_train_data[il1_folds[[l]],] #Set the validation set of 1 (by keeping only that previously dropped sample)
                # trainframe_y & valframe_y
                il1_train_y = il1_train_data[,1] # train_y
                il1_test_y = il1_test_data[,1]   # val_y
                # trainframe_x & valframe_x
                il1_train_x = il1_train_data[,-c(1,ncol(il1_train_data))] #train_x (## should not transmit to self but to new name)
                il1_test_x = il1_test_data[,-c(1,ncol(il1_test_data))] #val_x
                #<<<<<<<<<<<<<<END OF PRE-BUILDING THE SETS NEEDED
                #>>>>>>>>>>>>>>START OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
                #---2-estimate the fts with two-sided fisher exact test p-values or two sided unpaired t-test p-values
                oob.list = c()#??? # not used
                nfeats.list = c()#??? # not used
                mtry1 = round(sqrt(ncol(il1_train_x))) # param mtry as default 
                # sens_ol1 = length(which(il1_train_y %in% c("CR","PR","SD"))) # deprecation1 applied
                sens_ol1 = length(which(il1_train_y %in% c("Sen"))) # deprecation1 applied ## proportion or classes (to correct)
                # res_ol1 = length(which(il1_train_y == "PD")) # deprecation1 applied
                res_ol1 = length(which(il1_train_y == "Res")) # deprecation1 applied
                prop.sens_ol1 = sens_ol1/nrow(il1_train_data)
                prop.res_ol1  = res_ol1/nrow(il1_train_data)
                set.seed(seed1) # setting the seed
                #training on training
                rf1 = randomForest(il1_train_x,il1_train_y,mtry = mtry1,importance = TRUE,ntree = ntree, classwt = c(Res = prop.sens_ol1,Sen = prop.res_ol1)) ## class weighting
                # rf2 = randomForest(ol2_train_data,ol2_train_y,mtry = mtry1,importance = TRUE,ntree = ntree, sampsize = c(min(c(sens,res)),min(c(sens,res)))) ## class weighting
                
                il1_pred_2probs_w_allfts = predict(rf1,il1_test_x, type = "prob") # predictions probs following the classes for all samples in val set
                il1_pred_call_w_allfts = c() # the predictions that are called after thres decision
                thres = 0.5
                for (r in 1:nrow(il1_pred_2probs_w_allfts)){
                  if(il1_pred_2probs_w_allfts[r,1] > thres) {
                    il1_pred_call_w_allfts[r] = "Res"
                  }else{
                    if(il1_pred_2probs_w_allfts[r,1] == thres) {
                      set.seed(5678)
                      il1_pred_call_w_allfts[r] = sample(c("Res","Sen"),1)
                    } else {
                      il1_pred_call_w_allfts[r] = "Sen"}
                  }
                }
                #<<<<<<<<<<<<<<END OF PREDICTION WITH ALL FTS (TO COMPARE WITH MC MODELS)
                #>>>>>>>>>>>>>>ESTIMATING FTS FOR FTS-RANKING
                #---process of limiting training data to only fts who shows variation (at least one variation in a sample) ## isolate 
                u=apply(il1_train_x, 2, function(col) length(unique(col))) # a c() of the sum of the uniques values in each col
                selCols=which(u != 1)
                il1_train_x = il1_train_x[,selCols]
                #last_stop_monday
                #---feature selection using univariate and fts ranking : capture pvals.ft or pvals.tt for all the fts that are variant  (a vector of fts's p values in the order of the fts)
                if(sel.meth == "fisher"){ ## isolate the initialisation of method selection at start of script (seen before)
                  pvals.ft = apply(il1_train_x,2,function(x){ ## isolate this method in module
                    contigency.tab = matrix(c(length(x[which(il1_train_y == "Sen" & x == 1)]), #TP #in a matrix of 2 rows ie row 1 ##Sen is Pos here (good) so install harmony vs all_model
                                              length(x[which(il1_train_y == "Sen" & x == 0)]), #FP row 2
                                              length(x[which(il1_train_y == "Res" & x == 1)]), #FN row 1
                                              length(x[which(il1_train_y == "Res" & x == 0)])),#TN row 2 
                                            nrow = 2,
                                            dimnames = list(Gene = c("Mutated", "WT"), # rownames  (mut=being in row1, res=being in row2)
                                                            Response = c("Sen", "Res"))) # colnames (sen=being in col1, res=being in col2) hence the precedent build of contingency matrix
                    fisher.test(contigency.tab, alternative = "two.sided")$p.value # output the p-value of comparing the cols Sen and res in the contingency matrix and it will be attributed to ft (col) to compared the features later and rank them
                  })
                }else{ # case of reals values fts
                  pvals.tt = apply(il1_train_x,2, 
                                   function(x){ ## isolate this method in module
                                     # check if the arrays have a single value in all cases and 
                                     # if these values are different between sensitive and resistant cases (case the values changes but you dont learn really)
                                     if (length(unique(x[which(il1_train_y == "Sen")])) == 1 & # which(il1_train_y == "Sen") means index of rows that have class Sen # si tous ces indices sont une seule unique value
                                         length(unique(x[which(il1_train_y == "Res")])) == 1){ # et meme chose pour Ren alors...
                                       if (unique(x[which(il1_train_y == "Sen")]) != unique(x[which(il1_train_y == "Res")])){ -1 } else { 1 } #...si en plus de cela, ces deux grp de lignes sont differentes, adjuger -1 de p_value (val utlime de p_value pour meilleure cap de classification) 
                                       # autrement et toujours dans le cas "1 unique Ren et 1 unique Sen", mettre 1 comme p_value (car cela semble random)
                                     }else { # cas ou 2 groupes de valeurs uniques correpondantes aux deux classes nont pas lieu, faire le t-test
                                       a = t.test(x[which(il1_train_y == "Sen")],x[which(il1_train_y == "Res")],paired = F) # unpaired t-test so paired =false
                                       a$p.value} # get the p_value from the results just like in the 
                                   })
                }
                #<<<<<<<<<<<END OF ESTIMATING FTS FOR FTS-RANKING
                #>>>>>>>>>>>MCs PREDICTIONS MAKING
                #---3-training a model for each MC and test it to compare them and take the best OMC
                il2_col_il2_pred_call_mc = c() # a collector of prediction called at each loop on complxities
                for (mc in 2:maxfeats){ #inner loop 2 # loop on the MCs
                  if(sel.meth == "fisher") topfeats = names(pvals.ft[order(pvals.ft, decreasing = F)][1:mc]) else topfeats = names(pvals.tt[order(pvals.tt, decreasing = F)][1:mc])
                  ## isolate as ranking function for p_value
                  # rank it, keep this rank from original, get the names, get the top MC or those names) ## try to gives the rank at end of estimating fts to select in it directly here
                  il2_train_mc_x = il1_train_x[,which(colnames(il1_train_x) %in% topfeats)] # train with MC fts
                  il2_test_mc_x  = il1_test_x[,which(colnames(il1_test_x) %in% topfeats)] # test with MC fts
                  # set up a training of the built with elected fts MC model
                  mtry2 = round(sqrt(ncol(il2_train_mc_x))) # should be lower than used to because fs
                  set.seed(seed1)
                  rf2 = randomForest(il2_train_mc_x,il1_train_y,mtry = mtry2,importance = TRUE,ntree = ntree, classwt = c(Res = prop.sens_ol1,Sen = prop.res_ol1)) ## class weighting #same proportions of class than ol1
                  # rf2 = randomForest(ol2_train_data,ol2_train_y,mtry = mtry1,importance = TRUE,ntree = ntree, sampsize = c(min(c(sens,res)),min(c(sens,res)))) ## class weighting
                  # set up a prediction to estimate later the best MC with results
                  il2_pred_2probs_w_mc = predict(rf2,il2_test_mc_x,type = "prob")
                  #call the prediction and stash it
                  il2_pred_call_mc = c()
                  thres = 0.5
                  for (r in 1:nrow(il2_pred_2probs_w_mc)){
                    if(il2_pred_2probs_w_mc[r,1] > thres) {
                      il2_pred_call_mc[r] = "Res"
                    }else{
                      if(il2_pred_2probs_w_mc[r,1] == thres) {
                        set.seed(5678)
                        il2_pred_call_mc[r] = sample(c("Res","Sen"),1)
                      } else {
                        il2_pred_call_mc[r] = "Sen"}
                    }
                  }
                  il2_col_il2_pred_call_mc = c(il2_col_il2_pred_call_mc,il2_pred_call_mc) # stash the prediction
                } #end of inner loop 2 (loop on complexities)
                c(il1_pred_call_w_allfts,il2_col_il2_pred_call_mc,rownames(il1_test_data)) # vector result of job (1 to 37)
                # ... = a concatenation of 3 : 
                # (collector of predictions called for a fold before ranking fts (count 1 as collector init at each fold), collector of predictions called to estimate a mc(count 18 for 2:n/2=19 as collector init before start of each mc),sample name for resp in test of inner lopp1(count 1 as test has one sample bcuz LOO))
                # so vector result i (i in 1:37) containing 1,18,1 = 20 values
              } # end of inner loop 1 (loop on folds) ###last_stop
              #<<<<<<<<<<<<END OF PREDICTIONS MAKING
              #>>>>>>>>>>>ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
              #--- compute the mccs (1 for : the prediction in each fold, for all 37 folds) and (18 for : the 18 predictions in the 18 mc in a fold, for all 37 folds) #last_stop_tuesday
              calculate.mcc = function(predict, observe){
                TP = length(which(predict == "Sen" & observe == "Sen"))    # cells predicted sensitive which are actually sensitive (cell sensitive is +ve) ## its good as like said earlier sen is the pos class 
                TN = length(which(predict == "Res" & observe == "Res"))  # cells predicted resistant which are actually resistant (cell resistant is -ve)
                FP = length(which(predict == "Sen" & observe == "Res")) # cells predicted sensitive which are actually resistant (wrong prediction)
                FN = length(which(predict == "Res" & observe == "Sen"))       # cells predicted resistant which are actually sensitive (wrong prediction)
                dum1 = ((TP*TN)-(FP*FN))
                dum2 = sqrt(1.0*(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # 1.0 makes the multiplication float, thus avoiding integer overflow # remake formula to follow cycle of Pedro schematics
                if(dum2 == 0) dum2 = 1
                mcc = dum1/dum2
              } ##def isolate 
              ol2_train_y = ol2_train_y[match(rownames(ol2_train_data),as.numeric(M3[,ncol(M3)]))] 
              #as.numeric(M3[,ncol(M3) is a numeric vector that is all row of M3 and only the last col of M3 (it is normaly the rownames of different il1_test_data that been gone through ie all folds line succesively)
              #...rownames(ol2_train_data) are the names of samples 
              #...match(ofthesetwo)=a vector of the rownames that correspond ie the only ones that are here if all is okay
              #...giving this c() to select on ol2_train_y select only the resp values of those 2:378(37) samples we have the preds concatenated in M3
              med.MCC = apply(M3[,-ncol(M3)], 2, function(x) calculate.mcc(x,ol2_train_y)) # an added row of MCC as result
              # M3[,-ncol(M3)] M3 wo last col of M3
              # ...to each of the 19 cols that stays (1 pred before ranking, 18 for 1 in each of 18 mc), apply a calc of mcc to the col
              # ...result will be for a col icluding 37 folds so its a med (## say its a mean maybe...or check with Alex)
              #---choosing the best mc ie the omc
              omc_il1 = if (med.MCC[1] <= max(med.MCC[-1])) which.max(med.MCC[-1])+1 else ncol(data)-2 # rename as nfts_best_mc
              # c(-i) drops the element of index i of the vector #last_stop
              # which.max(med.MCC[-1]) give pos of the max of the 18 med.MCCs # +1 is added to have the mc value (2:n/2=19 with) of the col (bcuz 1st col is mc=2 and last col ie 18 is mc=19, ## do isolated diff indice mcs and col = 2-1 and add it as 1)
              # else : its # of all fts = " fts in data = ncol(data)-1col as resp - col(sample) = ncol(data) -2 = ncol(test_x ie il1_test_data)
              il2_col_of_omc_il1 = c(il2_col_of_omc_il1,omc_il1) #P1_prelim_U # num feats omc found and added to a list # a collector of nfts_omc_med_over_all_folds init before outer loop is updated
              #...contains the best_fts_mc at each fold of LOO of the outer loop2 ie 38 values
              ol2_train_data = ol2_train_data[,-c(1,ncol(ol2_train_data))] #train_x and test_x and remade
              ol2_test_data = ol2_test_data[,-c(1,ncol(ol2_test_data))]
              #<<<<<<<<<<<<<<<END OF ELECTION OF OMC BASED ON PREDICTION PERFORMANCE OF MCs
              #>>>>>>>>>>>>>START OF OMC PREDICTIONS MAKING
              # process of limiting data to conly cols that are informative (ie with variation and not only one value) (## isolate at first at top and use it)
              u=apply(ol2_train_data, 2, function(col) length(unique(col)))
              selCols = which(u != 1)
              ol2_train_data = ol2_train_data[,selCols]
              # fts ranking like earlier (in fact the same so ## isolate and reuse) # diff is earlier it was for 36 and here it is for 37 (train)
              if(sel.meth == "fisher"){
                pvals.ft = apply(ol2_train_data,2,function(x){
                  contigency.tab = matrix(c(length(x[which(ol2_train_y == "Sen" & x == 1)]),
                                            length(x[which(ol2_train_y == "Sen" & x == 0)]),
                                            length(x[which(ol2_train_y == "Res" & x == 1)]),
                                            length(x[which(ol2_train_y == "Res" & x == 0)])),
                                          nrow = 2,
                                          dimnames = list(Gene = c("Mutated", "WT"),
                                                          Response = c("Sen", "Res")))
                  fisher.test(contigency.tab, alternative = "two.sided")$p.value
                })
              }else{
                pvals.tt = apply(ol2_train_data,2,
                                 function(x){
                                   # check if the arrays have a single value in all cases and 
                                   # if these values are different between sensitive and resistant cases
                                   if (length(unique(x[which(ol2_train_y == "Sen")])) == 1 & 
                                       length(unique(x[which(ol2_train_y == "Res")])) == 1){
                                     if (unique(x[which(ol2_train_y == "Sen")]) != unique(x[which(ol2_train_y == "Res")])){
                                       -1 } else { 1 }
                                   }else {
                                     a = t.test(x[which(ol2_train_y == "Sen")],x[which(ol2_train_y == "Res")],paired = F)
                                     a$p.value}
                                 })
              }
              # rank the fts and get their prediction called
              if (omc_il1 == ncol(data) -2){ #" to review again # basically its if nfts = all_var gives pred_prob of all_model otherwise use the function for ranking fts with p_values ## normally isolate earlier
                ol2_pred_2probs_w_omc = predict(rf.mdl_allfts_fitted,ol2_test_data, type = "prob")
                ol2_pred_call_w_omc = ol2_pred_call_mdl_allfts # got it2
              } else {
                if(sel.meth == "fisher") topfeats = names(pvals.ft[order(pvals.ft, decreasing = F)][1:omc_il1]) else topfeats = names(pvals.tt[order(pvals.tt, decreasing = F)][1:omc_il1])
                #restrict train and test to fts of omc
                train.data2 = ol2_train_data[,which(colnames(ol2_train_data) %in% topfeats)]
                test.data2  = ol2_test_data[,which(colnames(ol2_test_data) %in% topfeats)]
                # the training equipement ## find a suffix for each training equipement or even isolate
                mtry3 = round(sqrt(ncol(train.data2)))
                # sens_mdl_allfts = length(which(ol2_train_y %in% c("CR","PR","SD"))) #deprecation1 applied
                sens_mdl_allfts = length(which(ol2_train_y %in% c("Sen"))) #deprecation1 applied
                # res_mdl_allfts = length(which(ol2_train_y == "PD")) #deprecation1 applied
                res_mdl_allfts = length(which(ol2_train_y == "Res")) #deprecation1 applied
                prop.sens_mdl_allfts = sens_mdl_allfts/nrow(train.data2)
                prop.res_mdl_allfts  = res_mdl_allfts/nrow(train.data2)
                set.seed(seed1)
                rf3 = randomForest(train.data2,ol2_train_y,mtry = mtry3,importance = TRUE,ntree = ntree, classwt = c(Res = prop.sens_mdl_allfts,Sen = prop.res_mdl_allfts)) ## class weighting
                
                set.seed(5678)
                ol2_pred_2probs_w_omc = predict(rf3,test.data2, type = "prob") # predictions of class prob
                ol2_pred_call_w_omc = c() # prediction called ## isolated
                thres = 0.5
                for (r in 1:nrow(ol2_pred_2probs_w_omc)){
                  if(ol2_pred_2probs_w_omc[r,1] > thres) {
                    ol2_pred_call_w_omc[r] = "Res"
                  }else{
                    if(ol2_pred_2probs_w_omc[r,1] == thres) {
                      set.seed(5678)
                      ol2_pred_call_w_omc[r] = sample(c("Res","Sen"),1)
                    } else {
                      ol2_pred_call_w_omc[r] = "Sen"}
                  }
                }
                names(ol2_pred_call_w_omc) = rownames(ol2_test_data) # gives the name of the sample to collector of prediction call         #ol2_col_ol2_pred_call_mc #ol2_pred_call_w_omc
                rownames(ol2_pred_2probs_w_omc)= rownames(ol2_test_data) # gives the name of the sample to predition # got it2
              } # end of judging the feature ranking of the omc in all folds of the training data
              #---adding to keep tracker of predictions in folds of outer loop 2 but for the omc model ##isolated
              ol2_pred_2probs_w_omc = cbind(rep(j, nrow(ol2_pred_2probs_w_omc)),rownames(ol2_pred_2probs_w_omc),ol2_pred_2probs_w_omc)
              colnames(ol2_pred_2probs_w_omc) = c("Fold", "PDXindex","Res","Sen")
              print.pred = rbind(print.pred,ol2_pred_2probs_w_omc)
              ol2_col_pred_call_1by_seed_w_omc = c(ol2_col_pred_call_1by_seed_w_omc,ol2_pred_call_w_omc)
              # >>>>>>>>>> END OF OMC PREDICTIONS MAKING
            } # end of outer loop 2 (on each fold for LOOCV on all training set, predictions are reaped)
            #>>>>>>>>>>>>METRICS computations FOR EACH SEED
            #~~~~~(finishing touches for outer loop 1 collectors)
            # lets report the predictions with each seed in a file from the temp dataframe ## no need to write, juste make df of it
            write.csv(print.pred, row.names = F,file = paste("../",mth,".LOOCV2_predicted/",cancertype,"/",paste(treatmentName,feat.type,sep = "."),"/Predicted_Replicate",n,".csv", sep = ""))  #omc
            write.csv(print.pred.all, row.names = F,file = paste("../",mth,".LOOCV2_predicted/",cancertype,"/",paste(treatmentName,feat.type,"allfts",sep = "."),"/Predicted_Replicate",n,".csv", sep = "")) #allmodel
            #lets add a median of all the OMCs from OL2, one from each fold on the training set (getting one OMC estimation by seed) # P1 (P1U)
            il1_col_med_il2_col_omc = c(il1_col_med_il2_col_omc,median(il2_col_of_omc_il1)) #P1U # a list that collect the # of fts found in at least 50% at each seed analysis
            # ~~~~~~(for the omc)
            # 0: the predictions called for the full dataset LOO folds
            ol2_col_pred_call_1by_seed_w_omc = ol2_col_pred_call_1by_seed_w_omc[match(rownames(data),names(ol2_col_pred_call_1by_seed_w_omc))] # 2 prelims #??? # affect samples names as they are the same number
            # 1: define mcc and calculate it . 
            ## for next computation, mk fucntion in module and create classes reaped from dframe after mgmt4 or use 1 and 0 (sens = 1 and res = 0)
            TP = length(which(ol2_col_pred_call_1by_seed_w_omc == "Sen" & dataBin == "Sen"))    # cells predicted sensitive which are actually sensitive (cell sensitive is +ve)
            TN = length(which(ol2_col_pred_call_1by_seed_w_omc == "Res" & dataBin == "Res"))  # cells predicted resistant which are actually resistant (cell resistant is -ve)
            FP = length(which(ol2_col_pred_call_1by_seed_w_omc == "Sen" & dataBin == "Res")) # cells predicted sensitive which are actually resistant (wrong prediction)
            FN = length(which(ol2_col_pred_call_1by_seed_w_omc == "Res" & dataBin == "Sen"))       # cells predicted resistant which are actually sensitive (wrong prediction)
            dum1 = ((TP*TN)-(FP*FN))
            dum2 = sqrt(1.0*(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # 1.0 makes the multiplication float, thus avoiding integer overflow
            if(dum2 == 0) mcc_w_omc = NaN else {mcc_w_omc = dum1/dum2}
            # 2 : define the others metrics  
            ## isolate these in a module
            precision = TP/(TP+FP); # def precision
            recall = TP/(TP+FN); # def recall
            specif = TN/(TN+FP); # def spec
            f1 = 2*precision*recall/(precision+recall); # def f1
            num_pred_call_as_cPos_w_omc = sum(ol2_pred_call_w_omc == "Sen"); # def num_pred_call_as_cPos_w_omc
            # 2 : ...and calculate them (lets update the metrics for a seed) 
            ol1_MCC_col_omc_for_1seed = c(ol1_MCC_col_omc_for_1seed,mcc_w_omc) #P2U
            ol1_PREC_col_omc_for_1seed = c(ol1_PREC_col_omc_for_1seed,precision) #P6U
            ol1_RECALL_col_omc_for_1seed = c(ol1_RECALL_col_omc_for_1seed, recall) #P7U
            ol1_SPEC_col_omc_for_1seed = c(ol1_SPEC_col_omc_for_1seed, specif) #P8U
            ol1_F1_col_omc_for_1seed = c(ol1_F1_col_omc_for_1seed,f1) #P9U
            ol1_col_num_pred_call_as_cPos_w_omc_by_1seed = c(ol1_col_num_pred_call_as_cPos_w_omc_by_1seed,num_pred_call_as_cPos_w_omc) #P10U
            # ~~~~~~(for the full dataset model)
            # 0: the predictions called for the full dataset LOO folds
            ol2_col_pred_call_1by_seed_w_allfts = ol2_col_pred_call_1by_seed_w_allfts[match(rownames(data),names(ol2_col_pred_call_1by_seed_w_allfts))] # affect samples names as they are the same number
            # 1: define mcc...
            ##again isolate in module
            TPa = length(which(ol2_col_pred_call_1by_seed_w_allfts == "Sen" & dataBin == "Sen"))    # cells predicted sensitive which are actually sensitive (cell sensitive is +ve)
            TNa = length(which(ol2_col_pred_call_1by_seed_w_allfts == "Res" & dataBin == "Res"))  # cells predicted resistant which are actually resistant (cell resistant is -ve)
            FPa = length(which(ol2_col_pred_call_1by_seed_w_allfts == "Sen" & dataBin == "Res")) # cells predicted sensitive which are actually resistant (wrong prediction)
            FNa = length(which(ol2_col_pred_call_1by_seed_w_allfts == "Res" & dataBin == "Sen"))       # cells predicted resistant which are actually sensitive (wrong prediction)
            dum1a = ((TPa*TNa)-(FPa*FNa))
            dum2a = sqrt(1.0*(TPa+FPa)*(TPa+FNa)*(TNa+FPa)*(TNa+FNa)) # 1.0 makes the multiplication float, thus avoiding integer overflow
            if(dum2a == 0) mcc_w_allfts = NaN else {mcc_w_allfts = dum1a/dum2a}
            # : 2:...and calculate it
            ol1_MCC_col_allfts_for_1seed = c(ol1_MCC_col_allfts_for_1seed,mcc_w_allfts)
            # all metrics for a seed are computed
            # >>>>>>>>>>>>>>>>>END OF METRICS FOR EACH SEED
          } # outer loop 1 (on the seeds)
          # >>>>>>>>METRICS BASED ON ALL SEEDS 
          MCC_med_omc_for_allseeds = median(ol1_MCC_col_omc_for_1seed) #P3I,P3U,P3F (for each seed there is one ol1_MCC_col_omc_for_1seed and its for the omc)
          MCC_med_allfts_for_allseeds = median(ol1_MCC_col_allfts_for_1seed) #P5I,P5U,P5F (for each seed there is one ol1_MCC_col_allfts_for_1seed and its for the all_fts model)
          PREC_med_omc_for_allseeds = median(ol1_PREC_col_omc_for_1seed) #P6F (for each seed, the next 5 metrics only exist for the OMC model. ## do it the all_fts also : no need, its the random but still ask)
          RECALL_med_omc_for_allseeds = median(ol1_RECALL_col_omc_for_1seed) #P7F
          SPEC_med_omc_for_allseeds = median(ol1_SPEC_col_omc_for_1seed) #P8F
          F1_med_omc_for_allseeds = median(ol1_F1_col_omc_for_1seed) #P9F
          num_pred_call_as_cPos_w_omc_by_allseeds = sum(ol1_col_num_pred_call_as_cPos_w_omc_by_1seed) #P10F
          #<<<<<<<<<<<<<<<<<< END OF METRICS BASED ON ALL SEEDS
          #lets report everything in 11 metrics (37 values)
          M2[i,] = c(count, il1_col_med_il2_col_omc, ol1_MCC_col_omc_for_1seed, MCC_med_omc_for_allseeds,ol1_MCC_col_allfts_for_1seed,MCC_med_allfts_for_allseeds,PREC_med_omc_for_allseeds, RECALL_med_omc_for_allseeds, SPEC_med_omc_for_allseeds, F1_med_omc_for_allseeds, num_pred_call_as_cPos_w_omc_by_allseeds)
        } else {
          M2[i,] = c(count, rep(NA,37)) #!!! in the event of class sparsity, on the line i of M2, report the models number for the the treatmadetests has been made on and NA to the metrics boxes
        } # closing condittion on class sparsity in data defined so no analysis and no results (NAs)
      }else{
        message(paste("Treatment ",treatmentid," - ",treatmentName, " has not been tested on ",cancertype," when probed profile is ",feat.type, "\n",sep = ""))
      } # noloop4 (from L378) # closing non existing file after ctype, profile and drug are searched
    } #noloop3 # closing loop on drugs
    # add the treatment id and the treatment name as 1st and 2nd columns of M2 like in a concatenation way
    M2 = data.frame(treatmentid=treatments$Treatment_ID, drugName=treatments$Treatment_Details, M2)
    # name the columns of M2 dataframe
    colnames(M2) = c("Treatment_ID", "Treatment_Name","Number of models",paste("Median nfeats rep",seq(1,10,1)),paste("MCC Rep",seq(1,10,1)), 
                     "MCC.5cv.med", paste("MCC (allfts) Rep",seq(1,10,1)),"MCC.5cv.med.allfts", "PREC.5cv.med", "RECALL.5cv.med", "SPEC.5cv.med", "F1.5cv.med", "nResp.5cv")
    
    write.csv(M2, file = paste("../results_selfeats/single.cls.",output.type,".", ntree, "trees.", mth, "_selbest",feat.type,"_",sel.meth,".",cancertype,".LOOCVxLOOCV.10x.meanMCC.csv", sep=""), row.names = F)
  } #noloop2 # closing loop on profiles
  end.time.cancertypeloop <- Sys.time()
  time.taken.cancertypeloop <- end.time.cancertypeloop - start.time.cancertypeloop
  cat(paste("Analysis of ",cancertype," with all profiles has been done in a : ", "\n",sep = ""))
  time.taken.cancertypeloop
} #noloop1 # closing loop on ctypes
end.time <- Sys.time()
time.taken <- end.time - start.time
cat(paste("Full analysis has been done in a : ", "\n",sep = ""))
time.taken
#last_stop==>put in python


