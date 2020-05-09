##! add the intro part to keep if working

### step : launch R or Rstudio in sudo

##### Librairy installation issues noted 
#---(not to use because devtools is hard to install on out of the box machine)
# # Workflow package installation from Github 
# install.packages("devtools")
# library(devtools) 
# 
# devtools::install_github("r-lib/remotes") 
# library(remotes) 
# packageVersion("remotes") # has to be 1.1.1.9000 or later 
# remotes::install_github("b-klaus/maEndToEnd", ref="master")

# #---(not to use because developpement version of bioconductor manager is not going together with some versions of R
# # step : Workflow package installation from Bioconductor
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# # The following initializes usage of Bioc devel
# BiocManager::install(version='devel')
# BiocManager::install("maEndToEnd")
# #### same as this
# if (!require("BiocManager"))
#   install.packages("BiocManager")
# BiocManager::install("maEndToEnd", version = "devel")
# # This give this erreor : Error: Bioconductor version '3.11' requires R version '4.0'; see https://bioconductor.org/install
# # ie the bioconductor version you are setting to use is not the one in accordance with your present R version
# # so we have to get the proper bioconductor version 

  # step : install the proper version of bioinconductor for the proper R version  that you have (see https://www.bioconductor.org/install/)
          if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
          BiocManager::install(version = "3.10")
# step : install the workflow package         
  BiocManager::install("maEndToEnd")
# issue 1 : some dependencies are not available so we go get them 
# use this in the terminal of linux to install dependencies (sudo apt-get install -y libssl-dev; see all installed through the result of the cmd history of terminal)
# - see the history of all instals  

# step : call the wf lib
suppressPackageStartupMessages({library("maEndToEnd")})

    # 3 : Downloading the raw data from ArrayExpress
    raw_data_dir <- tempdir()
      
    if (!dir.exists(raw_data_dir)) {
      dir.create(raw_data_dir)
    }
  
anno_AE <- getAE("E-MTAB-2967", path = raw_data_dir, type = "raw") 

##! last stop (previous worked : libs are in and last order is in)

# 5 Import of annotation data and microarray expression data as “ExpressionSet”

sdrf_location <- file.path(raw_data_dir, "E-MTAB-2967.sdrf.txt")
SDRF <- read.delim(sdrf_location)

rownames(SDRF) <- SDRF$Array.Data.File
SDRF <- AnnotatedDataFrame(SDRF)
# 
raw_data <- oligo::read.celfiles(filenames = file.path(raw_data_dir, 
                                                       SDRF$Array.Data.File),
                                 verbose = FALSE, phenoData = SDRF)
stopifnot(validObject(raw_data))
# 
head(Biobase::pData(raw_data))
# 
Biobase::pData(raw_data) <- Biobase::pData(raw_data)[, c("Source.Name",
"Characteristics.individual.",
"Factor.Value.disease.",
"Factor.Value.phenotype.")]

# 6 Quality control of the raw data
# 
Biobase::exprs(raw_data)[1:5, 1:5]
#   
exp_raw <- log2(Biobase::exprs(raw_data))
PCA_raw <- prcomp(t(exp_raw), scale. = FALSE)

percentVar <- round(100*PCA_raw$sdev^2/sum(PCA_raw$sdev^2),1)
sd_ratio <- sqrt(percentVar[2] / percentVar[1])

dataGG <- data.frame(PC1 = PCA_raw$x[,1], PC2 = PCA_raw$x[,2],
                     Disease = pData(raw_data)$Factor.Value.disease.,
                     Phenotype = pData(raw_data)$Factor.Value.phenotype.,
                     Individual = pData(raw_data)$Characteristics.individual.)

ggplot(dataGG, aes(PC1, PC2)) +
  geom_point(aes(shape = Disease, colour = Phenotype)) +
  ggtitle("PCA plot of the log-transformed raw expression data") +
  xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
  ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
  theme(plot.title = element_text(hjust = 0.5))+
  coord_fixed(ratio = sd_ratio) +
  scale_shape_manual(values = c(4,15)) + 
  scale_color_manual(values = c("darkorange2", "dodgerblue4"))

# 
oligo::boxplot(raw_data, target = "core", 
               main = "Boxplot of log2-intensitites for the raw data")
# 
##!    not important but check it later if really worked ## not done in routine tests
arrayQualityMetrics(expressionset = raw_data,
                    outdir = tempdir(),
                    force = TRUE, do.logtransform = TRUE,
                    intgroup = c("Factor.Value.disease.", "Factor.Value.phenotype.")) 
# 7 Background adjustment, calibration, summarization and annotation
#   Background adjustment

# Across-array normalization (calibration)

# Summarization
  head(ls("package:hugene10sttranscriptcluster.db"))
# Old and new “probesets” of Affymetrix microarrays 

# One-step preprocessing in oligo
  
# 8 Relative Log Expression data quality analysis
    palmieri_eset <- oligo::rma(raw_data, target = "core", normalize = FALSE)
# Computing the RLE

# Plotting the RLE
    ##! check this boxplot out (why isn't it showing, maybe because multiple dowload have been made of the data )
      row_medians_assayData <- 
        Biobase::rowMedians(as.matrix(Biobase::exprs(palmieri_eset)))
      
      RLE_data <- sweep(Biobase::exprs(palmieri_eset), 1, row_medians_assayData)
      
      RLE_data <- as.data.frame(RLE_data)
      RLE_data_gathered <- 
        tidyr::gather(RLE_data, patient_array, log2_expression_deviation)
      
      ggplot2::ggplot(RLE_data_gathered, aes(patient_array,
                                             log2_expression_deviation)) + 
        geom_boxplot(outlier.shape = NA) + 
        ylim(c(-2, 2)) + 
        theme(axis.text.x = element_text(colour = "aquamarine4", 
                                         angle = 60, size = 6.5, hjust = 1 ,
                                         face = "bold"))
# 9 RMA calibration of the data
          palmieri_eset_norm <- oligo::rma(raw_data, target = "core")
  # Some mathematical background on normalization (calibration) and background correction

# Quality assessment of the calibrated data
#
# PCA analysis
        exp_palmieri <- Biobase::exprs(palmieri_eset_norm)
        PCA <- prcomp(t(exp_palmieri), scale = FALSE)
        
        percentVar <- round(100*PCA$sdev^2/sum(PCA$sdev^2),1)
        sd_ratio <- sqrt(percentVar[2] / percentVar[1])
        
        dataGG <- data.frame(PC1 = PCA$x[,1], PC2 = PCA$x[,2],
                             Disease = 
                               Biobase::pData(palmieri_eset_norm)$Factor.Value.disease.,
                             Phenotype = 
                               Biobase::pData(palmieri_eset_norm)$Factor.Value.phenotype.)
        
        
        ggplot(dataGG, aes(PC1, PC2)) +
          geom_point(aes(shape = Disease, colour = Phenotype)) +
          ggtitle("PCA plot of the calibrated, summarized data") +
          xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
          ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
          theme(plot.title = element_text(hjust = 0.5)) +
          coord_fixed(ratio = sd_ratio) +
          scale_shape_manual(values = c(4,15)) + 
          scale_color_manual(values = c("darkorange2", "dodgerblue4"))
# Heatmap clustering analysis 
# 
        phenotype_names <- ifelse(str_detect(pData
                                             (palmieri_eset_norm)$Factor.Value.phenotype.,
                                             "non"), "non_infl.", "infl.")
        
        disease_names <- ifelse(str_detect(pData
                                           (palmieri_eset_norm)$Factor.Value.disease.,
                                           "Crohn"), "CD", "UC")
        
        annotation_for_heatmap <- 
          data.frame(Phenotype = phenotype_names,  Disease = disease_names)
        
        row.names(annotation_for_heatmap) <- row.names(pData(palmieri_eset_norm))
#
        dists <- as.matrix(dist(t(exp_palmieri), method = "manhattan"))
        
        rownames(dists) <- row.names(pData(palmieri_eset_norm))
        hmcol <- rev(colorRampPalette(RColorBrewer::brewer.pal(9, "YlOrRd"))(255))
        colnames(dists) <- NULL
        diag(dists) <- NA
        
        ann_colors <- list(
          Phenotype = c(non_infl. = "chartreuse4", infl. = "burlywood3"),
          Disease = c(CD = "blue4", UC = "cadetblue2")
        )
        pheatmap(dists, col = (hmcol), 
                 annotation_row = annotation_for_heatmap,
                 annotation_colors = ann_colors,
                 legend = TRUE, 
                 treeheight_row = 0,
                 legend_breaks = c(min(dists, na.rm = TRUE), 
                                   max(dists, na.rm = TRUE)), 
                 legend_labels = (c("small distance", "large distance")),
                 main = "Clustering heatmap for the calibrated samples")

# 10 Filtering based on intensity
        palmieri_medians <- rowMedians(Biobase::exprs(palmieri_eset_norm))
        
        hist_res <- hist(palmieri_medians, 100, col = "cornsilk1", freq = FALSE, 
                         main = "Histogram of the median intensities", 
                         border = "antiquewhite4",
                         xlab = "Median intensities")
# nb : the threshold is just after the last decadence in bars height before the peak of the hist
        man_threshold <- 4
        
        hist_res <- hist(palmieri_medians, 100, col = "cornsilk", freq = FALSE, 
                         main = "Histogram of the median intensities",
                         border = "antiquewhite4",
                         xlab = "Median intensities")
        
        abline(v = man_threshold, col = "coral4", lwd = 2)
        
# 
        no_of_samples <- 
          table(paste0(pData(palmieri_eset_norm)$Factor.Value.disease., "_", 
                       pData(palmieri_eset_norm)$Factor.Value.phenotype.))
        no_of_samples
# 
          samples_cutoff <- min(no_of_samples)
          
          idx_man_threshold <- apply(Biobase::exprs(palmieri_eset_norm), 1,
                                     function(x){
                                       sum(x > man_threshold) >= samples_cutoff})
          table(idx_man_threshold)
# 
          palmieri_manfiltered <- subset(palmieri_eset_norm, idx_man_threshold)   
          
# 11 Annotation of the transcript clusters
          anno_palmieri <- AnnotationDbi::select(hugene10sttranscriptcluster.db,
                                                 keys = (featureNames(palmieri_manfiltered)),
                                                 columns = c("SYMBOL", "GENENAME"),
                                                 keytype = "PROBEID")
          
          anno_palmieri <- subset(anno_palmieri, !is.na(SYMBOL)) ##! check if we really want to filtered the unindentified 
# Removing multiple mappings
            anno_grouped <- group_by(anno_palmieri, PROBEID)
            anno_summarized <- 
              dplyr::summarize(anno_grouped, no_of_matches = n_distinct(SYMBOL))
            
            head(anno_summarized)
#   
            anno_filtered <- filter(anno_summarized, no_of_matches > 1)
          
          head(anno_filtered)
# 
          probe_stats <- anno_filtered 
          
          nrow(probe_stats)
# 
          dim(probe_stats)
#   
            ids_to_exlude <- (featureNames(palmieri_manfiltered) %in% probe_stats$PROBEID)
            
            table(ids_to_exlude)
# 
            palmieri_final <- subset(palmieri_manfiltered, !ids_to_exlude)
            
            validObject(palmieri_final)
#   
                head(anno_palmieri)
#
                fData(palmieri_final)$PROBEID <- rownames(fData(palmieri_final))
# 
                fData(palmieri_final) <- left_join(fData(palmieri_final), anno_palmieri)
#   
                # restore rownames after left_join
                rownames(fData(palmieri_final)) <- fData(palmieri_final)$PROBEID 
                
                validObject(palmieri_final)
# Building custom annotations (##! to read again to include our own custom annots)  
#     
# 12 Linear models        
#  Linear models for microarrays
# A linear model for the data
                individual <- 
                  as.character(Biobase::pData(palmieri_final)$Characteristics.individual.)
                
                tissue <- str_replace_all(Biobase::pData(palmieri_final)$Factor.Value.phenotype.,
                                          " ", "_")
                
                tissue <- ifelse(tissue == "non-inflamed_colonic_mucosa",
                                 "nI", "I")
                
                disease <- 
                  str_replace_all(Biobase::pData(palmieri_final)$Factor.Value.disease.,
                                  " ", "_")
                disease <- 
                  ifelse(str_detect(Biobase::pData(palmieri_final)$Factor.Value.disease., 
                                    "Crohn"), "CD", "UC")   
#
                i_CD <- individual[disease == "CD"]
                design_palmieri_CD <- model.matrix(~ 0 + tissue[disease == "CD"] + i_CD)
                colnames(design_palmieri_CD)[1:2] <- c("I", "nI")
                rownames(design_palmieri_CD) <- i_CD 
                
                i_UC <- individual[disease == "UC"]
                design_palmieri_UC <- model.matrix(~ 0 + tissue[disease == "UC"] + i_UC )
                colnames(design_palmieri_UC)[1:2] <- c("I", "nI")
                rownames(design_palmieri_UC) <- i_UC
#   
                head(design_palmieri_CD[, 1:6])
                head(design_palmieri_UC[, 1:6])
# Analysis of differential expression based on a single gene
# Illustration of the fitted linear model on the CRAT gene
                tissue_CD <- tissue[disease == "CD"]
                crat_expr <- Biobase::exprs(palmieri_final)["8164535", disease == "CD"]
                crat_data <- as.data.frame(crat_expr)
                colnames(crat_data)[1] <- "org_value"
                crat_data <- mutate(crat_data, individual = i_CD, tissue_CD)
                
                crat_data$tissue_CD <- factor(crat_data$tissue_CD, levels = c("nI", "I"))
                
                ggplot(data = crat_data, aes(x = tissue_CD, y = org_value, 
                                             group = individual, color = individual)) +
                  geom_line() +
                  ggtitle("Expression changes for the CRAT gene")
#   
                crat_coef <- lmFit(palmieri_final[,disease == "CD"],
                                   design = design_palmieri_CD)$coefficients["8164535",]
                
                crat_coef
#   
                crat_fitted <- design_palmieri_CD %*% crat_coef
                rownames(crat_fitted) <- names(crat_expr)
                colnames(crat_fitted) <- "fitted_value"
                
                crat_fitted
# 
                crat_data$fitted_value <- crat_fitted
                
                ggplot(data = crat_data, aes(x = tissue_CD, y = fitted_value, 
                                             group = individual, color = individual)) +
                  geom_line() +
                  ggtitle("Fitted expression changes for the CRAT gene")  
# Differential expression analysis of the CRAT gene
                crat_noninflamed <- na.exclude(crat_data$org_value[tissue == "nI"])
                crat_inflamed <- na.exclude(crat_data$org_value[tissue == "I"])
                res_t <- t.test(crat_noninflamed ,crat_inflamed , paired = TRUE)
                res_t
# Contrasts and hypotheses tests
                contrast_matrix_CD <- makeContrasts(I-nI, levels = design_palmieri_CD)
                
                palmieri_fit_CD <- eBayes(contrasts.fit(lmFit(palmieri_final[,disease == "CD"],
                                                              design = design_palmieri_CD),
                                                        contrast_matrix_CD))
                
                contrast_matrix_UC <- makeContrasts(I-nI, levels = design_palmieri_UC)
                
                palmieri_fit_UC <- eBayes(contrasts.fit(lmFit(palmieri_final[,disease == "UC"],
                                                              design = design_palmieri_UC),
                                                        contrast_matrix_UC))
# Extracting results
                table_CD <- topTable(palmieri_fit_CD, number = Inf)
                head(table_CD)
# 
                hist(table_CD$P.Value, col = brewer.pal(3, name = "Set2")[1],
                     main = "inflamed vs non-inflamed - Crohn's disease", xlab = "p-values")
#   
                table_UC <- topTable(palmieri_fit_UC, number = Inf)
                head(table_UC)
# 
                hist(table_UC$P.Value, col = brewer.pal(3, name = "Set2")[2],
                     main = "inflamed vs non-inflamed - Ulcerative colitis", xlab = "p-values")
# Multiple testing FDR, and comparison with results from the original paper
                nrow(subset(table_UC, P.Value < 0.001))
#   
                tail(subset(table_UC, P.Value < 0.001))
#     
                fpath <- system.file("extdata", "palmieri_DE_res.xlsx", package = "maEndToEnd")
                palmieri_DE_res <- sapply(1:4, function(i) read.xlsx(cols = 1, fpath, 
                                                                     sheet = i, startRow = 4))
                
                names(palmieri_DE_res) <- c("CD_UP", "CD_DOWN", "UC_UP", "UC_DOWN")
                palmieri_DE_res <- lapply(palmieri_DE_res, as.character)
                paper_DE_genes_CD <- Reduce("c", palmieri_DE_res[1:2])
                paper_DE_genes_UC <- Reduce("c", palmieri_DE_res[3:4])
                
                overlap_CD <- length(intersect(subset(table_CD, P.Value < 0.001)$SYMBOL,  
                                               paper_DE_genes_CD)) / length(paper_DE_genes_CD)
                
                
                overlap_UC <- length(intersect(subset(table_UC, P.Value < 0.001)$SYMBOL,
                                               paper_DE_genes_UC)) / length(paper_DE_genes_UC)
                overlap_CD
# 
                overlap_UC 
#   
                total_genenumber_CD <- length(subset(table_CD, P.Value < 0.001)$SYMBOL)
                total_genenumber_UC <- length(subset(table_UC, P.Value < 0.001)$SYMBOL)
                
                total_genenumber_CD
#   
                total_genenumber_UC
# Visualization of DE analysis results - volcano plot
                volcano_names <- ifelse(abs(palmieri_fit_CD$coefficients)>=1, 
                                        palmieri_fit_CD$genes$SYMBOL, NA)
                
                
                volcanoplot(palmieri_fit_CD, coef = 1L, style = "p-value", highlight = 100, 
                            names = volcano_names,
                            xlab = "Log2 Fold Change", ylab = NULL, pch=16, cex=0.35)
# 13 Gene ontology (GO) based enrichment analysis
                DE_genes_CD <- subset(table_CD, adj.P.Val < 0.1)$PROBEID
# Matching the background set of genes
                back_genes_idx <- genefilter::genefinder(palmieri_final, 
                                                         as.character(DE_genes_CD), 
                                                         method = "manhattan", scale = "none")
#         
                back_genes_idx <- sapply(back_genes_idx, function(x)x$indices)
#   
                back_genes <- featureNames(palmieri_final)[back_genes_idx]
                back_genes <- setdiff(back_genes, DE_genes_CD)
                
                
                intersect(back_genes, DE_genes_CD)
#   
                length(back_genes)
# 
                multidensity(list(
                  all = table_CD[,"AveExpr"] ,
                  fore = table_CD[DE_genes_CD , "AveExpr"],
                  back = table_CD[rownames(table_CD) %in% back_genes, "AveExpr"]),
                  col = c("#e46981", "#ae7ee2", "#a7ad4a"),
                  xlab = "mean expression",
                  main = "DE genes for CD-background-matching")
# Running topGO
                gene_IDs <- rownames(table_CD)
                in_universe <- gene_IDs %in% c(DE_genes_CD, back_genes)
                in_selection <- gene_IDs %in% DE_genes_CD 
                
                all_genes <- in_selection[in_universe]
                all_genes <- factor(as.integer(in_selection[in_universe]))
                names(all_genes) <- gene_IDs[in_universe] 
 #  
                top_GO_data <- new("topGOdata", ontology = "BP", allGenes = all_genes,
                                   nodeSize = 10, annot = annFUN.db, affyLib = "hugene10sttranscriptcluster.db")
 # 
                result_top_GO_elim <- 
                  runTest(top_GO_data, algorithm = "elim", statistic = "Fisher")
                result_top_GO_classic <- 
                  runTest(top_GO_data, algorithm = "classic", statistic = "Fisher")
# 
                res_top_GO <- GenTable(top_GO_data, Fisher.elim = result_top_GO_elim,
                                       Fisher.classic = result_top_GO_classic,
                                       orderBy = "Fisher.elim" , topNodes = 100)
                
                genes_top_GO <- printGenes(top_GO_data, whichTerms = res_top_GO$GO.ID,
                                           chip = "hugene10sttranscriptcluster.db", geneCutOff = 1000)
                
                res_top_GO$sig_genes <- sapply(genes_top_GO, function(x){
                  str_c(paste0(x[x$'raw p-value' == 2, "Symbol.id"],";"), 
                        collapse = "")
                })
                
                head(res_top_GO[,1:8], 20)
# Visualization of the GO-analysis results (part commented out)
                # showSigOfNodes(top_GO_data, score(result_top_GO_elim), firstSigNodes = 3,
                #                useInfo = 'def')
  
# A pathway enrichment analysis using reactome
                entrez_ids <- mapIds(hugene10sttranscriptcluster.db, 
                                     keys = rownames(table_CD), 
                                     keytype = "PROBEID",
                                     column = "ENTREZID")
#   
                reactome_enrich <- enrichPathway(gene = entrez_ids[DE_genes_CD], 
                                                 universe = entrez_ids[c(DE_genes_CD, 
                                                                         back_genes)],
                                                 organism = "human",
                                                 pvalueCutoff = 0.05,
                                                 qvalueCutoff = 0.9, 
                                                 readable = TRUE)
                
                reactome_enrich@result$Description <- paste0(str_sub(
                  reactome_enrich@result$Description, 1, 20),
                  "...")
                
                head(as.data.frame(reactome_enrich))[1:6]
# Visualizing the reactome based analysis results
                barplot(reactome_enrich)
#   
                emapplot(reactome_enrich, showCategory = 10)
#  15 Session information
                  gc()
# 
                  length(getLoadedDLLs())
# 
                  sessionInfo()
# EOF 
                      
                    
                  
                
                  
                  
                  
                    
                  
                  
                  
                  
              
                  
                
                
                
                
                
                
                    
                
                
                
                
                  
                
                  
                  
                
                  
                
                
               
          
          
          
          
          
          
          
          
          
          
          
          
        
        
          
        