# new script from template below source for heatmaps and dendrograms 
# source : https://jcoliver.github.io/learn-r/008-ggplot-dendrograms-and-heatmaps.html

#----Getting started
rm(list = ls()) # clean up the environment

# creating two folders we’ll use to organize our efforts
dir.create("data")
dir.create("output")

# Load dependencies
library("ggplot2")
library("ggdendro")
library("reshape2")
library("grid")

# download data file .csv and put it in data folder (manually done)

#------Data preparation
# Read in data
otter <- read.csv(file = "data/otter-mandible-data.csv", header = TRUE)
# data preview 
# head(otter)
# data restriction 
two.species <- c("A. cinerea", "L. canadensis")
otter <- otter[otter$species %in% two.species, ]
# scale the data variables (columns 4-9) for measures with mean  0 and a variance 1
otter.scaled <- otter
otter.scaled[, c(4:9)] <- scale(otter.scaled[, 4:9])

#-----Clustering
# To make our figure, we will build the two plots (the cluster diagram and the heatmap) separately
# then use the grid framework to put them together. 
#- We start by making the dendrogram (or cluster).
# Run clustering
otter.matrix <- as.matrix(otter.scaled[, -c(1:3)])
rownames(otter.matrix) <- otter.scaled$accession
otter.dendro <- as.dendrogram(hclust(d = dist(x = otter.matrix)))
# Create dendro + make the tips labels (the samples names) a bit smaller:
dendro.plot <- ggdendrogram(data = otter.dendro, rotate = TRUE) + theme(axis.text.y = element_text(size = 6))
# Preview the plot
# print(dendro.plot)

#---Heatmap
# # Data wrangling : - transforming data to a “long” format 
# (a single measure in each row by storing the variable name in a colusing melt from package reshape2)
otter.long <- melt(otter.scaled, id = c("species", "museum", "accession"))
# Extract the order of the tips in the dendrogram to use it force the hm tips order
#solving issue 1-tips of the dendrogram are not in the same order as the y-axis of the heatmap. 
# Re-order the heatmap to match the structure of the clustering.
# use the order of the tips in the dendrogram to re-order the rows in our heatmap. 
# The order of those tips are stored in the otter.dendro object, 
# and we can extract it with the order.dendro function
otter.order <- order.dendrogram(otter.dendro)
# Order the levels according to their position in the cluster
otter.long$accession <- factor(x = otter.long$accession,
                               levels = otter.scaled$accession[otter.order], 
                               ordered = TRUE)
# By default, ggplot use the level order of the y-axis labels as the means of ordering the rows in the heatmap
# Using the order of the dendrogram tips (we stored this above in the otter.order vector), 
# we can re-level the otter.long$accession column to match the order of the tips in the hm
# Order the levels according to their position in the cluster


# - Create heatmap plot
# create the heatmap at the end, because the underlying data (otter.long) has changed
# use that new data frame, otter.long, to create our heatmap (ggplot package, geom_tile layer for a heatmap)
# include the samples names size reduction like prevously with theme
heatmap.plot <- ggplot(data = otter.long, aes(x = variable, y = accession)) +
  geom_tile(aes(fill = value)) +
  scale_fill_gradient2() +
  theme(axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "top")
# + theme(axis.text.y = element_text(size = 6)) is absent because it ot needed as both plots are aligned
#solving issue 2-have the dendrogram and heatmap closer together, without a legend separating them
# move the legend to the top of the heatmap. 
# done using the theme layer when creating the heatmap, setting the value of legend.position to “top”
#solving issue 3-dendrogram should be vertically stretched so each tip lines up with a row in the heatmap plot
# Align dendrogram tips with heatmap rows : 
# through trial and error, we change 2 parameters on the viewport of the dendro when printing it
# 1-: y; vertical justification of the dendro. closer ro 0 is lower and closer to 1 is higher
# use 0.5
# 2-: height : proportion of the viewport used by the dendro. initially 1.0, as the legend 
# occupy bit of top, rduce it a bit to 0.92
# these values change from device to device used, but try y=0.4 and height=0.85
#soving 4 as bonus : remove redundant tips of the hm because aligned already wth tips of dendro
# use in theme the element_blank() when calling ggplot in the heatmap

# ------put all together
grid.newpage()
print(heatmap.plot, vp = viewport(x = 0.4, y = 0.5, width = 0.8, height = 1.0))
print(dendro.plot, vp = viewport(x = 0.90, y = 0.43, width = 0.2, height = 0.92))


#----------------------------------------------------



row_clust <- hclust(dist(otter.matrix, method = 'euclidean'), method = 'ward.D2')

plot(row_clust)

sort(cutree(row_clust, k=2))














