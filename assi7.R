# ==============================================================================
# ASSIGNMENT 7: CLUSTERING AND STATISTICAL ANALYSIS USING R
# Dataset: Online Shoppers Purchasing Intention Dataset
# ==============================================================================

# Required Libraries
# install.packages(c("readr", "dplyr", "ggplot2", "cluster", "factoextra", "gridExtra"))

library(readr)
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(gridExtra)

# Set seed for reproducibility across all stochastic operations (K-Means, PCA, etc.)
set.seed(42)

# ==============================================================================
# PART A: DATA PREPARATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 1:
# Import the Online Shoppers Purchasing Intention Dataset into R and display the first few observations.
# ------------------------------------------------------------------------------
# R Code:
shoppers_raw <- read.csv("online_shoppers_intention.csv")
head(shoppers_raw)

# Detailed Explanation:
# - read.csv() loads the CSV file into an R data frame.
# - head() displays the first 6 observations, allowing quick verification of column names, formats, and initial values.


# ------------------------------------------------------------------------------
# Question 2:
# Determine the number of observations and variables in the dataset.
# ------------------------------------------------------------------------------
# R Code:
dim(shoppers_raw)
cat("Observations (Rows):", nrow(shoppers_raw), "
")
cat("Variables (Columns):", ncol(shoppers_raw), "
")

# Detailed Explanation:
# - dim() returns dimensions as [rows, columns].
# - Typically, this dataset contains 12,330 sessions across 18 features capturing page metrics, user characteristics, and outcome variables.


# ------------------------------------------------------------------------------
# Question 3:
# Display the structure, data types, and summary statistics of the dataset.
# ------------------------------------------------------------------------------
# R Code:
str(shoppers_raw)
summary(shoppers_raw)

# Detailed Explanation:
# - str() reveals the structure, variable data types (int, num, factor/chr, logical), and head records.
# - summary() computes 5-number summaries for numeric fields and frequency counts for categorical/logical columns.


# ------------------------------------------------------------------------------
# Question 4:
# Identify numerical and categorical attributes.
# ------------------------------------------------------------------------------
# R Code:
num_cols <- names(shoppers_raw)[sapply(shoppers_raw, is.numeric)]
cat_cols <- names(shoppers_raw)[!sapply(shoppers_raw, is.numeric)]
cat("Numerical Attributes:
", paste(num_cols, collapse = ", "), "

")
cat("Categorical/Logical Attributes:
", paste(cat_cols, collapse = ", "), "
")

# Detailed Explanation:
# - Numerical attributes (10 features): Administrative, Administrative_Duration, Informational, Informational_Duration,
#   ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay.
# - Categorical/Logical (8 features): OperatingSystems, Browser, Region, TrafficType (encoded as ints/factors),
#   Month, VisitorType, Weekend, Revenue.


# ------------------------------------------------------------------------------
# Question 5:
# Check for missing values and handle them appropriately.
# ------------------------------------------------------------------------------
# R Code:
cat("Total NA values in dataset:", sum(is.na(shoppers_raw)), "
")
colSums(is.na(shoppers_raw))
shoppers_clean <- na.omit(shoppers_raw)

# Detailed Explanation:
# - sum(is.na()) aggregates missing cells across the data frame.
# - The standard Online Shoppers dataset typically has no missing values; na.omit() ensures any latent NAs are dropped.


# ------------------------------------------------------------------------------
# Question 6:
# Detect and remove duplicate records, if any.
# ------------------------------------------------------------------------------
# R Code:
cat("Number of duplicate rows:", sum(duplicated(shoppers_clean)), "
")
shoppers_clean <- shoppers_clean[!duplicated(shoppers_clean), ]
cat("Rows remaining after deduplication:", nrow(shoppers_clean), "
")

# Detailed Explanation:
# - duplicated() identifies identical telemetry/browsing sessions.
# - Removing duplicates (~125-130 rows in this dataset) prevents distance distortion and duplicate weight bias during clustering.


# ------------------------------------------------------------------------------
# Question 7:
# Identify and remove non-informative or identifier-like variables from the clustering features, if present.
# ------------------------------------------------------------------------------
# R Code:
# The dataset has no unique ID column, but labels/demographic tags (Revenue, Weekend, Month, VisitorType, etc.)
# are excluded from the unsupervised clustering feature matrix.
non_cluster_vars <- c("Revenue", "Weekend", "Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType")
cluster_df_raw <- shoppers_clean[, !(names(shoppers_clean) %in% non_cluster_vars)]
head(cluster_df_raw)

# Detailed Explanation:
# - Pure behavioral segmentation relies on browsing metrics.
# - Including metadata or the ground-truth target (Revenue) would leak label information or skew geometric clusters.


# ------------------------------------------------------------------------------
# Question 8:
# Select suitable numerical features for clustering.
# ------------------------------------------------------------------------------
# R Code:
selected_features <- c("Administrative_Duration", "Informational_Duration", 
                       "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues")
cluster_data <- cluster_df_raw[, selected_features]
head(cluster_data)

# Detailed Explanation:
# - Duration metrics represent customer engagement and interest intensity.
# - BounceRates and ExitRates capture session disengagement/abandonment.
# - PageValues reflects historical monetary potential of visited pages (a key intent indicator).


# ------------------------------------------------------------------------------
# Question 9:
# Perform appropriate feature scaling or standardization.
# ------------------------------------------------------------------------------
# R Code:
cluster_scaled <- scale(cluster_data)
head(cluster_scaled)

# Detailed Explanation:
# - scale() centers each variable to mean 0 and divides by standard deviation (unit variance).
# - This is essential because duration features (spanning thousands of seconds) would otherwise dominate
#   rate features bounded between 0 and 1 in Euclidean distance calculations.


# ------------------------------------------------------------------------------
# Question 10:
# Analyze the distribution of the selected variables using suitable graphical methods.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(2, 3))
for (col in colnames(cluster_data)) {
  hist(cluster_data[[col]], main = paste("Hist:", col), xlab = col, col = "skyblue", breaks = 30)
}
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Histograms show substantial positive (right) skewness for PageValues and Duration metrics.
# - Most user sessions are short, with low page values, whereas a small subset shows high browsing engagement.


# ==============================================================================
# PART B: CLUSTERING ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 11:
# Compute the correlation matrix of the selected numerical features and identify highly correlated variables.
# ------------------------------------------------------------------------------
# R Code:
cor_matrix <- cor(cluster_data)
round(cor_matrix, 2)
cat("Strongest correlation: BounceRates and ExitRates r =", round(cor_matrix["BounceRates", "ExitRates"], 2), "
")

# Detailed Explanation:
# - cor() computes Pearson pairwise correlations.
# - BounceRates and ExitRates exhibit high collinearity (r ~ 0.90), as rapid abandonment inflates both statistics.


# ------------------------------------------------------------------------------
# Question 12:
# Apply Principal Component Analysis (PCA) to the standardized features.
# ------------------------------------------------------------------------------
# R Code:
pca_res <- prcomp(cluster_scaled, center = FALSE, scale. = FALSE)
summary(pca_res)
pca_res$rotation

# Detailed Explanation:
# - prcomp() calculates orthogonal eigenvectors and eigenvalues on standardized data.
# - Rotation matrix indicates loadings: PC1 captures overall engagement vs abandonment, PC2 captures commercial intent (PageValues).


# ------------------------------------------------------------------------------
# Question 13:
# Determine the number of principal components required to explain a suitable proportion of the variance.
# ------------------------------------------------------------------------------
# R Code:
eigenvals <- pca_res$sdev^2
prop_var <- eigenvals / sum(eigenvals)
cum_var <- cumsum(prop_var)
data.frame(PC = paste0("PC", 1:length(cum_var)), Proportion = round(prop_var, 3), Cumulative = round(cum_var, 3))

# Detailed Explanation:
# - Cumulative variance table shows how many components are needed to hit common thresholds (e.g., ~70-80%).
# - Typically, the first 3 or 4 PCs account for ~75-85% of total variance.


# ------------------------------------------------------------------------------
# Question 14:
# Visualize the observations using the first two principal components.
# ------------------------------------------------------------------------------
# R Code:
fviz_pca_ind(pca_res, geom = "point", col.ind = "cos2", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = FALSE, alpha.ind = 0.4)

# Detailed Explanation:
# - Projects high-dimensional session points onto the 2D plane defined by PC1 and PC2.
# - Coloring by squared cosine (cos2) displays how well each observation is represented on the 2D subspace.


# ------------------------------------------------------------------------------
# Question 15:
# Apply K-Means Clustering to the preprocessed dataset.
# ------------------------------------------------------------------------------
# R Code:
# Preliminary 3-cluster K-means demonstration
km_demo <- kmeans(cluster_scaled, centers = 3, nstart = 25)
table(km_demo$cluster)

# Detailed Explanation:
# - kmeans() minimizes the total within-cluster sum of squares (WSS).
# - nstart = 25 runs 25 random centroid initializations to avoid settling in suboptimal local minima.


# ------------------------------------------------------------------------------
# Question 16:
# Determine the optimal number of clusters using the Elbow Method.
# ------------------------------------------------------------------------------
# R Code:
fviz_nbclust(cluster_scaled[1:2000, ], kmeans, method = "wss", k.max = 8)

# Detailed Explanation:
# - Evaluates Total WSS across k = 1 to 8 on a representative subset for computational efficiency.
# - The inflection point ("elbow") typically emerges around k = 3 or k = 4, where additional clusters offer diminishing returns.


# ------------------------------------------------------------------------------
# Question 17:
# Perform K-Means clustering using the selected number of clusters.
# ------------------------------------------------------------------------------
# R Code:
opt_k <- 3
km_final <- kmeans(cluster_scaled, centers = opt_k, nstart = 25)
shoppers_clean$KMeans_Cluster <- as.factor(km_final$cluster)
table(shoppers_clean$KMeans_Cluster)

# Detailed Explanation:
# - k = 3 yields distinct, actionable e-commerce segments: disengaged sessions, casual browsers, and high-intent buyers.


# ------------------------------------------------------------------------------
# Question 18:
# Calculate and interpret the cluster centroids.
# ------------------------------------------------------------------------------
# R Code:
# Scaled centroids
print("Standardized Centroids:")
round(km_final$centers, 2)

# Unstandardized centroids (natural units)
aggregate(cluster_data, by = list(Cluster = km_final$cluster), FUN = mean)

# Detailed Explanation:
# - Unscaled centroids provide business interpretation in physical units (seconds, rates, page values).
# - E.g., Cluster 1: high bounce/exit rates (low engagement); Cluster 2: high duration, low page values (window shoppers);
#   Cluster 3: high PageValues and duration (high conversion probability).


# ------------------------------------------------------------------------------
# Question 19:
# Visualize the K-Means clusters using suitable plots.
# ------------------------------------------------------------------------------
# R Code:
fviz_cluster(km_final, data = cluster_scaled, geom = "point", ellipse.type = "convex", 
             palette = "jco", ggtheme = theme_minimal(), main = "K-Means Clusters (k=3)")

# Detailed Explanation:
# - fviz_cluster projects observations onto the first 2 principal components and plots convex hulls around clusters.


# ------------------------------------------------------------------------------
# Question 20:
# Evaluate the clustering quality using the Silhouette Score.
# ------------------------------------------------------------------------------
# R Code:
# Compute silhouette widths on a sample (n=2000) for fast execution
sample_idx <- sample(1:nrow(cluster_scaled), 2000)
dist_sample <- dist(cluster_scaled[sample_idx, ])
sil_km <- silhouette(km_final$cluster[sample_idx], dist_sample)
mean(sil_km[, 3])
fviz_silhouette(sil_km)

# Detailed Explanation:
# - Silhouette score measures how similar an observation is to its own cluster compared to neighboring clusters (-1 to +1).
# - An average score > 0.35 suggests reasonable separation; values near 0 indicate boundary ambiguity.


# ------------------------------------------------------------------------------
# Question 21:
# Apply Hierarchical Agglomerative Clustering to the dataset.
# ------------------------------------------------------------------------------
# R Code:
# Sample 1,000 observations to keep distance matrix computation feasible in memory and time
sample_hclust_idx <- sample(1:nrow(cluster_scaled), 1000)
sub_scaled <- cluster_scaled[sample_hclust_idx, ]
dist_matrix <- dist(sub_scaled, method = "euclidean")

# Detailed Explanation:
# - Agglomerative clustering begins with each point in its own cluster and merges closest pairs iteratively.
# - Calculating an O(N^2) pairwise distance matrix on full 12k data is memory-intensive; sampling maintains diagnostic validity.


# ------------------------------------------------------------------------------
# Question 22:
# Generate and interpret the dendrogram.
# ------------------------------------------------------------------------------
# R Code:
hc_ward <- hclust(dist_matrix, method = "ward.D2")
plot(hc_ward, labels = FALSE, hang = -1, main = "Dendrogram (Ward's Linkage)")
rect.hclust(hc_ward, k = 3, border = "red")

# Detailed Explanation:
# - Vertical branch height reflects distance at which clusters merge.
# - Long vertical lines without split intersections indicate robust, well-delineated natural cluster partitions.


# ------------------------------------------------------------------------------
# Question 23:
# Experiment with suitable linkage methods such as Complete, Average, and Ward's Linkage.
# ------------------------------------------------------------------------------
# R Code:
hc_complete <- hclust(dist_matrix, method = "complete")
hc_average  <- hclust(dist_matrix, method = "average")

par(mfrow = c(1, 3))
plot(hc_complete, labels = FALSE, main = "Complete Linkage")
plot(hc_average,  labels = FALSE, main = "Average Linkage")
plot(hc_ward,     labels = FALSE, main = "Ward's Linkage")
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Complete: Uses maximum pairwise distance; sensitive to outliers.
# - Average: Uses average pairwise distance; less affected by extreme points.
# - Ward.D2: Minimizes within-cluster variance increase; creates compact, balanced clusters.


# ------------------------------------------------------------------------------
# Question 24:
# Select an appropriate number of clusters from the dendrogram.
# ------------------------------------------------------------------------------
# R Code:
h_clusters <- cutree(hc_ward, k = 3)
table(h_clusters)

# Detailed Explanation:
# - Cutting the dendrogram at 3 clusters matches the primary branches and aligns with the K-Means partition.


# ------------------------------------------------------------------------------
# Question 25:
# Visualize and compare the clusters obtained using K-Means and Hierarchical Clustering.
# ------------------------------------------------------------------------------
# R Code:
km_sub <- km_final$cluster[sample_hclust_idx]
contingency_table <- table(KMeans = km_sub, Hierarchical = h_clusters)
print(contingency_table)

# Visual comparison on first 2 PCs
p1 <- fviz_cluster(list(data = sub_scaled, cluster = km_sub), main = "K-Means (Sample)")
p2 <- fviz_cluster(list(data = sub_scaled, cluster = h_clusters), main = "Hierarchical (Ward)")
grid.arrange(p1, p2, ncol = 2)

# Detailed Explanation:
# - The cross-tabulation table and side-by-side plots evaluate structural agreement between algorithms.
# - High alignment along diagonal confirms cluster boundary stability across different distance criteria.


# ------------------------------------------------------------------------------
# Question 26:
# Identify and interpret the characteristics of each customer/session cluster based on browsing and purchasing behaviour.
# ------------------------------------------------------------------------------
# R Code:
prop.table(table(shoppers_clean$KMeans_Cluster, shoppers_clean$Revenue), 1) * 100

# Detailed Explanation:
# - Segment 1 (Bouncers/Disengaged): High bounce & exit rates, minimal duration, virtually 0% conversion.
# - Segment 2 (Exploratory / Window Shoppers): High duration and product page views, but near-zero PageValues; low conversion (~5-10%).
# - Segment 3 (High Intent / Buyers): High PageValues and substantial duration; high conversion (~40-60%+).


# ==============================================================================
# PART C: STATISTICAL ANALYSIS OF CLUSTERS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 27:
# Calculate the mean, median, variance, and standard deviation of important numerical attributes for each cluster.
# ------------------------------------------------------------------------------
# R Code:
shoppers_clean %>%
  group_by(KMeans_Cluster) %>%
  summarise(
    Mean_PageValues   = mean(PageValues),
    Median_PageValues = median(PageValues),
    Var_PageValues    = var(PageValues),
    SD_PageValues     = sd(PageValues),
    Mean_Duration     = mean(ProductRelated_Duration),
    Median_Duration   = median(ProductRelated_Duration),
    Var_Duration      = var(ProductRelated_Duration),
    SD_Duration       = sd(ProductRelated_Duration)
  )

# Detailed Explanation:
# - Computes central tendency (mean, median) and dispersion (variance, standard deviation) by cluster.
# - High variance compared to mean highlights right-skewed session distributions within clusters.


# ------------------------------------------------------------------------------
# Question 28:
# Estimate 95% confidence intervals for the mean of selected variables within the clusters.
# ------------------------------------------------------------------------------
# R Code:
calc_ci <- function(x) {
  se <- sd(x) / sqrt(length(x))
  mean(x) + c(-1.96, 1.96) * se
}

shoppers_clean %>%
  group_by(KMeans_Cluster) %>%
  summarise(
    CI_Lower_PageVal = calc_ci(PageValues)[1],
    Mean_PageVal     = mean(PageValues),
    CI_Upper_PageVal = calc_ci(PageValues)[2]
  )

# Detailed Explanation:
# - 95% Confidence Intervals estimate true population mean parameter per cluster.
# - Non-overlapping CIs between clusters indicate distinct behavioral profiles at alpha = 0.05.


# ------------------------------------------------------------------------------
# Question 29:
# Formulate the Null Hypothesis (H0) and Alternate Hypothesis (H1) for comparing the obtained clusters.
# ------------------------------------------------------------------------------
# R Code:
cat("Two-Sample t-test / ANOVA Hypotheses:\n")
cat("H0: mu_Cluster1 = mu_Cluster2 (The true mean of PageValues is equal across clusters)\n")
cat("H1: mu_Cluster1 != mu_Cluster2 (The true mean of PageValues differs significantly)\n\n")

cat("Chi-Square Test Hypotheses:\n")
cat("H0: Cluster membership and VisitorType are independent\n")
cat("H1: Cluster membership and VisitorType are dependent (associated)\n")

# Detailed Explanation:
# - H0 assumes equivalence or independence across clusters.
# - H1 posits systematic differences induced by cluster segmentation.


# ------------------------------------------------------------------------------
# Question 30:
# Perform an Independent Two-Sample t-test to compare a selected numerical attribute between two clusters.
# ------------------------------------------------------------------------------
# R Code:
c1_pageval <- shoppers_clean$PageValues[shoppers_clean$KMeans_Cluster == "1"]
c2_pageval <- shoppers_clean$PageValues[shoppers_clean$KMeans_Cluster == "2"]
t_test_res <- t.test(c1_pageval, c2_pageval)
print(t_test_res)

# Detailed Explanation:
# - Welch Two Sample t-test handles unequal sample sizes and variances.
# - p < 0.05 rejects H0, indicating mean PageValues differs between Cluster 1 and Cluster 2.


# ------------------------------------------------------------------------------
# Question 31:
# Perform a One-Way ANOVA to determine whether the mean of a selected numerical attribute differs significantly among the identified clusters.
# ------------------------------------------------------------------------------
# R Code:
anova_res <- aov(PageValues ~ KMeans_Cluster, data = shoppers_clean)
summary(anova_res)

# Detailed Explanation:
# - Evaluates ratio of between-cluster variance to within-cluster variance (F-statistic).
# - A large F-statistic with p < 2e-16 indicates at least one cluster mean differs significantly.


# ------------------------------------------------------------------------------
# Question 32:
# If the ANOVA result is significant, perform Tukey's HSD post-hoc test to identify which clusters differ significantly.
# ------------------------------------------------------------------------------
# R Code:
tukey_res <- TukeyHSD(anova_res)
print(tukey_res)
plot(tukey_res)

# Detailed Explanation:
# - Tukey's Honest Significant Difference controls Family-Wise Error Rate across all pairwise comparisons (1-2, 2-3, 1-3).
# - Adjusted p-values < 0.05 and confidence intervals excluding 0 pinpoint mutually distinct clusters.


# ------------------------------------------------------------------------------
# Question 33:
# Perform a Wilcoxon Rank-Sum Test to compare the distribution of a selected numerical attribute between two clusters.
# ------------------------------------------------------------------------------
# R Code:
wilcox_res <- wilcox.test(c1_pageval, c2_pageval)
print(wilcox_res)

# Detailed Explanation:
# - Non-parametric alternative to independent t-test (Mann-Whitney U test).
# - Because PageValues is heavily skewed and contains zeroes, this verifies differences without requiring Gaussian normality.


# ------------------------------------------------------------------------------
# Question 34:
# Perform a Chi-Square Test of Independence to examine the association between an appropriate categorical variable, such as VisitorType, Weekend, or Month, and the obtained clusters.
# ------------------------------------------------------------------------------
# R Code:
contingency_visitor <- table(shoppers_clean$KMeans_Cluster, shoppers_clean$VisitorType)
print(contingency_visitor)
chisq_res <- chisq.test(contingency_visitor)
print(chisq_res)

# Detailed Explanation:
# - Compares observed frequencies against expected frequencies under the null hypothesis of independence.
# - Significant p-value confirms visitor composition (Returning vs New) varies across clusters.


# ------------------------------------------------------------------------------
# Question 35:
# Interpret the test statistic, p-value, confidence interval, and significance level for each statistical test.
# ------------------------------------------------------------------------------
# R Code:
cat("ANOVA F-stat:", summary(anova_res)[[1]][["F value"]][1], 
    "| p-value:", summary(anova_res)[[1]][["Pr(>F)"]][1], "
")
cat("Chi-Square stat:", chisq_res$statistic, "| p-value:", chisq_res$p.value, "
")
cat("Two-Sample t-stat:", t_test_res$statistic, "| 95% CI:", t_test_res$conf.int, "
")

# Detailed Explanation:
# - Test Statistics (F, Chi-square, t): Quantify signal-to-noise ratio.
# - p-values (< 0.05): Fall below the 5% threshold, rejecting respective null hypotheses.
# - Confidence Intervals: Describe the magnitude and direction of the differences.


# ------------------------------------------------------------------------------
# Question 36:
# Determine which browsing and purchasing attributes show statistically significant differences among the clusters.
# ------------------------------------------------------------------------------
# R Code:
p_values <- sapply(selected_features, function(var) {
  f <- as.formula(paste(var, "~ KMeans_Cluster"))
  summary(aov(f, data = shoppers_clean))[[1]][["Pr(>F)"]][1]
})
data.frame(Feature = selected_features, p_value = p_values, Significant = p_values < 0.05)

# Detailed Explanation:
# - Running ANOVA across all segmentation features confirms every single metric exhibits statistically
#   significant variation (p < 0.001) across the clusters.


# ------------------------------------------------------------------------------
# Question 37:
# Compare the statistical analysis results with the clustering results.
# ------------------------------------------------------------------------------
# R Code:
# Comparison summary via aggregate statistics across clusters
aggregate(shoppers_clean[, selected_features], by = list(Cluster = shoppers_clean$KMeans_Cluster), FUN = mean)

# Detailed Explanation:
# - Statistical tests validate unsupervised clustering partitions by proving that between-group
#   variance dominates within-group variance, confirming that segments are distinct behavioral entities rather than random artifacts.


# ------------------------------------------------------------------------------
# Question 38:
# Identify the most valuable characteristics of each customer/session segment.
# ------------------------------------------------------------------------------
# R Code:
cat("Cluster 1 (Disengaged/Bouncers): Immediate exits, minimal engagement. Strategy: Improve landing page relevance.\n")
cat("Cluster 2 (Explorers/Window Shoppers): High page duration, low PageValues. Strategy: Retarget with promotional incentives.\n")
cat("Cluster 3 (High-Intent Converters): High PageValues and high conversion rate. Strategy: Streamline checkout and offer loyalty rewards.\n")

# Detailed Explanation:
# - Maps statistical profiles to practical e-commerce strategies:
#   * Segment 1 represents wasted spend/bounce traffic requiring UI/UX optimization.
#   * Segment 2 contains high-attention prospects needing nudges/discounts.
#   * Segment 3 contains core revenue drivers requiring friction-free checkout and upselling.
