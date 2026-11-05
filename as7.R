library(readr)
library(dplyr)
library(corrplot)
library(cluster)
library(psych)

cat("1\n")
df <- read.csv("online_shoppers_intention.csv")
head(df)

cat("2\n")
dim(df)

cat("3\n")
str(df)
summary(df)

cat("4\n")
colSums(is.na(df))

cat("5\n")
df_clean <- df[!duplicated(df), ]
cat("Original rows:", nrow(df), "| Cleaned rows:", nrow(df_clean), "\n")
cat("Duplicate rows removed:", nrow(df) - nrow(df_clean), "\n")

cat("6\n")
num_cols <- c(
  "Administrative",
  "Administrative_Duration",
  "Informational",
  "Informational_Duration",
  "ProductRelated",
  "ProductRelated_Duration",
  "BounceRates",
  "ExitRates",
  "PageValues",
  "SpecialDay"
)
df_num <- df_clean[, num_cols]

cat("7\n")
scaled_num <- scale(df_num)

cat("8\n")
par(mfrow = c(2, 5), mar = c(3, 3, 2, 1))
for (col in num_cols) {
  hist(df_num[[col]], main = col, col = "skyblue", xlab = "", border = "white")
}
par(mfrow = c(1, 1))

cat("9\n")
cor_matrix <- cor(df_num)
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)

cat("10\n")
high_corr <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)
print(high_corr)

cat("11\n")
pca_res <- prcomp(scaled_num, center = TRUE, scale. = FALSE)
summary(pca_res)

cat("12\n")
var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
plot(
  var_explained,
  type = "b",
  pch = 19,
  col = "blue",
  xlab = "Principal Component",
  ylab = "Proportion of Variance Explained",
  main = "Scree Plot - Variance Explained by PCs"
)

cat("13\n")
plot(
  pca_res$x[, 1],
  pca_res$x[, 2],
  col = rgb(0, 0.4, 0.8, 0.3),
  pch = 16,
  xlab = "PC1",
  ylab = "PC2",
  main = "PCA Observation Plot (PC1 vs PC2)"
)

cat("14\n")
wcss <- numeric(10)
set.seed(123)
for (k in 1:10) {
  wcss[k] <- sum(kmeans(scaled_num, centers = k, nstart = 10)$withinss)
}
plot(
  1:10,
  wcss,
  type = "b",
  pch = 19,
  col = "darkgreen",
  xlab = "Number of Clusters (k)",
  ylab = "Total Within-Cluster SS",
  main = "Elbow Method for Optimal k"
)

cat("15\n")
set.seed(123)
km_res <- kmeans(scaled_num, centers = 3, nstart = 25)
df_clean$km_cluster <- as.factor(km_res$cluster)

cat("16\n")
centroids_km <- aggregate(
  df_num,
  by = list(Cluster = km_res$cluster),
  mean
)
print(round(centroids_km, 2))

cat("17\n")
plot(
  pca_res$x[, 1],
  pca_res$x[, 2],
  col = km_res$cluster,
  pch = 16,
  xlab = "PC1",
  ylab = "PC2",
  main = "K-Means Clusters (k = 3)"
)
legend("topright", legend = paste("Cluster", 1:3), col = 1:3, pch = 16)

cat("18\n")
sil_km <- silhouette(km_res$cluster, dist(scaled_num))
cat("Mean Silhouette Width:", mean(sil_km[, 3]), "\n")
plot(sil_km, col = 1:3, border = NA, main = "Silhouette Plot for K-Means")

cat("19\n")
dist_matrix <- dist(scaled_num)
hc_ward <- hclust(dist_matrix, method = "ward.D2")
plot(
  hc_ward,
  main = "Dendrogram (Ward's Linkage)",
  xlab = "",
  sub = "",
  labels = FALSE
)
rect.hclust(hc_ward, k = 3, border = "red")

cat("20\n")
df_clean$hc_cluster <- as.factor(cutree(hc_ward, k = 3))
table(KMeans = df_clean$km_cluster, Hierarchical = df_clean$hc_cluster)

cat("21\n")
describeBy(df_clean$PageValues, group = df_clean$km_cluster, mat = TRUE)

cat("22\n")
calc_ci <- function(x) {
  res <- t.test(x)
  c(
    Mean = mean(x),
    Std_Dev = sd(x),
    CI_Lower = res$conf.int[1],
    CI_Upper = res$conf.int[2]
  )
}
by(df_clean$PageValues, df_clean$km_cluster, calc_ci)

cat("23\n")
by(df_clean$ProductRelated_Duration, df_clean$km_cluster, calc_ci)

cat("24\n")
c1_pv <- df_clean$PageValues[df_clean$km_cluster == 1]
c2_pv <- df_clean$PageValues[df_clean$km_cluster == 2]
ttest_res <- t.test(c1_pv, c2_pv, var.equal = FALSE)
print(ttest_res)

cat("25\n")
wilcox_res <- wilcox.test(c1_pv, c2_pv)
print(wilcox_res)

cat("26\n")
anova_res <- aov(PageValues ~ km_cluster, data = df_clean)
summary(anova_res)

cat("27\n")
tukey_res <- TukeyHSD(anova_res)
print(tukey_res)

cat("28\n")
chisq_visitor <- chisq.test(table(df_clean$km_cluster, df_clean$VisitorType))
print(chisq_visitor)

cat("29\n")
chisq_revenue <- chisq.test(table(df_clean$km_cluster, df_clean$Revenue))
print(chisq_revenue)

cat("30\n")
aggregate(
  cbind(
    Administrative,
    Administrative_Duration,
    Informational,
    Informational_Duration,
    ProductRelated,
    ProductRelated_Duration,
    BounceRates,
    ExitRates,
    PageValues,
    SpecialDay
  ) ~ km_cluster,
  data = df_clean,
  FUN = mean
)

cat("31\n")
aggregate(
  Revenue ~ km_cluster,
  data = df_clean,
  FUN = function(x) mean(as.logical(x)) * 100
)

cat("32\n")
table(df_clean$km_cluster, df_clean$VisitorType)

cat("33\n")
table(df_clean$km_cluster, df_clean$Revenue)
