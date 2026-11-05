titanic_df <- read.csv("Titanic-Dataset.csv")

1
head(titanic_df, 10)
tail(titanic_df, 10)

2
nrow(titanic_df)
ncol(titanic_df)
dim(titanic_df)

3
str(titanic_df)

4
num_vars <- names(titanic_df)[sapply(titanic_df, is.numeric)]
cat_vars <- names(titanic_df)[sapply(titanic_df, function(x) is.character(x) | is.factor(x))]
print(num_vars)
print(cat_vars)

5
colSums(is.na(titanic_df))

6
age_median <- median(titanic_df$Age, na.rm = TRUE)
titanic_df$Age[is.na(titanic_df$Age)] <- age_median

7
get_mode <- function(v) {
  uniqv <- unique(na.omit(v))
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
mode_embarked <- get_mode(titanic_df$Embarked)
titanic_df$Embarked[is.na(titanic_df$Embarked)] <- mode_embarked

8
duplicate_count <- sum(duplicated(titanic_df))
print(duplicate_count)
titanic_df <- titanic_df[!duplicated(titanic_df), ]

9
titanic_df$Survived <- factor(titanic_df$Survived)
titanic_df$Pclass <- factor(titanic_df$Pclass)
titanic_df$Sex <- factor(titanic_df$Sex)
titanic_df$Embarked <- factor(titanic_df$Embarked)

10
names(titanic_df) <- make.names(names(titanic_df))

11
titanic_df$Age_scaled <- as.numeric(scale(titanic_df$Age))
titanic_df$Fare_scaled <- as.numeric(scale(titanic_df$Fare))

12
summary(titanic_df[sapply(titanic_df, is.numeric)])

13
fare_mean <- mean(titanic_df$Fare, na.rm = TRUE)
fare_median <- median(titanic_df$Fare, na.rm = TRUE)
fare_mode <- get_mode(titanic_df$Fare)
fare_var <- var(titanic_df$Fare, na.rm = TRUE)
fare_sd <- sd(titanic_df$Fare, na.rm = TRUE)
cat("Mean:", fare_mean, "\nMedian:", fare_median, "\nMode:", fare_mode,
    "\nVariance:", fare_var, "\nStandard Deviation:", fare_sd, "\n")

14
age_range <- range(titanic_df$Age, na.rm = TRUE)
age_quartiles <- quantile(titanic_df$Age, na.rm = TRUE)
age_iqr <- IQR(titanic_df$Age, na.rm = TRUE)
cat("Range:", age_range[1], "to", age_range[2], "\n")
print(age_quartiles)
cat("IQR:", age_iqr, "\n")

15
par(mfrow = c(1, 2))
hist(titanic_df$Age, main = "Histogram of Age", xlab = "Age",
     col = "skyblue", border = "white")
hist(titanic_df$Fare, main = "Histogram of Fare", xlab = "Fare",
     col = "coral", border = "white")
par(mfrow = c(1, 1))

16
plot(density(titanic_df$Age), main = "Density Plot of Age",
     xlab = "Age", col = "blue", lwd = 2)
plot(density(titanic_df$Fare), main = "Density Plot of Fare",
     xlab = "Fare", col = "red", lwd = 2)

17
boxplot(titanic_df$Age, main = "Boxplot of Age", ylab = "Age",
        col = "lightgreen")
boxplot(titanic_df$Fare, main = "Boxplot of Fare", ylab = "Fare",
        col = "gold")
age_outliers <- boxplot.stats(titanic_df$Age)$out
fare_outliers <- boxplot.stats(titanic_df$Fare)$out
cat("Age Outliers Count:", length(age_outliers), "\n")
cat("Fare Outliers Count:", length(fare_outliers), "\n")

18
par(mfrow = c(1, 2))
barplot(table(titanic_df$Pclass), main = "Passenger Class Frequency",
        xlab = "Class", ylab = "Count", col = "steelblue")
barplot(table(titanic_df$Sex), main = "Gender Frequency",
        xlab = "Gender", ylab = "Count", col = c("pink", "lightblue"))
par(mfrow = c(1, 1))

19
cat("1. Age Distribution: Right-skewed. Most passengers were young adults, and the median age was approximately 28 years.\n")
cat("2. Fare Distribution: Highly right-skewed. Most passengers paid relatively low fares, while a small number paid very high fares.\n")

20
plot(titanic_df$Age, titanic_df$Fare,
     main = "Scatter Plot: Age vs Fare",
     xlab = "Age (Years)", ylab = "Fare ($)",
     col = "darkblue", pch = 19)

21
pearson_cor <- cor(titanic_df$Age, titanic_df$Fare, method = "pearson",
                   use = "complete.obs")
cat("Pearson Correlation (Age vs Fare):", round(pearson_cor, 4), "\n")

22
boxplot(Fare ~ Pclass, data = titanic_df,
        main = "Fare by Passenger Class",
        xlab = "Passenger Class", ylab = "Fare ($)",
        col = c("gold", "lightgreen", "skyblue"))

23
boxplot(Age ~ Sex, data = titanic_df,
        main = "Age by Gender",
        xlab = "Gender", ylab = "Age (Years)",
        col = c("pink", "lightblue"))

24
surv_gender_tab <- table(titanic_df$Survived, titanic_df$Sex)
barplot(surv_gender_tab,
        main = "Survival Status by Gender",
        xlab = "Gender", ylab = "Passenger Count",
        col = c("coral", "seagreen3"),
        legend.text = c("Died (0)", "Survived (1)"),
        args.legend = list(x = "topleft"))

25
pclass_surv_tab <- table(Pclass = titanic_df$Pclass,
                         Survived = titanic_df$Survived)
print(pclass_surv_tab)

26
chi_test_class <- chisq.test(pclass_surv_tab)
print(chi_test_class)

27
survival_by_gender <- prop.table(surv_gender_tab, margin = 2)
survival_by_class <- prop.table(pclass_surv_tab, margin = 1)
cat("Survival proportion by gender:\n")
print(round(survival_by_gender, 4))
cat("Survival proportion by passenger class:\n")
print(round(survival_by_class, 4))
cat("Females had a higher survival rate than males. Survival rates decreased from first class to third class.\n")

28
num_cols <- titanic_df[, c("Age", "Fare", "SibSp", "Parch")]
pairs(num_cols, main = "Scatter Plot Matrix", col = "steelblue")

29
cor_matrix <- cor(num_cols, use = "complete.obs")
print(round(cor_matrix, 4))

30
heatmap(cor_matrix, main = "Correlation Heatmap",
        Rowv = NA, Colv = NA, col = cm.colors(256))

31
pca_res <- prcomp(num_cols, scale. = TRUE)
summary(pca_res)

32
pc_scores <- as.data.frame(pca_res$x)
plot(pc_scores$PC1, pc_scores$PC2,
     col = ifelse(titanic_df$Survived == 1, "seagreen3", "red3"),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "PCA Score Plot: PC1 vs PC2")
legend("topright", legend = c("Died (0)", "Survived (1)"),
       col = c("red3", "seagreen3"), pch = 19)

33
loadings <- pca_res$rotation
pc1_top <- names(which.max(abs(loadings[, "PC1"])))
pc2_top <- names(which.max(abs(loadings[, "PC2"])))
cat("Variable with highest contribution to PC1:", pc1_top,
    "(Loading:", round(loadings[pc1_top, "PC1"], 4), ")\n")
cat("Variable with highest contribution to PC2:", pc2_top,
    "(Loading:", round(loadings[pc2_top, "PC2"], 4), ")\n")

34
cat("1. SibSp and Parch show positive correlation, reflecting family travel patterns.\n")
cat("2. The first two principal components summarize a substantial proportion of the total variance.\n")
cat("3. Survival patterns may vary with passenger age, fare, family size, gender, and passenger class.\n")
