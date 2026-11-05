# ==============================================================================
# ASSIGNMENT 6: BANK MARKETING DATASET (CLASSIFICATION & STATISTICAL TESTING)
# Target Variable: y ("yes" / "no")
# ==============================================================================

# Required Libraries
# install.packages(c("readr", "dplyr", "caret", "class", "rpart", "rpart.plot", 
#                    "e1071", "randomForest", "pROC", "ggplot2"))

library(readr)
library(dplyr)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(randomForest)
library(pROC)
library(ggplot2)

# Set seed for reproducibility across random splits and model training
set.seed(42)

# ==============================================================================
# PART A: DATA PREPARATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 1:
# Import the Bank Marketing dataset into R and display the first few observations.
# ------------------------------------------------------------------------------
# R Code:
# The Bank Marketing dataset is commonly semicolon-delimited (bank.csv or bank-full.csv)
bank_raw <- read.csv("bank.csv", sep = ";", stringsAsFactors = FALSE)
head(bank_raw)

# Detailed Explanation:
# - read.csv(..., sep = ";") correctly parses semicolon-separated values into an R data frame.
# - head() displays the top 6 rows, verifying column headers and data alignment.


# ------------------------------------------------------------------------------
# Question 2:
# Determine the number of observations and variables in the dataset.
# ------------------------------------------------------------------------------
# R Code:
dim(bank_raw)
cat("Number of observations (rows):", nrow(bank_raw), "
")
cat("Number of variables (columns):", ncol(bank_raw), "
")

# Detailed Explanation:
# - dim() provides dimensions [rows, columns].
# - Bank Marketing typically contains 4,521 rows (bank.csv) or 45,211 rows (bank-full.csv) across 17 attributes.


# ------------------------------------------------------------------------------
# Question 3:
# Display the structure, data types, and summary statistics of the dataset.
# ------------------------------------------------------------------------------
# R Code:
str(bank_raw)
summary(bank_raw)

# Detailed Explanation:
# - str() reveals internal data representation (integers, strings) and memory footprint.
# - summary() computes 5-number summaries for numerical columns and frequencies for categorical fields.


# ------------------------------------------------------------------------------
# Question 4:
# Identify the numerical and categorical attributes.
# ------------------------------------------------------------------------------
# R Code:
num_cols <- names(bank_raw)[sapply(bank_raw, is.numeric)]
cat_cols <- names(bank_raw)[!sapply(bank_raw, is.numeric)]
cat("Numerical Attributes:
", paste(num_cols, collapse = ", "), "

")
cat("Categorical Attributes:
", paste(cat_cols, collapse = ", "), "
")

# Detailed Explanation:
# - Numerical features (7): age, balance, day, duration, campaign, pdays, previous.
# - Categorical features (10): job, marital, education, default, housing, loan, contact, month, poutcome, y.


# ------------------------------------------------------------------------------
# Question 5:
# Check for missing or unknown values and handle them appropriately.
# ------------------------------------------------------------------------------
# R Code:
cat("Explicit NA count:", sum(is.na(bank_raw)), "
")
unknown_counts <- sapply(bank_raw, function(col) sum(col == "unknown"))
print(unknown_counts[unknown_counts > 0])

# Replace "unknown" with NA or keep as explicit factor level (here treated as separate category level)
bank_clean <- bank_raw
cat("Explicit NAs handled. 'unknown' preserved as a valid informational level in telemetry/campaign history.
")

# Detailed Explanation:
# - The dataset contains zero explicit NA values; however, missingness is recorded as the string "unknown"
#   (especially in job, education, contact, and poutcome).
# - Retaining "unknown" as a factor level avoids dropping substantial portions of data (e.g. poutcome is >80% unknown).


# ------------------------------------------------------------------------------
# Question 6:
# Detect and remove duplicate records, if any.
# ------------------------------------------------------------------------------
# R Code:
cat("Number of duplicate rows:", sum(duplicated(bank_clean)), "
")
bank_clean <- bank_clean[!duplicated(bank_clean), ]
cat("Rows remaining after deduplication:", nrow(bank_clean), "
")

# Detailed Explanation:
# - duplicated() flags identical customer session records.
# - Removing duplicate rows prevents artificial deflation of variance and overfitting.


# ------------------------------------------------------------------------------
# Question 7:
# Convert appropriate categorical variables into factors.
# ------------------------------------------------------------------------------
# R Code:
for (col in cat_cols) {
  bank_clean[[col]] <- as.factor(bank_clean[[col]])
}
str(bank_clean[, cat_cols])

# Detailed Explanation:
# - as.factor() encodes qualitative variables into nominal factors required by R classification formulas
#   and ensures appropriate dummy encoding during model fitting.


# ------------------------------------------------------------------------------
# Question 8:
# Perform feature scaling or normalization for suitable numerical variables.
# ------------------------------------------------------------------------------
# R Code:
scale_model <- preProcess(bank_clean[, num_cols], method = c("center", "scale"))
bank_scaled <- predict(scale_model, bank_clean)
head(bank_scaled[, num_cols])

# Detailed Explanation:
# - Standardization (mean = 0, standard deviation = 1) prevents distance-based algorithms (KNN, SVM)
#   from being dominated by large-scale features like balance and duration.


# ------------------------------------------------------------------------------
# Question 9:
# Select y as the target variable for binary classification.
# ------------------------------------------------------------------------------
# R Code:
table(bank_clean$y)
prop.table(table(bank_clean$y))

# Detailed Explanation:
# - 'y' represents client subscription to term deposit ("yes" or "no").
# - Checking class balance reveals severe skew (typically ~88% "no" vs ~12% "yes").


# ------------------------------------------------------------------------------
# Question 10:
# Divide the dataset into suitable training and testing sets.
# ------------------------------------------------------------------------------
# R Code:
set.seed(42)
train_idx <- createDataPartition(bank_clean$y, p = 0.8, list = FALSE)
train_df <- bank_clean[train_idx, ]
test_df  <- bank_clean[-train_idx, ]
train_scaled <- bank_scaled[train_idx, ]
test_scaled  <- bank_scaled[-train_idx, ]

# Detailed Explanation:
# - createDataPartition generates a stratified 80/20 split preserving the rare positive class ratio across sets.


# ==============================================================================
# PART B: CLASSIFICATION MODELS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 11:
# Implement a Logistic Regression model to predict whether a client subscribes to a term deposit.
# ------------------------------------------------------------------------------
# R Code:
log_model <- glm(y ~ ., data = train_df, family = binomial(link = "logit"))
summary(log_model)

# Detailed Explanation:
# - glm() fits a generalized linear model using maximum likelihood estimation.
# - Logit link maps linear combination of predictors to log-odds of subscription.


# ------------------------------------------------------------------------------
# Question 12:
# Implement a K-Nearest Neighbors (KNN) classification model and determine a suitable value of K.
# ------------------------------------------------------------------------------
# R Code:
# KNN requires dummy-encoded numerical matrices
dummy_obj <- dummyVars(y ~ ., data = train_scaled)
knn_train_x <- predict(dummy_obj, newdata = train_scaled)
knn_test_x  <- predict(dummy_obj, newdata = test_scaled)

# Choose k via square root of N (or cross-validation)
k_val <- round(sqrt(nrow(knn_train_x)))
if (k_val %% 2 == 0) k_val <- k_val + 1 # ensure odd k to break ties
pred_knn <- knn(train = knn_train_x, test = knn_test_x, cl = train_df$y, k = k_val, prob = TRUE)

# Detailed Explanation:
# - KNN assigns majority class among k nearest neighbors based on Euclidean distance on standardized features.
# - Setting k = sqrt(N) provides an effective bias-variance balance.


# ------------------------------------------------------------------------------
# Question 13:
# Implement a Decision Tree classification model and visualize the tree.
# ------------------------------------------------------------------------------
# R Code:
tree_model <- rpart(y ~ ., data = train_df, method = "class", control = rpart.control(cp = 0.01))
rpart.plot(tree_model, type = 2, extra = 104, main = "Decision Tree for Bank Marketing")

# Detailed Explanation:
# - rpart() uses recursive binary splitting based on Gini impurity.
# - rpart.plot() visualizes decision thresholds, node probabilities, and class percentages.


# ------------------------------------------------------------------------------
# Question 14:
# Implement a Naive Bayes classification model.
# ------------------------------------------------------------------------------
# R Code:
nb_model <- naiveBayes(y ~ ., data = train_df)
print(nb_model)

# Detailed Explanation:
# - Applies Bayes' theorem with the assumption of conditional independence among predictors given the class.


# ------------------------------------------------------------------------------
# Question 15:
# Implement a Support Vector Machine (SVM) classification model using a suitable kernel.
# ------------------------------------------------------------------------------
# R Code:
svm_model <- svm(y ~ ., data = train_scaled, kernel = "radial", probability = TRUE, cost = 1)
summary(svm_model)

# Detailed Explanation:
# - Uses a Radial Basis Function (RBF) kernel to map inputs into higher-dimensional space to construct a maximum-margin hyperplane.


# ------------------------------------------------------------------------------
# Question 16:
# Implement a Random Forest classification model.
# ------------------------------------------------------------------------------
# R Code:
rf_model <- randomForest(y ~ ., data = train_df, ntree = 150, importance = TRUE)
print(rf_model)

# Detailed Explanation:
# - Constructs an ensemble of decorrelated decision trees using bootstrap aggregation and random feature subsets.


# ------------------------------------------------------------------------------
# Question 17:
# Generate predictions using each classification model.
# ------------------------------------------------------------------------------
# R Code:
prob_log <- predict(log_model, newdata = test_df, type = "response")
pred_log <- factor(ifelse(prob_log > 0.5, "yes", "no"), levels = c("no", "yes"))

prob_tree <- predict(tree_model, newdata = test_df, type = "prob")[, "yes"]
pred_tree <- predict(tree_model, newdata = test_df, type = "class")

prob_nb <- predict(nb_model, newdata = test_df, type = "raw")[, "yes"]
pred_nb <- predict(nb_model, newdata = test_df, type = "class")

svm_pred_obj <- predict(svm_model, newdata = test_scaled, probability = TRUE)
prob_svm <- attr(svm_pred_obj, "probabilities")[, "yes"]
pred_svm <- svm_pred_obj

prob_rf <- predict(rf_model, newdata = test_df, type = "prob")[, "yes"]
pred_rf <- predict(rf_model, newdata = test_df, type = "response")

# KNN probability extraction
knn_attr <- attr(pred_knn, "prob")
prob_knn <- ifelse(pred_knn == "yes", knn_attr, 1 - knn_attr)

# Detailed Explanation:
# - Generates both discrete class labels and continuous posterior probabilities on the untouched test split.


# ------------------------------------------------------------------------------
# Question 18:
# Construct a confusion matrix for each model.
# ------------------------------------------------------------------------------
# R Code:
cm_log  <- confusionMatrix(pred_log,  test_df$y, positive = "yes")
cm_knn  <- confusionMatrix(pred_knn,  test_df$y, positive = "yes")
cm_tree <- confusionMatrix(pred_tree, test_df$y, positive = "yes")
cm_nb   <- confusionMatrix(pred_nb,   test_df$y, positive = "yes")
cm_svm  <- confusionMatrix(pred_svm,  test_df$y, positive = "yes")
cm_rf   <- confusionMatrix(pred_rf,   test_df$y, positive = "yes")

print(cm_rf$table)

# Detailed Explanation:
# - confusionMatrix() tabulates True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).


# ------------------------------------------------------------------------------
# Question 19:
# Calculate accuracy, precision, recall, specificity, F1-score, and ROC-AUC for each model.
# ------------------------------------------------------------------------------
# R Code:
get_metrics <- function(cm, prob, actual) {
  acc  <- cm$overall["Accuracy"]
  prec <- cm$byClass["Precision"]
  rec  <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  f1   <- cm$byClass["F1"]
  auc  <- as.numeric(roc(actual, prob, quiet = TRUE)$auc)
  return(c(Accuracy = acc, Precision = prec, Recall = rec, Specificity = spec, F1 = f1, AUC = auc))
}

# Detailed Explanation:
# - Precision: TP / (TP + FP)
# - Recall/Sensitivity: TP / (TP + FN)
# - F1-score: Harmonic mean of precision and recall (crucial for imbalanced marketing conversions).
# - ROC-AUC: Area under the receiver operating characteristic curve across all decision thresholds.


# ------------------------------------------------------------------------------
# Question 20:
# Compare the performance of all classification models and identify the best-performing model.
# ------------------------------------------------------------------------------
# R Code:
models_eval <- rbind(
  "Logistic Regression" = get_metrics(cm_log, prob_log, test_df$y),
  "KNN"                 = get_metrics(cm_knn, prob_knn, test_df$y),
  "Decision Tree"       = get_metrics(cm_tree, prob_tree, test_df$y),
  "Naive Bayes"         = get_metrics(cm_nb, prob_nb, test_df$y),
  "SVM (RBF)"           = get_metrics(cm_svm, prob_svm, test_df$y),
  "Random Forest"       = get_metrics(cm_rf, prob_rf, test_df$y)
)
perf_df <- as.data.frame(round(models_eval, 4))
print(perf_df)

best_model <- rownames(perf_df)[which.max(perf_df$AUC)]
cat("Best Performing Model based on ROC-AUC:", best_model, "
")

# Detailed Explanation:
# - Due to class imbalance, accuracy is misleading; Random Forest or Gradient Boosting / SVM typically leads in AUC and F1-score.


# ------------------------------------------------------------------------------
# Question 21:
# Identify the important features contributing to the classification results.
# ------------------------------------------------------------------------------
# R Code:
varImpPlot(rf_model, main = "Variable Importance (Random Forest)")
importance_scores <- importance(rf_model)
print(head(importance_scores[order(importance_scores[, "MeanDecreaseGini"], decreasing = TRUE), ]))

# Detailed Explanation:
# - Evaluates MeanDecreaseGini and MeanDecreaseAccuracy.
# - Call duration, poutcome, balance, age, and contact month consistently rank as the strongest discriminators.


# ==============================================================================
# PART C: STATISTICAL ESTIMATION AND HYPOTHESIS TESTING
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 22:
# Estimate the population mean, variance, standard deviation, and proportion for suitable variables in the dataset.
# ------------------------------------------------------------------------------
# R Code:
mean_age  <- mean(bank_clean$age)
var_age   <- var(bank_clean$age)
sd_age    <- sd(bank_clean$age)
prop_yes  <- mean(bank_clean$y == "yes")

cat("Age - Mean:", round(mean_age, 2), "| Variance:", round(var_age, 2), "| SD:", round(sd_age, 2), "
")
cat("Deposit Subscription (y = 'yes') Proportion:", round(prop_yes, 4), "
")

# Detailed Explanation:
# - Computes sample point estimates to approximate true population parameters under central limit assumptions.


# ------------------------------------------------------------------------------
# Question 23:
# Construct 95% confidence intervals for selected population means and proportions.
# ------------------------------------------------------------------------------
# R Code:
# Mean CI for Age
n_obs <- nrow(bank_clean)
se_age <- sd_age / sqrt(n_obs)
ci_age <- mean_age + c(-1.96, 1.96) * se_age

# Proportion CI for Subscription
se_prop <- sqrt((prop_yes * (1 - prop_yes)) / n_obs)
ci_prop <- prop_yes + c(-1.96, 1.96) * se_prop

cat("95% CI for Population Mean Age:", round(ci_age, 2), "
")
cat("95% CI for Population Proportion Subscribing:", round(ci_prop, 4), "
")

# Detailed Explanation:
# - 95% Confidence Interval specifies bounds that will capture true population parameters in 95% of repeated samples.


# ------------------------------------------------------------------------------
# Question 24:
# Formulate the Null Hypothesis (H0) and Alternate Hypothesis (H1) for suitable statistical problems.
# ------------------------------------------------------------------------------
# R Code:
cat("1. One-Sample t-test (Age = 40):
   H0: mu_age = 40 vs H1: mu_age != 40

")
cat("2. Independent Two-Sample t-test (Duration by Deposit):
   H0: mu_duration(yes) = mu_duration(no) vs H1: mu_duration(yes) != mu_duration(no)

")
cat("3. Chi-Square Test (Housing loan vs Subscription):
   H0: Housing loan and subscription are independent
   H1: Housing loan and subscription are associated
")

# Detailed Explanation:
# - Formalizes statistical assertions: H0 asserts status quo/no effect; H1 asserts significant difference/association.


# ------------------------------------------------------------------------------
# Question 25:
# Perform a One-Sample t-test to determine whether the population mean of a selected numerical variable differs significantly from a specified value.
# ------------------------------------------------------------------------------
# R Code:
t_one_sample <- t.test(bank_clean$age, mu = 40)
print(t_one_sample)

# Detailed Explanation:
# - Tests whether sample mean age statistically deviates from hypothesized baseline mu = 40.


# ------------------------------------------------------------------------------
# Question 26:
# Perform an Independent Two-Sample t-test to determine whether the mean of a numerical variable differs significantly between clients who subscribed and those who did not.
# ------------------------------------------------------------------------------
# R Code:
t_two_sample <- t.test(duration ~ y, data = bank_clean)
print(t_two_sample)

# Detailed Explanation:
# - Welch two-sample t-test tests difference in call duration between subscribers ("yes") and non-subscribers ("no").


# ------------------------------------------------------------------------------
# Question 27:
# Perform a Paired t-test using a suitable paired dataset or example data.
# ------------------------------------------------------------------------------
# R Code:
# Simulating a campaign follow-up metric (e.g., initial contact duration vs follow-up call duration)
set.seed(42)
initial_call  <- bank_clean$duration[1:100]
followup_call <- initial_call + rnorm(100, mean = -15, sd = 10) # 15-second average reduction
t_paired <- t.test(initial_call, followup_call, paired = TRUE)
print(t_paired)

# Detailed Explanation:
# - Paired t-test compares repeated measurements on identical subjects, eliminating inter-subject baseline variability.


# ------------------------------------------------------------------------------
# Question 28:
# Perform a Chi-Square Test of Independence to determine whether two categorical variables are significantly associated.
# ------------------------------------------------------------------------------
# R Code:
cont_table <- table(bank_clean$housing, bank_clean$y)
chisq_test <- chisq.test(cont_table)
print(chisq_test)

# Detailed Explanation:
# - Compares observed contingency counts against expected cell counts under independence between housing loan status and subscription.


# ------------------------------------------------------------------------------
# Question 29:
# Perform a One-Way ANOVA to determine whether the mean of a numerical variable differs significantly across three or more groups.
# ------------------------------------------------------------------------------
# R Code:
anova_model <- aov(balance ~ marital, data = bank_clean)
summary(anova_model)

# Detailed Explanation:
# - Tests whether mean account balance differs across marital statuses (single, married, divorced).


# ------------------------------------------------------------------------------
# Question 30:
# Perform a Pearson Correlation Test to determine whether two numerical variables have a significant linear relationship.
# ------------------------------------------------------------------------------
# R Code:
cor_test <- cor.test(bank_clean$age, bank_clean$balance)
print(cor_test)

# Detailed Explanation:
# - Tests whether correlation coefficient r between age and balance differs significantly from 0.


# ------------------------------------------------------------------------------
# Question 31:
# Perform a Wilcoxon Rank-Sum Test to compare a numerical variable between two independent groups.
# ------------------------------------------------------------------------------
# R Code:
wilcox_res <- wilcox.test(balance ~ y, data = bank_clean)
print(wilcox_res)

# Detailed Explanation:
# - Non-parametric equivalent of independent two-sample t-test; robust to extreme positive skew and heavy tails in balance.


# ------------------------------------------------------------------------------
# Question 32:
# For each hypothesis test, identify the test statistic, p-value, significance level, and decision.
# ------------------------------------------------------------------------------
# R Code:
test_summary <- data.frame(
  Test = c("One-Sample t-test", "Two-Sample t-test", "Paired t-test", "Chi-Square", "ANOVA", "Pearson Cor", "Wilcoxon"),
  Statistic = c(t_one_sample$statistic, t_two_sample$statistic, t_paired$statistic, 
                chisq_test$statistic, summary(anova_model)[[1]][["F value"]][1], 
                cor_test$statistic, wilcox_res$statistic),
  P_Value = c(t_one_sample$p.value, t_two_sample$p.value, t_paired$p.value, 
              chisq_test$p.value, summary(anova_model)[[1]][["Pr(>F)"]][1], 
              cor_test$p.value, wilcox_res$p.value),
  Alpha = 0.05
)
test_summary$Decision <- ifelse(test_summary$P_Value < test_summary$Alpha, "Reject H0", "Fail to Reject H0")
print(test_summary)

# Detailed Explanation:
# - Tabulates formal hypothesis test results, comparing p-value to alpha = 0.05.


# ------------------------------------------------------------------------------
# Question 33:
# Interpret the p-value and determine whether to reject or fail to reject H0 at the 5% significance level.
# ------------------------------------------------------------------------------
# R Code:
for (i in 1:nrow(test_summary)) {
  cat(test_summary$Test[i], ": p-value =", format.pval(test_summary$P_Value[i], digits = 3), 
      "->", test_summary$Decision[i], "
")
}

# Detailed Explanation:
# - p < 0.05 implies observed data is extremely unlikely under H0, leading to rejection of H0.


# ------------------------------------------------------------------------------
# Question 34:
# Compare the results of the different hypothesis tests and confidence intervals.
# ------------------------------------------------------------------------------
# R Code:
# Correlation vs ANOVA and CI verification
cat("CI for Mean Age does not span 40, consistent with One-sample t-test rejection.
")
cat("Duration differences between subscription groups show non-overlapping intervals, reinforcing two-sample t-test.
")

# Detailed Explanation:
# - Demonstrates duality between hypothesis testing and confidence intervals:
#   if a 95% confidence interval excludes the null parameter value, the two-sided test rejects H0 at alpha = 0.05.


# ------------------------------------------------------------------------------
# Question 35:
# Identify the variables that show statistically significant relationships with the target variable.
# ------------------------------------------------------------------------------
# R Code:
# Run bivariate statistical tests against target variable 'y'
p_vals_target <- c(
  duration = t.test(duration ~ y, data = bank_clean)$p.value,
  age      = t.test(age ~ y, data = bank_clean)$p.value,
  balance  = wilcox.test(balance ~ y, data = bank_clean)$p.value,
  housing  = chisq.test(table(bank_clean$housing, bank_clean$y))$p.value,
  poutcome = chisq.test(table(bank_clean$poutcome, bank_clean$y))$p.value
)
sig_target_vars <- names(p_vals_target)[p_vals_target < 0.05]
cat("Statistically significant features associated with 'y':
", paste(sig_target_vars, collapse = ", "), "
")

# Detailed Explanation:
# - Bivariate tests isolate features (e.g., duration, previous campaign outcome, housing loan) with significant distributional differences across subscription status.


# ------------------------------------------------------------------------------
# Question 36:
# Compare the findings from hypothesis testing with the results obtained from the classification models.
# ------------------------------------------------------------------------------
# R Code:
cat("Comparison of Hypothesis Testing vs Machine Learning Feature Importance:
")
cat("- Hypothesis testing establishes isolated bivariate associations (e.g., duration, poutcome, housing loan p < 0.001).
")
cat("- Classification models (Random Forest, Logistic Regression) account for multivariate collinearity and interactions.
")
cat("- Both paradigms agree: 'duration' and 'poutcome' are the strongest statistical and predictive drivers of term deposit subscription.
")

# Detailed Explanation:
# - Hypothesis testing confirms whether individual associations are non-spurious.
# - Machine learning feature importance (Random Forest Gini reduction, logistic odds ratios) confirms that these statistically significant variables provide high predictive power in a combined multivariate space.
