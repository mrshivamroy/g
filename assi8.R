# ==============================================================================
# ASSIGNMENT 8: REGRESSION ANALYSIS ON PARKINSON'S TELEMONITORING DATASET
# Target Variable: total_UPDRS
# ==============================================================================

# Required Libraries
# install.packages(c("readr", "dplyr", "ggplot2", "caret", "glmnet", 
#                    "rpart", "rpart.plot", "randomForest", "e1071", 
#                    "gbm", "car"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(gbm)
library(car)

# Set seed for reproducible results across all splits and stochastic algorithms
set.seed(42)

# ==============================================================================
# PART A: DATA PREPARATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 1:
# Import the Parkinson’s Telemonitoring dataset into R and display the first six observations.
# ------------------------------------------------------------------------------
# R Code:
parkinsons_raw <- read.csv("parkinsons_updrs.data")
head(parkinsons_raw, 6)

# Detailed Explanation:
# - read.csv() imports comma-separated tabular data from disk into an R data frame.
# - head(parkinsons_raw, 6) displays the first six records along with all column headers,
#   allowing rapid manual verification of successful file import and data structure integrity.


# ------------------------------------------------------------------------------
# Question 2:
# Determine the number of observations and variables in the dataset.
# ------------------------------------------------------------------------------
# R Code:
dim(parkinsons_raw)
cat("Number of observations (rows):", nrow(parkinsons_raw), "\n")
cat("Number of variables (columns):", ncol(parkinsons_raw), "\n")

# Detailed Explanation:
# - dim() returns a 2-element integer vector: [rows, columns].
# - nrow() yields the sample size (typically 5,875 recordings across 42 patients).
# - ncol() gives the number of features recorded (22 attributes).


# ------------------------------------------------------------------------------
# Question 3:
# Display the structure, data types, and summary statistics of the dataset.
# ------------------------------------------------------------------------------
# R Code:
str(parkinsons_raw)
summary(parkinsons_raw)

# Detailed Explanation:
# - str() compactly displays the internal data types (numeric, integer) and initial values for each column.
# - summary() computes 5-number summaries (Min, 1st Qu, Median, Mean, 3rd Qu, Max) for continuous features
#   and counts for categorical columns, exposing scales, skewness, and potential extreme outliers.


# ------------------------------------------------------------------------------
# Question 4:
# Identify the numerical and categorical attributes.
# ------------------------------------------------------------------------------
# R Code:
sapply(parkinsons_raw, class)

# Detailed Explanation:
# - Categorical attribute: 'sex' (binary indicator coded as 0 or 1).
# - Identification attribute: 'subject.' (discrete integer participant index).
# - All remaining 20 features (age, test_time, motor_UPDRS, total_UPDRS, jitter measures,
#   shimmer measures, NHR, HNR, RPDE, DFA, PPE) are continuous numerical measurements.


# ------------------------------------------------------------------------------
# Question 5:
# Identify the dependent variable and independent variables for predicting total_UPDRS.
# ------------------------------------------------------------------------------
# R Code:
dep_var <- "total_UPDRS"
indep_vars <- setdiff(names(parkinsons_raw), c("subject.", "motor_UPDRS", "total_UPDRS"))
print(dep_var)
print(indep_vars)

# Detailed Explanation:
# - Dependent variable: 'total_UPDRS' (continuous clinical score representing disease burden).
# - Independent variables: demographic measures ('age', 'sex', 'test_time') and acoustic voice measures
#   ('Jitter...', 'Shimmer...', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE').
# - 'subject.' and 'motor_UPDRS' are strictly excluded from the predictor space.


# ------------------------------------------------------------------------------
# Question 6:
# Check for missing values and describe how they should be handled.
# ------------------------------------------------------------------------------
# R Code:
sum(is.na(parkinsons_raw))
colSums(is.na(parkinsons_raw))

# Detailed Explanation:
# - sum(is.na()) sums all missing cell occurrences across the entire dataset.
# - The UCI Parkinson’s Telemonitoring dataset contains 0 missing entries.
# - If missing data were present, imputation methods (such as median imputation for skewed continuous
#   voice features or k-Nearest Neighbors) would be preferable over complete case deletion to preserve statistical power.


# ------------------------------------------------------------------------------
# Question 7:
# Check for duplicate observations and remove them if necessary.
# ------------------------------------------------------------------------------
# R Code:
sum(duplicated(parkinsons_raw))
parkinsons_clean <- parkinsons_raw[!duplicated(parkinsons_raw), ]

# Detailed Explanation:
# - duplicated() flags rows that are exact duplicates of previous rows.
# - Subsetting using [!duplicated(...), ] eliminates redundancy, preventing duplicated telemetry recordings
#   from artificially deflating error estimates.


# ------------------------------------------------------------------------------
# Question 8:
# Remove the subject# identifier variable and explain why it should not be used as a predictor.
# ------------------------------------------------------------------------------
# R Code:
# The column name is typically 'subject.' in R's read.csv import
parkinsons_clean <- parkinsons_clean[, !names(parkinsons_clean) %in% c("subject.", "subject#")]

# Detailed Explanation:
# - 'subject#' is an arbitrary administrative index assigned to individuals.
# - Including an ID as a numerical predictor introduces spurious correlations; treating it as a factor
#   causes high cardinality overfitting that fails completely on unseen subjects (cannot generalize).


# ------------------------------------------------------------------------------
# Question 9:
# Exclude motor_UPDRS from the predictor set and explain how its inclusion may cause target leakage.
# ------------------------------------------------------------------------------
# R Code:
parkinsons_model_df <- parkinsons_clean[, !names(parkinsons_clean) %in% "motor_UPDRS"]

# Detailed Explanation:
# - In clinical practice, total_UPDRS is directly derived as the sum of motor_UPDRS and non-motor/mental scores.
# - Keeping motor_UPDRS causes severe data/target leakage: models trivially predict total_UPDRS from its
#   constituent component rather than learning the underlying voice telemonitoring signal, ruining real-world utility.


# ------------------------------------------------------------------------------
# Question 10:
# Convert categorical variables, such as sex, into suitable data types for regression modelling.
# ------------------------------------------------------------------------------
# R Code:
parkinsons_model_df$sex <- as.factor(parkinsons_model_df$sex)
levels(parkinsons_model_df$sex) <- c("Female", "Male")

# Detailed Explanation:
# - as.factor() ensures R's regression engine (lm, glmnet) treats 'sex' as a nominal indicator,
#   automatically expanding it into a single dummy variable (0 vs 1) rather than assuming a continuous slope.


# ------------------------------------------------------------------------------
# Question 11:
# Examine the distributions of the target variable and important numerical predictors using histograms and boxplots.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(2, 2))
hist(parkinsons_model_df$total_UPDRS, main = "Histogram of total_UPDRS", xlab = "total_UPDRS", col = "skyblue")
boxplot(parkinsons_model_df$total_UPDRS, main = "Boxplot of total_UPDRS", col = "skyblue")
hist(parkinsons_model_df$age, main = "Histogram of Age", xlab = "Age", col = "salmon")
boxplot(parkinsons_model_df$PPE, main = "Boxplot of PPE", col = "salmon")
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Histograms evaluate unimodality, symmetry, and normal distribution assumptions.
# - Boxplots reveal quartiles, interquartile ranges (IQR), and presence of outliers (e.g., in voice disturbance metrics like PPE and Jitter).


# ------------------------------------------------------------------------------
# Question 12:
# Divide the dataset into training and testing sets.
# ------------------------------------------------------------------------------
# R Code:
set.seed(42)
train_idx <- createDataPartition(parkinsons_model_df$total_UPDRS, p = 0.8, list = FALSE)
train_df <- parkinsons_model_df[train_idx, ]
test_df  <- parkinsons_model_df[-train_idx, ]

# Detailed Explanation:
# - caret::createDataPartition creates a stratified 80/20 train/test split preserving the continuous
#   distribution quantiles of total_UPDRS across both sets.
# - An independent test split ensures an honest assessment of generalization error without data snooping.


# ------------------------------------------------------------------------------
# Question 13:
# Apply feature scaling or normalization where required for regression models.
# ------------------------------------------------------------------------------
# R Code:
scale_params <- preProcess(train_df[, -which(names(train_df) %in% c("total_UPDRS", "sex"))], method = c("center", "scale"))
train_scaled <- predict(scale_params, train_df)
test_scaled  <- predict(scale_params, test_df)

# Matrix format for glmnet (Ridge/Lasso/Elastic Net)
x_train <- model.matrix(total_UPDRS ~ ., data = train_scaled)[, -1]
y_train <- train_scaled$total_UPDRS
x_test  <- model.matrix(total_UPDRS ~ ., data = test_scaled)[, -1]
y_test  <- test_scaled$total_UPDRS

# Detailed Explanation:
# - Standardization (mean = 0, standard deviation = 1) prevents scale-sensitive algorithms (Ridge, Lasso,
#   Elastic Net, SVR) from penalizing or weighting features arbitrarily based on raw measurement units.
# - Critically, preProcess parameters are fitted ONLY on train_df and applied to test_df to prevent information leakage.


# ==============================================================================
# PART B: REGRESSION ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 14:
# Generate a correlation matrix and identify predictors that are strongly correlated with total_UPDRS.
# ------------------------------------------------------------------------------
# R Code:
num_cols <- sapply(train_df, is.numeric)
cor_mat <- cor(train_df[, num_cols])
cor_target <- sort(cor_mat[, "total_UPDRS"], decreasing = TRUE)
print(round(cor_target, 3))

# Detailed Explanation:
# - cor() computes pairwise Pearson correlation coefficients.
# - Sorting correlations with total_UPDRS highlights predictors with strongest linear association (e.g., age, PPE, RPDE, HNR).


# ------------------------------------------------------------------------------
# Question 15:
# Create scatter plots to examine the relationships between total_UPDRS and selected numerical predictors.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(1, 3))
plot(train_df$age, train_df$total_UPDRS, xlab = "Age", ylab = "total_UPDRS", main = "total_UPDRS vs Age", col = "darkblue", pch = 19)
plot(train_df$PPE, train_df$total_UPDRS, xlab = "PPE", ylab = "total_UPDRS", main = "total_UPDRS vs PPE", col = "darkgreen", pch = 19)
plot(train_df$HNR, train_df$total_UPDRS, xlab = "HNR", ylab = "total_UPDRS", main = "total_UPDRS vs HNR", col = "darkred", pch = 19)
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Scatter plots visualize bivariate associations, revealing whether relationships are strictly linear,
#   heteroscedastic, or clustered, and detecting non-linear curvatures.


# ------------------------------------------------------------------------------
# Question 16:
# Develop a Simple Linear Regression model using one suitable predictor.
# ------------------------------------------------------------------------------
# R Code:
# 'age' is chosen as it exhibits one of the strongest monotonic relationships with UPDRS
slr_model <- lm(total_UPDRS ~ age, data = train_df)
summary(slr_model)

# Detailed Explanation:
# - lm() fits an ordinary least squares (OLS) line minimizing the sum of squared residuals: y = beta_0 + beta_1 * age + e.


# ------------------------------------------------------------------------------
# Question 17:
# Interpret the regression equation, coefficient, intercept, R-squared value, and p-value of the Simple Linear Regression model.
# ------------------------------------------------------------------------------
# R Code:
coef(slr_model)
summary(slr_model)$r.squared
summary(slr_model)$coefficients["age", "Pr(>|t|)"]

# Detailed Explanation:
# - Intercept (beta_0): The expected baseline total_UPDRS score when age = 0 (an extrapolation benchmark).
# - Slope (beta_1): The average change in total_UPDRS score for each additional year of patient age.
# - R-squared: Proportion of total variance in total_UPDRS explained by age alone.
# - p-value (< 0.05): Establishes statistical significance, rejecting the null hypothesis that age has no linear association with UPDRS.


# ------------------------------------------------------------------------------
# Question 18:
# Develop a Multiple Linear Regression model using the selected independent variables.
# ------------------------------------------------------------------------------
# R Code:
mlr_model <- lm(total_UPDRS ~ ., data = train_df)
summary(mlr_model)

# Detailed Explanation:
# - Fits an OLS model across all clean demographic and dysphonia speech features simultaneously.


# ------------------------------------------------------------------------------
# Question 19:
# Interpret the coefficients, R-squared, adjusted R-squared, residual standard error, and overall significance of the Multiple Linear Regression model.
# ------------------------------------------------------------------------------
# R Code:
summary(mlr_model)$r.squared
summary(mlr_model)$adj.r.squared
summary(mlr_model)$sigma
anova(mlr_model)

# Detailed Explanation:
# - Coefficients: Partial regression slopes; change in total_UPDRS per 1-unit increase in a feature, holding all other features constant.
# - R-squared: Total proportion of variance explained by all predictors combined.
# - Adjusted R-squared: Variance explained penalized for the number of predictors, avoiding artificial inflation.
# - Residual Standard Error (sigma): Standard deviation of the unexplained residuals in UPDRS units.
# - Overall F-statistic p-value: Confirms whether the full set of predictors provides statistically significant explanatory power over an intercept-only model.


# ------------------------------------------------------------------------------
# Question 20:
# Develop a Polynomial Regression model to investigate nonlinear relationships between selected predictors and total_UPDRS.
# ------------------------------------------------------------------------------
# R Code:
poly_model <- lm(total_UPDRS ~ poly(age, 2) + poly(PPE, 2) + sex, data = train_df)
summary(poly_model)

# Detailed Explanation:
# - poly(..., 2) generates orthogonal 1st and 2nd degree polynomial terms for age and pitch period entropy (PPE).
# - This captures parabolic / non-linear saturation effects without numerical collinearity problems.


# ------------------------------------------------------------------------------
# Question 21:
# Develop a Ridge Regression model using L2 regularization.
# ------------------------------------------------------------------------------
# R Code:
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Detailed Explanation:
# - alpha = 0 specifies L2 Ridge regularization, which minimizes SSE + lambda * sum(beta_j^2).
# - Shrinks coefficients toward zero to stabilize variance and mitigate multicollinearity without dropping features.


# ------------------------------------------------------------------------------
# Question 22:
# Develop a Lasso Regression model using L1 regularization and identify the predictors selected by the model.
# ------------------------------------------------------------------------------
# R Code:
lasso_model <- glmnet(x_train, y_train, alpha = 1)
coef(lasso_model, s = 0.1)

# Detailed Explanation:
# - alpha = 1 enforces L1 penalty (lambda * sum(|beta_j|)).
# - L1 regularization produces sparse models by driving uninformative coefficients exactly to zero, performing intrinsic feature selection.


# ------------------------------------------------------------------------------
# Question 23:
# Develop an Elastic Net Regression model by combining L1 and L2 regularization.
# ------------------------------------------------------------------------------
# R Code:
elastic_model <- glmnet(x_train, y_train, alpha = 0.5)

# Detailed Explanation:
# - alpha = 0.5 balances L1 sparsity and L2 grouping properties, handling correlated voice acoustic measures more robustly than pure Lasso.


# ------------------------------------------------------------------------------
# Question 24:
# Determine suitable regularization parameters for Ridge, Lasso, and Elastic Net using cross-validation.
# ------------------------------------------------------------------------------
# R Code:
cv_ridge   <- cv.glmnet(x_train, y_train, alpha = 0)
cv_lasso   <- cv.glmnet(x_train, y_train, alpha = 1)
cv_elastic <- cv.glmnet(x_train, y_train, alpha = 0.5)

best_lambda_ridge   <- cv_ridge$lambda.min
best_lambda_lasso   <- cv_lasso$lambda.min
best_lambda_elastic <- cv_elastic$lambda.min
print(c(Ridge = best_lambda_ridge, Lasso = best_lambda_lasso, Elastic = best_lambda_elastic))

# Detailed Explanation:
# - 10-fold cross-validation estimates out-of-sample mean squared error across candidate lambda grids.
# - lambda.min identifies the penalty strength that minimizes cross-validated test loss.


# ------------------------------------------------------------------------------
# Question 25:
# Develop a Decision Tree Regression model and visualize the resulting tree.
# ------------------------------------------------------------------------------
# R Code:
tree_model <- rpart(total_UPDRS ~ ., data = train_df, method = "anova", control = rpart.control(cp = 0.01))
rpart.plot(tree_model, main = "Decision Tree Regression for total_UPDRS")

# Detailed Explanation:
# - rpart() recursively partitions predictor space into homogeneous leaves using ANOVA variance reduction.
# - rpart.plot() graphs the hierarchical decision splits and mean target values per terminal node.


# ------------------------------------------------------------------------------
# Question 26:
# Develop a Random Forest Regression model and identify the important predictor variables.
# ------------------------------------------------------------------------------
# R Code:
rf_model <- randomForest(total_UPDRS ~ ., data = train_df, ntree = 200, importance = TRUE)
varImpPlot(rf_model, main = "Variable Importance: Random Forest")

# Detailed Explanation:
# - Combines an ensemble of de-correlated bagged trees trained on bootstrap samples and random feature subsets.
# - varImpPlot displays %IncMSE (increase in MSE when feature values are permuted) and IncNodePurity (total decrease in node impurity).


# ------------------------------------------------------------------------------
# Question 27:
# Develop a Support Vector Regression model using a suitable kernel.
# ------------------------------------------------------------------------------
# R Code:
svr_model <- svm(total_UPDRS ~ ., data = train_scaled, kernel = "radial", cost = 10, gamma = 0.1)

# Detailed Explanation:
# - Radial Basis Function (RBF) kernel projects input space into infinite-dimensional Hilbert space.
# - Finds a hyperplane within an epsilon-insensitive tube, penalizing errors exceeding epsilon with cost C.


# ------------------------------------------------------------------------------
# Question 28:
# Develop a Gradient Boosting Regression model and examine its predictive performance.
# ------------------------------------------------------------------------------
# R Code:
gbm_train <- train_df
gbm_train$sex <- as.numeric(gbm_train$sex) - 1
gbm_model <- gbm(total_UPDRS ~ ., data = gbm_train, distribution = "gaussian", 
                 n.trees = 500, interaction.depth = 4, shrinkage = 0.05, cv.folds = 5)
best_trees <- gbm.perf(gbm_model, method = "cv")

# Detailed Explanation:
# - Sequentially trains shallow trees where each successive tree fits the negative gradient (pseudo-residuals) of the previous ensemble.
# - gbm.perf() uses 5-fold CV to determine the optimal number of iterations, preventing overfitting.


# ------------------------------------------------------------------------------
# Question 29:
# Generate predictions on the test dataset using all developed regression models.
# ------------------------------------------------------------------------------
# R Code:
pred_slr     <- predict(slr_model, newdata = test_df)
pred_mlr     <- predict(mlr_model, newdata = test_df)
pred_poly    <- predict(poly_model, newdata = test_df)
pred_ridge   <- as.vector(predict(cv_ridge, newx = x_test, s = "lambda.min"))
pred_lasso   <- as.vector(predict(cv_lasso, newx = x_test, s = "lambda.min"))
pred_elastic <- as.vector(predict(cv_elastic, newx = x_test, s = "lambda.min"))
pred_tree    <- predict(tree_model, newdata = test_df)
pred_rf      <- predict(rf_model, newdata = test_df)
pred_svr     <- predict(svr_model, newdata = test_scaled)

gbm_test <- test_df
gbm_test$sex <- as.numeric(gbm_test$sex) - 1
pred_gbm     <- predict(gbm_model, newdata = gbm_test, n.trees = best_trees)

# Detailed Explanation:
# - Uses predict() to evaluate models against the untouched test set (test_df / test_scaled / x_test).


# ------------------------------------------------------------------------------
# Question 30:
# Evaluate the models using MAE, MSE, RMSE, and R-squared.
# ------------------------------------------------------------------------------
# R Code:
calc_metrics <- function(actual, pred) {
  mae  <- mean(abs(actual - pred))
  mse  <- mean((actual - pred)^2)
  rmse <- sqrt(mse)
  r2   <- 1 - (sum((actual - pred)^2) / sum((actual - mean(actual))^2))
  return(c(MAE = mae, MSE = mse, RMSE = rmse, R2 = r2))
}

# Detailed Explanation:
# - MAE: Average absolute deviation; robust to extreme outliers.
# - MSE & RMSE: Penalize larger prediction discrepancies quadratically (RMSE is in original scale units).
# - R2: Out-of-sample coefficient of determination measuring variance captured by model predictions.


# ------------------------------------------------------------------------------
# Question 31:
# Prepare a comparison table containing the performance measures of all regression models.
# ------------------------------------------------------------------------------
# R Code:
models_list <- list(
  "Simple Linear Regression"    = pred_slr,
  "Multiple Linear Regression"  = pred_mlr,
  "Polynomial Regression"       = pred_poly,
  "Ridge Regression"            = pred_ridge,
  "Lasso Regression"            = pred_lasso,
  "Elastic Net Regression"      = pred_elastic,
  "Decision Tree"               = pred_tree,
  "Random Forest"               = pred_rf,
  "Support Vector Regression"   = pred_svr,
  "Gradient Boosting"           = pred_gbm
)

results_table <- do.call(rbind, lapply(models_list, function(p) calc_metrics(test_df$total_UPDRS, p)))
results_df <- as.data.frame(results_table)
print(round(results_df, 4))

# Detailed Explanation:
# - Compiles all evaluation metrics across every candidate architecture into a structured comparative data frame.


# ------------------------------------------------------------------------------
# Question 32:
# Plot actual versus predicted total_UPDRS values for the developed models.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(1, 2))
plot(test_df$total_UPDRS, pred_mlr, xlab = "Actual total_UPDRS", ylab = "Predicted total_UPDRS",
     main = "MLR: Actual vs Predicted", col = rgb(0.2, 0.4, 0.8, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(test_df$total_UPDRS, pred_rf, xlab = "Actual total_UPDRS", ylab = "Predicted total_UPDRS",
     main = "Random Forest: Actual vs Predicted", col = rgb(0.1, 0.7, 0.3, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Points tightly adhering to the 45-degree reference line (y = x) indicate high predictive accuracy.


# ------------------------------------------------------------------------------
# Question 33:
# Generate residual plots for the Simple Linear Regression and Multiple Linear Regression models.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(1, 2))
plot(fitted(slr_model), resid(slr_model), xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted (SLR)", col = "darkblue", pch = 16)
abline(h = 0, col = "red", lty = 2)

plot(fitted(mlr_model), resid(mlr_model), xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted (MLR)", col = "darkgreen", pch = 16)
abline(h = 0, col = "red", lty = 2)
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Assesses homoscedasticity and linearity. An ideal residual plot displays an unstructured horizontal band around 0.


# ------------------------------------------------------------------------------
# Question 34:
# Compare the performance of linear, regularized, tree-based, kernel-based, and boosting regression models.
# ------------------------------------------------------------------------------
# R Code:
summary_comparison <- results_df[order(results_df$RMSE), ]
print(round(summary_comparison, 4))

# Detailed Explanation:
# - Ensemble tree-based (Random Forest, Gradient Boosting) and non-linear kernel models (SVR) generally
#   outperform standard OLS and linear regularized models due to their ability to capture complex non-linear
#   interactions inherent in high-dimensional dysphonia speech metrics.


# ------------------------------------------------------------------------------
# Question 35:
# Identify the most suitable model based on the test-set performance and justify the selection.
# ------------------------------------------------------------------------------
# R Code:
best_model_name <- rownames(results_df)[which.min(results_df$RMSE)]
cat("Best Model Selected based on Lowest Test RMSE:", best_model_name, "\n")

# Detailed Explanation:
# - Selection is justified by minimizing out-of-sample RMSE and MAE while maximizing out-of-sample R^2,
#   demonstrating superior generalization capability without overfitting training noise.


# ==============================================================================
# PART C: STATISTICAL ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 36:
# Formulate the null and alternative hypotheses for testing the overall significance of the Multiple Linear Regression model.
# ------------------------------------------------------------------------------
# R Code:
# (Hypothesis statement representation in R output)
cat("H0: beta_1 = beta_2 = ... = beta_k = 0 (No predictor has a linear relationship with total_UPDRS)\n")
cat("H1: At least one beta_j != 0 (At least one predictor has a significant linear relationship)\n")

# Detailed Explanation:
# - Null Hypothesis (H0): None of the included independent variables explain variation in total_UPDRS.
# - Alternative Hypothesis (H1): At least one regression coefficient is non-zero.


# ------------------------------------------------------------------------------
# Question 37:
# Perform the overall F-test and interpret the F-statistic and p-value.
# ------------------------------------------------------------------------------
# R Code:
mlr_fstat <- summary(mlr_model)$fstatistic
p_val_f <- pf(mlr_fstat[1], mlr_fstat[2], mlr_fstat[3], lower.tail = FALSE)
cat("F-statistic:", mlr_fstat[1], "on", mlr_fstat[2], "and", mlr_fstat[3], "DF, p-value:", p_val_f, "\n")

# Detailed Explanation:
# - pf() calculates the upper-tail probability under the F-distribution.
# - A large F-statistic with an infinitesimally small p-value (< 2.2e-16) confirms that the regression model
#   explains a statistically significant portion of total variance compared to an intercept-only model.


# ------------------------------------------------------------------------------
# Question 38:
# Determine whether the overall regression model is statistically significant at the 5% significance level.
# ------------------------------------------------------------------------------
# R Code:
alpha_level <- 0.05
is_significant <- p_val_f < alpha_level
cat("Is model significant at 5% level (alpha = 0.05)?:", is_significant, "\n")

# Detailed Explanation:
# - Since the p-value is substantially less than alpha = 0.05, we reject the null hypothesis H0.
# - The model provides a statistically significant fit to the clinical data.


# ------------------------------------------------------------------------------
# Question 39:
# Calculate and interpret the 95% confidence intervals for the regression coefficients.
# ------------------------------------------------------------------------------
# R Code:
confint(mlr_model, level = 0.95)

# Detailed Explanation:
# - confint() yields lower and upper 95% bounds for each regression slope.
# - If an interval excludes zero, the predictor is statistically significant at alpha = 0.05.
# - If an interval straddles zero, the feature cannot be distinguished from having zero effect holding others constant.


# ------------------------------------------------------------------------------
# Question 40:
# Identify statistically significant predictors using coefficient p-values and confidence intervals.
# ------------------------------------------------------------------------------
# R Code:
coef_summary <- summary(mlr_model)$coefficients
sig_predictors <- coef_summary[coef_summary[, "Pr(>|t|)"] < 0.05, ]
print(sig_predictors)

# Detailed Explanation:
# - Features with t-statistic p-values < 0.05 are statistically significant contributors to total_UPDRS.
#   These are confirmed by confidence intervals strictly omitting zero.


# ------------------------------------------------------------------------------
# Question 41:
# Calculate the Variance Inflation Factor (VIF) and identify the presence of multicollinearity.
# ------------------------------------------------------------------------------
# R Code:
vif_vals <- car::vif(mlr_model)
print(round(vif_vals, 2))
cat("Features with severe multicollinearity (VIF > 10):\n")
print(names(vif_vals[vif_vals > 10]))

# Detailed Explanation:
# - VIF gauges how much the variance of an estimated regression coefficient is inflated by collinearity.
# - VIF > 5 or 10 denotes severe collinearity (common among voice perturbations like variants of Jitter and Shimmer).


# ------------------------------------------------------------------------------
# Question 42:
# Examine the assumptions of Multiple Linear Regression, including linearity, independence, normality of residuals, constant variance, and multicollinearity.
# ------------------------------------------------------------------------------
# R Code:
# Summary checks for standard OLS assumptions
shapiro_sample <- shapiro.test(sample(resid(mlr_model), 1000)) # Subsample for normality test
durbin_watson  <- car::dwt(mlr_model)                          # Independence of residuals
ncv_test       <- car::ncvTest(mlr_model)                      # Non-constant variance (homoscedasticity)

print(durbin_watson)
print(ncv_test)
print(shapiro_sample)

# Detailed Explanation:
# - Linearity: Checked via component-plus-residual and residual-fitted plots.
# - Independence: Durbin-Watson statistic close to 2 indicates no first-order autocorrelation.
# - Normality: Shapiro-Wilk or Q-Q plot checks Gaussian residual distribution.
# - Homoscedasticity: Breusch-Pagan / ncvTest tests for constant error variance.
# - Multicollinearity: Evaluated via VIF (from Q41).


# ------------------------------------------------------------------------------
# Question 43:
# Use residual-versus-fitted, Normal Q-Q, scale-location, and leverage plots to diagnose the regression model.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(2, 2))
plot(mlr_model)
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Residuals vs Fitted: Diagnoses non-linear patterns and non-zero mean errors.
# - Normal Q-Q: Assesses residual normality; deviations at tails indicate skew or heavy tails.
# - Scale-Location: Tests homoscedasticity; a flat horizontal trend indicates constant variance.
# - Residuals vs Leverage: Identifies influential points using Cook's distance contours.


# ------------------------------------------------------------------------------
# Question 44:
# Assess whether the residuals are approximately normally distributed and whether their variance is constant.
# ------------------------------------------------------------------------------
# R Code:
par(mfrow = c(1, 2))
hist(resid(mlr_model), breaks = 30, col = "lightblue", main = "Histogram of Residuals", xlab = "Residuals")
car::ncvTest(mlr_model)
par(mfrow = c(1, 1))

# Detailed Explanation:
# - The residual histogram and Q-Q plots reveal whether residuals are approximately bell-shaped or leptokurtic.
# - A significant ncvTest (p < 0.05) indicates heteroscedasticity, meaning error dispersion varies across predicted UPDRS values.


# ------------------------------------------------------------------------------
# Question 45:
# Compare statistically significant predictors from Multiple Linear Regression with feature importance results from Lasso, Random Forest, and Gradient Boosting.
# ------------------------------------------------------------------------------
# R Code:
# MLR significant features
mlr_sig_names <- rownames(sig_predictors)[-1] # drop intercept

# Lasso non-zero features
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
lasso_sig_names <- rownames(lasso_coefs)[which(lasso_coefs != 0)][-1]

# Random Forest top 5 features by %IncMSE
rf_imp <- importance(rf_model)
rf_top5 <- rownames(rf_imp[order(rf_imp[, "%IncMSE"], decreasing = TRUE), ])[1:5]

# GBM relative influence top 5 features
gbm_summary <- summary(gbm_model, plotit = FALSE)
gbm_top5 <- as.character(gbm_summary$var[1:5])

cat("MLR Significant Predictors:\n", paste(mlr_sig_names, collapse = ", "), "\n\n")
cat("Lasso Selected Predictors:\n", paste(lasso_sig_names, collapse = ", "), "\n\n")
cat("Random Forest Top 5 Features:\n", paste(rf_top5, collapse = ", "), "\n\n")
cat("Gradient Boosting Top 5 Features:\n", paste(gbm_top5, collapse = ", "), "\n")

# Detailed Explanation:
# - Linear models (MLR, Lasso) isolate linear predictors while penalizing redundant collinear features.
# - Non-linear ensembles (Random Forest, GBM) frequently prioritize voice features like PPE, DFA, and RPDE
#   due to complex non-linear splits and high-order interaction effects that linear models cannot capture.
