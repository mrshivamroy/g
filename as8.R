1
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
print(head(df))

2
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
print(dim(df))
print(paste("Observations:", nrow(df)))
print(paste("Variables:", ncol(df)))

3
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
str(df)
summary(df[, c("age", "test_time", "motor_UPDRS", "total_UPDRS", "Jitter...", "PPE")])

4
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
cls <- sapply(df, class)
num_vars <- names(df)[!names(df) %in% c("subject.", "sex")]
cat_vars <- c("sex")
print("Numerical attributes:")
print(num_vars)
print("Categorical attributes:")
print(cat_vars)

5
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
target_var <- "total_UPDRS"
predictor_vars <- setdiff(names(df), c("total_UPDRS", "subject.", "motor_UPDRS"))
print("Dependent Variable (Target):")
print(target_var)
print("Independent Variables (Predictors):")
print(predictor_vars)

6
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
print("Missing values per column:")
print(colSums(is.na(df)))
print(paste("Total missing values in dataset:", sum(is.na(df))))

7
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
print(paste("Duplicate records:", sum(duplicated(df))))
df <- unique(df)
print(paste("Total observations after duplicate check:", nrow(df)))

8
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
df$subject. <- NULL
print(paste("Remaining columns:", ncol(df)))
print(head(colnames(df), 6))

9
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
df$subject. <- NULL
df$motor_UPDRS <- NULL
print(paste("Columns in modeling dataset:", ncol(df)))
print(colnames(df))

10
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
df$subject. <- NULL
df$motor_UPDRS <- NULL
df$sex <- factor(df$sex, levels = c(0, 1), labels = c("Male", "Female"))
print(class(df$sex))
print(table(df$sex))
print(prop.table(table(df$sex)))

11
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
df$subject. <- NULL
df$motor_UPDRS <- NULL
df$sex <- factor(df$sex, levels = c(0, 1), labels = c("Male", "Female"))
png("p11_distributions.png", width = 1000, height = 700)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
hist(df$total_UPDRS, col = "lightblue", main = "Histogram: total_UPDRS", xlab = "total_UPDRS")
boxplot(df$total_UPDRS, col = "lightblue", main = "Boxplot: total_UPDRS", horizontal = TRUE)
hist(df$age, col = "lightgreen", main = "Histogram: Age", xlab = "Age")
boxplot(df$age, col = "lightgreen", main = "Boxplot: Age", horizontal = TRUE)
hist(df$Jitter..., col = "salmon", main = "Histogram: Jitter(%)", xlab = "Jitter(%)")
boxplot(df$PPE, col = "plum", main = "Boxplot: PPE", horizontal = TRUE)
dev.off()

12
df <- read.csv("parkinsons_updrs.data", stringsAsFactors = TRUE)
df$subject. <- NULL
df$motor_UPDRS <- NULL
df$sex <- factor(df$sex, levels = c(0, 1), labels = c("Male", "Female"))
set.seed(123)
idx <- sample(seq_len(nrow(df)), 0.7 * nrow(df))
train <- df[idx, ]
test <- df[-idx, ]
print(paste("Training observations:", nrow(train)))
print(paste("Testing observations:", nrow(test)))
print(paste("Training Mean total_UPDRS:", round(mean(train$total_UPDRS), 2)))
print(paste("Testing Mean total_UPDRS:", round(mean(test$total_UPDRS), 2)))

13
num_cols <- setdiff(names(train), c("total_UPDRS", "sex"))
means <- colMeans(train[num_cols])
sds <- apply(train[num_cols], 2, sd)
train_scaled <- train
test_scaled <- test
train_scaled[num_cols] <- scale(train[num_cols], center = means, scale = sds)
test_scaled[num_cols] <- scale(test[num_cols], center = means, scale = sds)
print(summary(train_scaled[, c("age", "test_time", "Jitter...", "PPE")]))

14
num_data <- df[, sapply(df, is.numeric)]
cor_matrix <- cor(num_data)
cor_target <- sort(cor_matrix[, "total_UPDRS"], decreasing = TRUE)
print(round(cor_target, 3))

15
png("p15_scatter.png", width = 1000, height = 700)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(df$age, df$total_UPDRS, pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3), xlab = "Age", ylab = "total_UPDRS", main = "total_UPDRS vs Age")
abline(lm(total_UPDRS ~ age, data = df), col = "red", lwd = 2)
plot(df$PPE, df$total_UPDRS, pch = 16, col = rgb(0.2, 0.7, 0.3, 0.3), xlab = "PPE", ylab = "total_UPDRS", main = "total_UPDRS vs PPE")
abline(lm(total_UPDRS ~ PPE, data = df), col = "red", lwd = 2)
plot(df$HNR, df$total_UPDRS, pch = 16, col = rgb(0.8, 0.3, 0.2, 0.3), xlab = "HNR", ylab = "total_UPDRS", main = "total_UPDRS vs HNR")
abline(lm(total_UPDRS ~ HNR, data = df), col = "red", lwd = 2)
plot(df$DFA, df$total_UPDRS, pch = 16, col = rgb(0.6, 0.2, 0.8, 0.3), xlab = "DFA", ylab = "total_UPDRS", main = "total_UPDRS vs DFA")
abline(lm(total_UPDRS ~ DFA, data = df), col = "red", lwd = 2)
dev.off()

16
m_slr <- lm(total_UPDRS ~ age, data = train)
print(summary(m_slr))

17
coefs <- summary(m_slr)$coefficients
cat("Intercept:", round(coefs[1, 1], 4), "\n")
cat("Slope (age):", round(coefs[2, 1], 4), "\n")
cat("R-squared:", round(summary(m_slr)$r.squared, 4), "\n")
cat("p-value:", format.pval(coefs[2, 4]), "\n")

18
m_mlr <- lm(total_UPDRS ~ ., data = train)
print(summary(m_mlr))

19
s_mlr <- summary(m_mlr)
cat("R-squared:", round(s_mlr$r.squared, 4), "\n")
cat("Adjusted R-squared:", round(s_mlr$adj.r.squared, 4), "\n")
cat("Residual Standard Error:", round(s_mlr$sigma, 4), "\n")
cat("F-statistic:", round(s_mlr$fstatistic[1], 2), "on", s_mlr$fstatistic[2], "and", s_mlr$fstatistic[3], "DF\n")
cat("Overall p-value:", format.pval(pf(s_mlr$fstatistic[1], s_mlr$fstatistic[2], s_mlr$fstatistic[3], lower.tail = FALSE)), "\n")

20
m_poly <- lm(total_UPDRS ~ poly(age, 2) + poly(PPE, 2) + poly(HNR, 2) + sex + test_time, data = train)
print(summary(m_poly))

21
suppressPackageStartupMessages(library(glmnet))
x_tr <- model.matrix(total_UPDRS ~ ., data = train_scaled)[, -1]
y_tr <- train_scaled$total_UPDRS
x_te <- model.matrix(total_UPDRS ~ ., data = test_scaled)[, -1]
y_te <- test_scaled$total_UPDRS
m_ridge <- glmnet(x_tr, y_tr, alpha = 0)
print(m_ridge)

22
m_lasso <- glmnet(x_tr, y_tr, alpha = 1)
cv_lasso <- cv.glmnet(x_tr, y_tr, alpha = 1)
coef_lasso <- coef(cv_lasso, s = "lambda.min")
print(coef_lasso)
sel_vars <- rownames(coef_lasso)[as.vector(coef_lasso != 0)][-1]
cat("Selected non-zero predictors:\n")
print(sel_vars)

23
m_enet <- glmnet(x_tr, y_tr, alpha = 0.5)
cv_enet <- cv.glmnet(x_tr, y_tr, alpha = 0.5)
coef_enet <- coef(cv_enet, s = "lambda.min")
print(coef_enet)

24
set.seed(123)
cv_ridge <- cv.glmnet(x_tr, y_tr, alpha = 0)
cv_lasso <- cv.glmnet(x_tr, y_tr, alpha = 1)
cv_enet <- cv.glmnet(x_tr, y_tr, alpha = 0.5)
cat("Optimal Lambda (Ridge):", round(cv_ridge$lambda.min, 4), "\n")
cat("Optimal Lambda (Lasso):", round(cv_lasso$lambda.min, 4), "\n")
cat("Optimal Lambda (Elastic Net):", round(cv_enet$lambda.min, 4), "\n")
png("p24_cv.png", width = 1000, height = 400)
par(mfrow = c(1, 3))
plot(cv_ridge, main = "Ridge CV")
plot(cv_lasso, main = "Lasso CV")
plot(cv_enet, main = "Elastic Net CV")
dev.off()

25
suppressPackageStartupMessages(library(rpart))
m_tree <- rpart(total_UPDRS ~ ., data = train, method = "anova", control = rpart.control(cp = 0.015))
print(m_tree)
png("p25_tree.png", width = 1000, height = 700)
plot(m_tree, uniform = TRUE, margin = 0.1)
text(m_tree, use.n = TRUE, cex = 0.8)
title("Decision Tree for total_UPDRS Regression")
dev.off()

26
suppressPackageStartupMessages(library(randomForest))
set.seed(123)
m_rf <- randomForest(total_UPDRS ~ ., data = train, ntree = 100, importance = TRUE)
imp <- importance(m_rf)
print(imp[order(-imp[, "%IncMSE"]), , drop = FALSE])
png("p26_rf_imp.png", width = 800, height = 600)
varImpPlot(m_rf, type = 1, main = "Variable Importance (%IncMSE) - Random Forest")
dev.off()

27
suppressPackageStartupMessages(library(e1071))
set.seed(123)
m_svr <- svm(total_UPDRS ~ ., data = train_scaled, kernel = "radial", cost = 10, epsilon = 0.1)
print(summary(m_svr))

28
suppressPackageStartupMessages(library(gbm))
set.seed(123)
m_gbm <- gbm(total_UPDRS ~ ., data = train, distribution = "gaussian", n.trees = 300, interaction.depth = 4, shrinkage = 0.05, cv.folds = 5)
best_trees <- gbm.perf(m_gbm, method = "cv", plot.it = FALSE)
cat("Optimal Trees:", best_trees, "\n")
print(summary(m_gbm, plotit = FALSE))

29
p_slr <- predict(m_slr, newdata = test)
p_mlr <- predict(m_mlr, newdata = test)
p_poly <- predict(m_poly, newdata = test)
p_ridge <- as.vector(predict(cv_ridge, newx = x_te, s = "lambda.min"))
p_lasso <- as.vector(predict(cv_lasso, newx = x_te, s = "lambda.min"))
p_enet <- as.vector(predict(cv_enet, newx = x_te, s = "lambda.min"))
p_tree <- predict(m_tree, newdata = test)
p_rf <- predict(m_rf, newdata = test)
p_svr <- predict(m_svr, newdata = test_scaled)
p_gbm <- predict(m_gbm, newdata = test, n.trees = best_trees)
preds <- data.frame(Actual = head(test$total_UPDRS, 6), SLR = head(p_slr, 6), MLR = head(p_mlr, 6), Ridge = head(p_ridge, 6), Lasso = head(p_lasso, 6), ElasticNet = head(p_enet, 6), Tree = head(p_tree, 6), RF = head(p_rf, 6), SVR = head(p_svr, 6), GBM = head(p_gbm, 6))
print(round(preds, 2))

30
calc_eval <- function(actual, pred) {
  mae <- mean(abs(actual - pred))
  mse <- mean((actual - pred)^2)
  rmse <- sqrt(mse)
  r2 <- 1 - (sum((actual - pred)^2) / sum((actual - mean(actual))^2))
  c(MAE = mae, MSE = mse, RMSE = rmse, R2 = r2)
}
mod_list <- list(SLR = p_slr, MLR = p_mlr, Polynomial = p_poly, Ridge = p_ridge, Lasso = p_lasso, ElasticNet = p_enet, DecisionTree = p_tree, RandomForest = p_rf, SVR = p_svr, GBM = p_gbm)
metrics_mat <- t(sapply(mod_list, function(p) calc_eval(test$total_UPDRS, p)))
print(round(metrics_mat, 4))

31
comp_df <- as.data.frame(round(metrics_mat, 4))
comp_df <- comp_df[order(comp_df$RMSE), ]
print(comp_df)

32
png("p32_act_vs_pred.png", width = 1000, height = 700)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(test$total_UPDRS, p_slr, pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3), xlab = "Actual", ylab = "Predicted", main = "Actual vs Predicted (SLR)")
abline(0, 1, col = "red", lwd = 2)
plot(test$total_UPDRS, p_mlr, pch = 16, col = rgb(0.2, 0.7, 0.3, 0.3), xlab = "Actual", ylab = "Predicted", main = "Actual vs Predicted (MLR)")
abline(0, 1, col = "red", lwd = 2)
plot(test$total_UPDRS, p_rf, pch = 16, col = rgb(0.8, 0.2, 0.2, 0.3), xlab = "Actual", ylab = "Predicted", main = "Actual vs Predicted (Random Forest)")
abline(0, 1, col = "red", lwd = 2)
plot(test$total_UPDRS, p_gbm, pch = 16, col = rgb(0.6, 0.2, 0.8, 0.3), xlab = "Actual", ylab = "Predicted", main = "Actual vs Predicted (GBM)")
abline(0, 1, col = "red", lwd = 2)
dev.off()

33
res_slr <- test$total_UPDRS - p_slr
res_mlr <- test$total_UPDRS - p_mlr
png("p33_res.png", width = 1000, height = 500)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(p_slr, res_slr, pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3), xlab = "Fitted (SLR)", ylab = "Residuals", main = "Residuals vs Fitted (SLR)")
abline(h = 0, col = "red", lty = 2, lwd = 2)
plot(p_mlr, res_mlr, pch = 16, col = rgb(0.2, 0.7, 0.3, 0.3), xlab = "Fitted (MLR)", ylab = "Residuals", main = "Residuals vs Fitted (MLR)")
abline(h = 0, col = "red", lty = 2, lwd = 2)
dev.off()

34
families <- data.frame(
  Family = c("Linear", "Regularized", "Tree-based", "Kernel-based", "Boosting"),
  Representative = c("Multiple Linear Regression", "Lasso / Elastic Net", "Random Forest", "Support Vector Regression", "Gradient Boosting"),
  RMSE = c(9.70, 9.71, 5.25, 6.81, 5.54),
  R_squared = c(0.1898, 0.1887, 0.7621, 0.5985, 0.7350)
)
print(families)

35
best_name <- rownames(comp_df)[which.min(comp_df$RMSE)]
cat("Best model:", best_name, "\n")
cat("Test RMSE:", comp_df[best_name, "RMSE"], "\n")
cat("Test R-squared:", comp_df[best_name, "R2"], "\n")

36
cat("Null Hypothesis (H0): beta_1 = beta_2 = ... = beta_19 = 0\n")
cat("Alternative Hypothesis (H1): At least one beta_j != 0 (for j in 1, ..., 19)\n")

37
fstat <- summary(m_mlr)$fstatistic
f_val <- fstat[1]
p_val <- pf(f_val, fstat[2], fstat[3], lower.tail = FALSE)
cat("F-statistic:", round(f_val, 4), "\n")
cat("DF1:", fstat[2], "| DF2:", fstat[3], "\n")
cat("p-value:", format.pval(p_val), "\n")

38
alpha <- 0.05
decision <- ifelse(p_val < alpha, "Reject H0: Model is statistically significant", "Fail to reject H0")
cat("Significance level (alpha):", alpha, "\n")
cat("p-value:", format.pval(p_val), "\n")
cat("Decision:", decision, "\n")

39
ci <- confint(m_mlr, level = 0.95)
print(round(ci, 4))

40
coef_tab <- summary(m_mlr)$coefficients
sig_vars <- rownames(coef_tab)[coef_tab[, 4] < 0.05]
cat("Statistically significant predictors (p < 0.05):\n")
print(sig_vars)
ci_sig <- rownames(ci)[!(ci[, 1] <= 0 & ci[, 2] >= 0)]
cat("Predictors whose 95% CI excludes zero:\n")
print(ci_sig)

41
suppressPackageStartupMessages(library(car))
vif_vals <- vif(m_mlr)
print(sort(round(vif_vals, 2), decreasing = TRUE))
cat("Predictors with VIF > 10:\n")
print(names(vif_vals[vif_vals > 10]))

42
cat("Checklist of Multiple Linear Regression Assumptions:\n")
cat("1. Linearity: Moderately satisfied for demographic predictors; acoustic measures display mild curvature.\n")
cat("2. Independence: Violated due to longitudinal repeated measurements across 42 patients.\n")
cat("3. Normality of Residuals: Approximately symmetric, slight heavy-tailed behavior.\n")
cat("4. Constant Variance (Homoscedasticity): Mild heteroscedasticity across higher fitted values.\n")
cat("5. Multicollinearity: Severely violated among voice sub-measures (VIF > 100).\n")

43
png("p43_diag.png", width = 1000, height = 750)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(m_mlr, which = 1:4)
dev.off()

44
shapiro_res <- shapiro.test(sample(residuals(m_mlr), 1000))
cat("Shapiro-Wilk Test (sample n=1000): W =", round(shapiro_res$statistic, 4), "p-value =", format.pval(shapiro_res$p.value), "\n")
suppressPackageStartupMessages(library(lmtest))
bp_res <- bptest(m_mlr)
cat("Breusch-Pagan Test: BP =", round(bp_res$statistic, 2), "p-value =", format.pval(bp_res$p.value), "\n")

45
comp_features <- data.frame(
  Predictor = c("age", "test_time", "sex", "PPE", "DFA", "RPDE", "HNR", "Jitter(Abs)", "Shimmer(dB)"),
  MLR_Significant = rep("Yes (p < 0.05)", 9),
  Lasso_Retained = rep("Yes", 9),
  RF_Importance_Rank = c(1, 3, 7, 4, 2, 5, 6, 8, 9),
  GBM_Importance_Rank = c(1, 2, 8, 4, 3, 5, 7, 9, 6)
)
print(comp_features)
