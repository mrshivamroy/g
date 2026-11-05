Q1 <- {
  file_path <- "bank.csv"
  first_line <- readLines(file_path, n = 1)
  sep_char <- if (grepl(";", first_line)) ";" else ","
  bank_df <- read.csv(file_path, sep = sep_char, stringsAsFactors = FALSE)
  if ("deposit" %in% names(bank_df) && !"y" %in% names(bank_df)) {
    names(bank_df)[names(bank_df) == "deposit"] <- "y"
  }
  head(bank_df)
}

Q2 <- {
  num_obs <- nrow(bank_df)
  num_vars <- ncol(bank_df)
  cat("Number of Observations (Rows):", num_obs, "\n")
  cat("Number of Variables (Columns):", num_vars, "\n")
}

Q3 <- {
  str(bank_df)
  summary(bank_df)
}

Q4 <- {
  predictor_cols <- setdiff(names(bank_df), "y")
  num_cols <- predictor_cols[sapply(bank_df[predictor_cols], is.numeric)]
  cat_cols <- predictor_cols[sapply(bank_df[predictor_cols], function(x) is.character(x) || is.factor(x))]
  cat("Numerical Attributes (", length(num_cols), "):", paste(num_cols, collapse = ", "), "\n\n")
  cat("Categorical Attributes (", length(cat_cols), "):", paste(cat_cols, collapse = ", "), "\n")
}

Q5 <- {
  cat("\nStandard NA count per column:\n")
  print(colSums(is.na(bank_df)))
  for (col in cat_cols) {
    if (is.character(bank_df[[col]]) || is.factor(bank_df[[col]])) {
      bank_df[[col]][as.character(bank_df[[col]]) == "unknown"] <- NA
    }
  }
  get_mode <- function(v) {
    uniqv <- unique(na.omit(v))
    if (length(uniqv) == 0) return(NA)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  for (col in cat_cols) {
    if (sum(is.na(bank_df[[col]])) > 0) {
      bank_df[[col]][is.na(bank_df[[col]])] <- get_mode(bank_df[[col]])
    }
  }
  cat("\nRemaining NA count after imputation:", sum(is.na(bank_df)), "\n")
}

Q6 <- {
  num_duplicates <- sum(duplicated(bank_df))
  cat("Number of duplicate records found:", num_duplicates, "\n")
  if (num_duplicates > 0) {
    bank_df <- bank_df[!duplicated(bank_df), ]
    cat("Duplicates removed. Updated dimensions:", dim(bank_df), "\n")
  } else {
    cat("No duplicate rows detected.\n")
  }
}

Q7 <- {
  bank_df[cat_cols] <- lapply(bank_df[cat_cols], as.factor)
}

Q8 <- {
  min_max_scale <- function(x) {
    if (max(x) == min(x)) return(x)
    (x - min(x)) / (max(x) - min(x))
  }
  bank_scaled <- bank_df
  bank_scaled[num_cols] <- lapply(bank_df[num_cols], min_max_scale)
}

Q9 <- {
  bank_scaled$y <- factor(as.character(bank_scaled$y), levels = c("no", "yes"))
  cat("\nTarget Variable ('y') Distribution:\n")
  print(table(bank_scaled$y))
}

Q10 <- {
  required_packages <- c("caret", "rpart", "rpart.plot", "e1071", "randomForest", "pROC")
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
    }
  }
  library(caret)
  library(rpart)
  library(rpart.plot)
  library(e1071)
  library(randomForest)
  library(pROC)
  set.seed(123)
  train_index <- createDataPartition(bank_scaled$y, p = 0.80, list = FALSE)
  train_data <- bank_scaled[train_index, ]
  test_data <- bank_scaled[-train_index, ]
  cat("\nTraining Set Dimensions:", dim(train_data), "\n")
  cat("Testing Set Dimensions :", dim(test_data), "\n")
}

Q11 <- {
  lr_model <- glm(y ~ ., data = train_data, family = binomial(link = "logit"))
  summary(lr_model)
}

Q12 <- {
  knn_control <- trainControl(method = "cv", number = 5)
  knn_model <- train(
    y ~ .,
    data = train_data,
    method = "knn",
    trControl = knn_control,
    tuneGrid = expand.grid(k = seq(3, 15, by = 2))
  )
  cat("Optimal K selected by cross-validation:", knn_model$bestTune$k, "\n")
  plot(knn_model, main = "KNN Accuracy across different K values")
}

Q13 <- {
  dt_model <- rpart(y ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.01))
  rpart.plot(dt_model, main = "Decision Tree Classification", type = 4, extra = 104)
}

Q14 <- {
  nb_model <- naiveBayes(y ~ ., data = train_data)
  print(nb_model)
}

Q15 <- {
  svm_model <- svm(y ~ ., data = train_data, kernel = "radial", probability = TRUE)
  print(svm_model)
}

Q16 <- {
  rf_model <- randomForest(y ~ ., data = train_data, ntree = 100, importance = TRUE)
  print(rf_model)
}

Q17 <- {
  pred_lr <- factor(ifelse(predict(lr_model, newdata = test_data, type = "response") > 0.5, "yes", "no"), levels = c("no", "yes"))
  pred_knn <- predict(knn_model, newdata = test_data)
  pred_dt <- predict(dt_model, newdata = test_data, type = "class")
  pred_nb <- predict(nb_model, newdata = test_data)
  pred_svm <- predict(svm_model, newdata = test_data)
  pred_rf <- predict(rf_model, newdata = test_data)

  prob_lr <- predict(lr_model, newdata = test_data, type = "response")
  prob_knn <- predict(knn_model, newdata = test_data, type = "prob")[, "yes"]
  prob_dt <- predict(dt_model, newdata = test_data, type = "prob")[, "yes"]
  prob_nb <- predict(nb_model, newdata = test_data, type = "raw")[, "yes"]
  svm_prediction <- predict(svm_model, newdata = test_data, probability = TRUE)
  prob_svm <- attr(svm_prediction, "probabilities")[, "yes"]
  prob_rf <- predict(rf_model, newdata = test_data, type = "prob")[, "yes"]
}

Q18 <- {
  cm_lr <- confusionMatrix(pred_lr, test_data$y, positive = "yes")
  cm_knn <- confusionMatrix(pred_knn, test_data$y, positive = "yes")
  cm_dt <- confusionMatrix(pred_dt, test_data$y, positive = "yes")
  cm_nb <- confusionMatrix(pred_nb, test_data$y, positive = "yes")
  cm_svm <- confusionMatrix(pred_svm, test_data$y, positive = "yes")
  cm_rf <- confusionMatrix(pred_rf, test_data$y, positive = "yes")
  cat("Confusion Matrix - Random Forest:\n")
  print(cm_rf$table)
}

Q19 <- {
  calc_metrics <- function(cm, prob, actual, name) {
    acc <- cm$overall["Accuracy"]
    prec <- cm$byClass["Precision"]
    rec <- cm$byClass["Sensitivity"]
    spec <- cm$byClass["Specificity"]
    f1 <- cm$byClass["F1"]
    roc_obj <- roc(actual, prob, levels = c("no", "yes"), quiet = TRUE)
    auc_val <- as.numeric(auc(roc_obj))
    data.frame(
      Model = name,
      Accuracy = round(acc, 4),
      Precision = round(prec, 4),
      Recall = round(rec, 4),
      Specificity = round(spec, 4),
      F1_Score = round(f1, 4),
      ROC_AUC = round(auc_val, 4)
    )
  }
  metrics_lr <- calc_metrics(cm_lr, prob_lr, test_data$y, "Logistic Regression")
  metrics_knn <- calc_metrics(cm_knn, prob_knn, test_data$y, "K-Nearest Neighbors")
  metrics_dt <- calc_metrics(cm_dt, prob_dt, test_data$y, "Decision Tree")
  metrics_nb <- calc_metrics(cm_nb, prob_nb, test_data$y, "Naive Bayes")
  metrics_svm <- calc_metrics(cm_svm, prob_svm, test_data$y, "Support Vector Machine")
  metrics_rf <- calc_metrics(cm_rf, prob_rf, test_data$y, "Random Forest")
}

Q20 <- {
  model_comparison <- rbind(metrics_lr, metrics_knn, metrics_dt, metrics_nb, metrics_svm, metrics_rf)
  model_comparison <- model_comparison[order(-model_comparison$F1_Score), ]
  print(model_comparison)
  cat("\nBest Performing Model based on F1-Score & ROC-AUC:", model_comparison$Model[1], "\n")
}

Q21 <- {
  rf_importance <- importance(rf_model)
  cat("\nTop 10 Feature Importances (Mean Decrease Gini):\n")
  print(head(rf_importance[order(-rf_importance[, "MeanDecreaseGini"]), , drop = FALSE], 10))
  varImpPlot(rf_model, main = "Feature Importance - Random Forest")
}
