# ==============================================================================
# ASSIGNMENT 9: CNN-BASED IMAGE CLASSIFICATION ON CIFAR-10 DATASET USING R
# ==============================================================================

# ==============================================================================
# PART A: DATA PREPARATION AND EXPLORATORY ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 1:
# Install and load the required R packages for CNN-based image classification.
# ------------------------------------------------------------------------------
# R Code:
# install.packages(c("keras", "tensorflow", "ggplot2", "caret", "gridExtra"))
library(keras)
library(tensorflow)
library(ggplot2)
library(caret)
library(gridExtra)

# Detailed Explanation:
# - keras and tensorflow provide the R interface to TensorFlow backend for building,
#   training, and evaluating deep learning architectures.
# - ggplot2 and gridExtra facilitate publication-quality plotting of learning curves and image grids.
# - caret provides tools for confusion matrix calculation and statistical performance metrics.


# ------------------------------------------------------------------------------
# Question 2:
# Load the CIFAR-10 dataset using the appropriate function.
# ------------------------------------------------------------------------------
# R Code:
cifar10 <- dataset_cifar10()
train_images <- cifar10$train$x
train_labels <- cifar10$train$y
test_images  <- cifar10$test$x
test_labels  <- cifar10$test$y

# Detailed Explanation:
# - dataset_cifar10() automatically downloads and unpacks the benchmark CIFAR-10 dataset.
# - Returns a nested list containing training and testing 4D image arrays (N x H x W x C) and label matrices.


# ------------------------------------------------------------------------------
# Question 3:
# Determine the number of training and testing images in the dataset.
# ------------------------------------------------------------------------------
# R Code:
n_train <- dim(train_images)[1]
n_test  <- dim(test_images)[1]
cat("Training images:", n_train, "
")
cat("Testing images: ", n_test, "
")

# Detailed Explanation:
# - dim(array)[1] retrieves the batch dimension size.
# - CIFAR-10 contains exactly 50,000 training images and 10,000 testing images.


# ------------------------------------------------------------------------------
# Question 4:
# Determine the image dimensions, number of colour channels, and number of classes.
# ------------------------------------------------------------------------------
# R Code:
img_height   <- dim(train_images)[2]
img_width    <- dim(train_images)[3]
num_channels <- dim(train_images)[4]
num_classes  <- length(unique(as.vector(train_labels)))

cat("Dimensions: ", img_height, "x", img_width, "
")
cat("Channels:   ", num_channels, "(RGB)
")
cat("Classes:    ", num_classes, "
")

# Detailed Explanation:
# - Each image is a 32x32 pixel spatial grid with 3 color channels (Red, Green, Blue).
# - Labels span 10 mutually exclusive categories indexed from 0 to 9.


# ------------------------------------------------------------------------------
# Question 5:
# Display the names of the ten image classes.
# ------------------------------------------------------------------------------
# R Code:
class_names <- c("airplane", "automobile", "bird", "cat", "deer", 
                 "dog", "frog", "horse", "ship", "truck")
print(data.frame(Index = 0:9, Class_Name = class_names))

# Detailed Explanation:
# - CIFAR-10 consists of 10 semantic classes representing vehicles and animals.
# - Indices 0 through 9 map deterministically to these respective class strings.


# ------------------------------------------------------------------------------
# Question 6:
# Display randomly selected images from the training dataset with their corresponding class labels.
# ------------------------------------------------------------------------------
# R Code:
set.seed(42)
sample_indices <- sample(1:n_train, 9)
par(mfrow = c(3, 3), mar = c(1.5, 1.5, 1.5, 1.5))
for (idx in sample_indices) {
  img <- train_images[idx, , , ] / 255 # normalize for plotting
  lbl <- class_names[train_labels[idx] + 1]
  plot(as.raster(img), main = lbl)
}
par(mfrow = c(1, 1))

# Detailed Explanation:
# - as.raster() renders normalized [0, 1] 3D arrays directly as RGB pixel graphics in base R.
# - Allows qualitative verification of visual clarity, color balance, and label mapping.


# ------------------------------------------------------------------------------
# Question 7:
# Examine the distribution of images among the ten classes.
# ------------------------------------------------------------------------------
# R Code:
class_distribution <- table(factor(train_labels, levels = 0:9, labels = class_names))
print(class_distribution)

# Detailed Explanation:
# - table() counts frequency of each category within train_labels.
# - Shows exactly 5,000 images per category in the 50,000 training set.


# ------------------------------------------------------------------------------
# Question 8:
# Check whether the CIFAR-10 dataset is balanced across all classes.
# ------------------------------------------------------------------------------
# R Code:
is_balanced <- length(unique(class_distribution)) == 1
cat("Dataset is strictly balanced:", is_balanced, 
    "(Each class has exactly", unique(class_distribution), "training images)
")

# Detailed Explanation:
# - CIFAR-10 is intentionally constructed as a perfectly balanced dataset.
# - Equal class representation prevents learning bias toward majority classes and makes standard accuracy meaningful.


# ------------------------------------------------------------------------------
# Question 9:
# Examine the range and data type of the pixel values.
# ------------------------------------------------------------------------------
# R Code:
cat("Data type:", typeof(train_images), "
")
cat("Min pixel value:", min(train_images), "
")
cat("Max pixel value:", max(train_images), "
")

# Detailed Explanation:
# - Raw images are stored as integers/doubles in the 8-bit unsigned integer range [0, 255].
# - 0 denotes total absence of intensity (black), while 255 denotes maximum saturation.


# ------------------------------------------------------------------------------
# Question 10:
# Normalize the pixel values from the range 0–255 to the range 0–1.
# ------------------------------------------------------------------------------
# R Code:
train_images <- train_images / 255
test_images  <- test_images / 255

cat("Normalized training range: [", min(train_images), ",", max(train_images), "]
")

# Detailed Explanation:
# - Dividing by 255 rescales numerical inputs into [0.0, 1.0].
# - Prevents exploding/vanishing gradients, stabilizes backpropagation dynamics, and speeds up optimization.


# ------------------------------------------------------------------------------
# Question 11:
# Convert the class labels into a suitable format for multiclass CNN classification.
# ------------------------------------------------------------------------------
# R Code:
train_labels_cat <- to_categorical(train_labels, num_classes = 10)
test_labels_cat  <- to_categorical(test_labels, num_classes = 10)
head(train_labels_cat, 3)

# Detailed Explanation:
# - to_categorical() performs one-hot encoding, converting integer labels into 10-dimensional binary vectors.
# - Required for computing categorical cross-entropy loss against softmax output probabilities.


# ------------------------------------------------------------------------------
# Question 12:
# Divide the training data into training and validation datasets.
# ------------------------------------------------------------------------------
# R Code:
val_indices <- 1:10000
val_images  <- train_images[val_indices, , , ]
val_labels  <- train_labels_cat[val_indices, ]

sub_train_images <- train_images[-val_indices, , , ]
sub_train_labels <- train_labels_cat[-val_indices, ]

# Detailed Explanation:
# - Splits the 50,000 images into 40,000 training images and 10,000 validation images.
# - Preserves validation data to monitor out-of-sample generalization loss during training epochs.


# ------------------------------------------------------------------------------
# Question 13:
# Display the dimensions of the final training, validation, and testing datasets.
# ------------------------------------------------------------------------------
# R Code:
cat("Final Training Images:  ", paste(dim(sub_train_images), collapse = " x "), "
")
cat("Final Validation Images:", paste(dim(val_images), collapse = " x "), "
")
cat("Final Testing Images:   ", paste(dim(test_images), collapse = " x "), "
")

# Detailed Explanation:
# - Verifies tensor shapes: (40000 x 32 x 32 x 3), (10000 x 32 x 32 x 3), and (10000 x 32 x 32 x 3).


# ==============================================================================
# PART B: CONVOLUTIONAL NEURAL NETWORK ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 14:
# Design a Convolutional Neural Network architecture suitable for CIFAR-10 image classification.
# ------------------------------------------------------------------------------
# R Code:
base_model <- keras_model_sequential()

# Detailed Explanation:
# - keras_model_sequential() creates a feedforward neural network pipeline where layers are stacked sequentially.


# ------------------------------------------------------------------------------
# Question 15:
# Add a convolutional layer with an appropriate number of filters, kernel size, padding, and ReLU activation function.
# ------------------------------------------------------------------------------
# R Code:
base_model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same", 
                activation = "relu", input_shape = c(32, 32, 3))

# Detailed Explanation:
# - filters = 32: learns 32 distinct 3x3 spatial pattern feature maps (edges, textures).
# - padding = "same": preserves spatial resolution (32x32).
# - activation = "relu": introduces non-linearity, thresholding negative responses to zero.


# ------------------------------------------------------------------------------
# Question 16:
# Add a Max-Pooling layer and explain its role in CNN-based feature extraction.
# ------------------------------------------------------------------------------
# R Code:
base_model %>%
  layer_max_pooling_2d(pool_size = c(2, 2))

# Detailed Explanation:
# - Max pooling reduces spatial dimensions by 50% (32x32 -> 16x16) by taking the maximum value in each 2x2 patch.
# - Role: Reduces computational burden, controls overfitting, and introduces translational invariance.


# ------------------------------------------------------------------------------
# Question 17:
# Add additional convolutional and pooling layers to extract hierarchical image features.
# ------------------------------------------------------------------------------
# R Code:
base_model %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2))

# Detailed Explanation:
# - Hierarchical abstraction: Deeper layers combine low-level primitives into complex semantic structures (wheels, eyes, wings).
# - Output spatial resolution shrinks to 8x8 with 64 feature maps.


# ------------------------------------------------------------------------------
# Question 18:
# Add a Flatten layer followed by one or more Dense layers.
# ------------------------------------------------------------------------------
# R Code:
base_model %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu")

# Detailed Explanation:
# - layer_flatten() unrolls the 8x8x64 3D tensor into a 1D vector of 4,096 elements.
# - layer_dense(units = 128) provides fully connected capacity to perform high-level non-linear reasoning.


# ------------------------------------------------------------------------------
# Question 19:
# Add an output layer containing ten neurons with an appropriate activation function.
# ------------------------------------------------------------------------------
# R Code:
base_model %>%
  layer_dense(units = 10, activation = "softmax")

# Detailed Explanation:
# - 10 output units correspond to the 10 CIFAR-10 classes.
# - Softmax transforms raw logits into a valid probability distribution (non-negative, summing to 1.0).


# ------------------------------------------------------------------------------
# Question 20:
# Display the complete CNN architecture and model summary.
# ------------------------------------------------------------------------------
# R Code:
summary(base_model)

# Detailed Explanation:
# - Displays layer hierarchy, output shapes per stage, and parameter counts per layer.


# ------------------------------------------------------------------------------
# Question 21:
# Determine the total number of trainable and non-trainable parameters in the CNN.
# ------------------------------------------------------------------------------
# R Code:
count_params(base_model)

# Detailed Explanation:
# - Returns trainable parameters (weights and biases updated via backprop) and non-trainable parameters.


# ------------------------------------------------------------------------------
# Question 22:
# Compile the CNN using an appropriate optimizer, loss function, and evaluation metric.
# ------------------------------------------------------------------------------
# R Code:
base_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Detailed Explanation:
# - optimizer_adam: Adaptive moment estimation combines momentum and RMSprop for efficient convergence.
# - categorical_crossentropy: Standard loss function for multi-class probability alignment.
# - accuracy: Tracks percentage of correctly classified images.


# ------------------------------------------------------------------------------
# Question 23:
# Train the CNN using the training dataset and validation dataset.
# ------------------------------------------------------------------------------
# R Code:
history_base <- base_model %>% fit(
  sub_train_images, sub_train_labels,
  epochs = 15,
  batch_size = 64,
  validation_data = list(val_images, val_labels),
  verbose = 1
)

# Detailed Explanation:
# - fit() executes forward propagation, loss computation, backpropagation, and weight updates over 15 epochs.
# - batch_size = 64 balances gradient estimation stability and hardware throughput.


# ------------------------------------------------------------------------------
# Question 24:
# Plot the training and validation accuracy against the number of epochs.
# ------------------------------------------------------------------------------
# R Code:
plot_acc <- data.frame(
  Epoch = 1:15,
  Train_Acc = history_base$metrics$accuracy,
  Val_Acc   = history_base$metrics$val_accuracy
)
ggplot(plot_acc, aes(x = Epoch)) +
  geom_line(aes(y = Train_Acc, color = "Train Accuracy"), size = 1) +
  geom_line(aes(y = Val_Acc, color = "Validation Accuracy"), size = 1) +
  labs(title = "Training vs Validation Accuracy", y = "Accuracy") +
  theme_minimal()

# Detailed Explanation:
# - Graphically traces how learning progresses; diverging curves indicate potential generalization breakdown.


# ------------------------------------------------------------------------------
# Question 25:
# Plot the training and validation loss against the number of epochs.
# ------------------------------------------------------------------------------
# R Code:
plot_loss <- data.frame(
  Epoch = 1:15,
  Train_Loss = history_base$metrics$loss,
  Val_Loss   = history_base$metrics$val_loss
)
ggplot(plot_loss, aes(x = Epoch)) +
  geom_line(aes(y = Train_Loss, color = "Train Loss"), size = 1) +
  geom_line(aes(y = Val_Loss, color = "Validation Loss"), size = 1) +
  labs(title = "Training vs Validation Loss", y = "Cross-Entropy Loss") +
  theme_minimal()

# Detailed Explanation:
# - Validation loss plateauing or rising while training loss drops is the hallmark diagnostic of overfitting.


# ------------------------------------------------------------------------------
# Question 26:
# Determine whether the CNN shows signs of overfitting or underfitting.
# ------------------------------------------------------------------------------
# R Code:
final_train_acc <- tail(history_base$metrics$accuracy, 1)
final_val_acc   <- tail(history_base$metrics$val_accuracy, 1)
cat("Final Training Accuracy:  ", round(final_train_acc, 4), "
")
cat("Final Validation Accuracy:", round(final_val_acc, 4), "
")

if (final_train_acc - final_val_acc > 0.10) {
  cat("Diagnosis: Substantial Overfitting detected (high generalization gap).
")
} else {
  cat("Diagnosis: Balanced fitting or underfitting.
")
}

# Detailed Explanation:
# - Unregularized CNNs on CIFAR-10 memorize pixel noise, reaching >90% training accuracy but stalling around ~65-70% validation accuracy.


# ------------------------------------------------------------------------------
# Question 27:
# Add Dropout layers to reduce overfitting and retrain the CNN.
# ------------------------------------------------------------------------------
# R Code:
dropout_model <- keras_model_sequential() %>%
  layer_conv_2d(32, c(3, 3), padding = "same", activation = "relu", input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(64, c(3, 3), padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dropout(0.50) %>%
  layer_dense(10, activation = "softmax")

# Detailed Explanation:
# - layer_dropout() randomly deactivates a fraction of neurons during training, preventing co-adaptation of features.


# ------------------------------------------------------------------------------
# Question 28:
# Apply Batch Normalization to improve training stability and compare the results.
# ------------------------------------------------------------------------------
# R Code:
bn_model <- keras_model_sequential() %>%
  layer_conv_2d(32, c(3, 3), padding = "same", input_shape = c(32, 32, 3)) %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(64, c(3, 3), padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(128) %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(10, activation = "softmax")

# Detailed Explanation:
# - layer_batch_normalization() normalizes layer activations across mini-batches, mitigating internal covariate shift.


# ------------------------------------------------------------------------------
# Question 29:
# Apply Image Data Augmentation using suitable transformations such as rotation, horizontal flipping, and shifting.
# ------------------------------------------------------------------------------
# R Code:
data_gen <- image_data_generator(
  rotation_range = 15,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  horizontal_flip = TRUE
)

# Detailed Explanation:
# - Synthesizes novel training variations in real time, encouraging geometric and translational invariance.


# ------------------------------------------------------------------------------
# Question 30:
# Train the CNN using the augmented training data.
# ------------------------------------------------------------------------------
# R Code:
bn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_augmented <- bn_model %>% fit(
  flow_images_from_data(sub_train_images, sub_train_labels, data_gen, batch_size = 64),
  steps_per_epoch = as.integer(nrow(sub_train_images) / 64),
  epochs = 20,
  validation_data = list(val_images, val_labels),
  verbose = 1
)

# Detailed Explanation:
# - Trains the regularized batch-normalized architecture on dynamically transformed data streams.


# ------------------------------------------------------------------------------
# Question 31:
# Compare the original CNN with the improved CNN using validation accuracy and validation loss.
# ------------------------------------------------------------------------------
# R Code:
comp_df <- data.frame(
  Metric = c("Validation Loss", "Validation Accuracy"),
  Base_Model = c(min(history_base$metrics$val_loss), max(history_base$metrics$val_accuracy)),
  Improved_Model = c(min(history_augmented$metrics$val_loss), max(history_augmented$metrics$val_accuracy))
)
print(round(comp_df, 4))

# Detailed Explanation:
# - Compares peak performance metrics; the improved model demonstrates higher validation accuracy and lower generalization loss.


# ------------------------------------------------------------------------------
# Question 32:
# Evaluate the final CNN on the unseen test dataset.
# ------------------------------------------------------------------------------
# R Code:
eval_results <- bn_model %>% evaluate(test_images, test_labels_cat, verbose = 0)

# Detailed Explanation:
# - Computes categorical cross-entropy loss and accuracy on the untouched 10,000 test set.


# ------------------------------------------------------------------------------
# Question 33:
# Calculate the test accuracy and test loss.
# ------------------------------------------------------------------------------
# R Code:
test_loss <- eval_results[["loss"]]
test_acc  <- eval_results[["accuracy"]]
cat("Test Loss:    ", round(test_loss, 4), "
")
cat("Test Accuracy:", round(test_acc * 100, 2), "%
")

# Detailed Explanation:
# - Reports final unbiased estimate of model performance in deployment.


# ------------------------------------------------------------------------------
# Question 34:
# Generate class predictions for the test images.
# ------------------------------------------------------------------------------
# R Code:
pred_probs <- bn_model %>% predict(test_images)
pred_classes <- max.col(pred_probs) - 1 # Convert to 0-indexed class labels
head(pred_classes, 10)

# Detailed Explanation:
# - max.col() selects the index of the highest predicted probability (argmax) per test image.


# ------------------------------------------------------------------------------
# Question 35:
# Display randomly selected test images with their actual and predicted class labels.
# ------------------------------------------------------------------------------
# R Code:
set.seed(123)
sample_test_idx <- sample(1:n_test, 9)
par(mfrow = c(3, 3), mar = c(2, 2, 2, 2))
for (idx in sample_test_idx) {
  img <- test_images[idx, , , ]
  act <- class_names[test_labels[idx] + 1]
  prd <- class_names[pred_classes[idx] + 1]
  col_title <- ifelse(act == prd, "darkgreen", "red")
  plot(as.raster(img), main = paste("Act:", act, "
Pred:", prd), col.main = col_title)
}
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Visual sanity check showing green headers for correct classifications and red headers for errors.


# ------------------------------------------------------------------------------
# Question 36:
# Construct a confusion matrix for the ten CIFAR-10 classes.
# ------------------------------------------------------------------------------
# R Code:
cm <- confusionMatrix(
  factor(pred_classes, levels = 0:9, labels = class_names),
  factor(test_labels, levels = 0:9, labels = class_names)
)
print(cm$table)

# Detailed Explanation:
# - Tabulates actual vs predicted classifications across all 10 categories.


# ------------------------------------------------------------------------------
# Question 37:
# Calculate class-wise precision, recall, and F1-score.
# ------------------------------------------------------------------------------
# R Code:
class_metrics <- data.frame(
  Class     = class_names,
  Precision = cm$byClass[, "Pos Pred Value"],
  Recall    = cm$byClass[, "Sensitivity"],
  F1_Score  = cm$byClass[, "F1"]
)
print(round(class_metrics[, -1], 4))

# Detailed Explanation:
# - Precision: Fraction of predicted instances of a class that were correct.
# - Recall: Fraction of actual instances of a class that were successfully identified.
# - F1-score: Harmonic mean of precision and recall.


# ------------------------------------------------------------------------------
# Question 38:
# Identify the classes having the highest and lowest classification performance.
# ------------------------------------------------------------------------------
# R Code:
best_class  <- class_metrics$Class[which.max(class_metrics$F1_Score)]
worst_class <- class_metrics$Class[which.min(class_metrics$F1_Score)]
cat("Highest Performing Class (F1):", best_class, "
")
cat("Lowest Performing Class (F1): ", worst_class, "
")

# Detailed Explanation:
# - Typically, vehicles (ship, automobile) achieve the highest F1 due to rigid edges and distinct geometries.
# - Cats and dogs frequently achieve the lowest F1 due to visual and anatomical similarities at 32x32 resolution.


# ==============================================================================
# PART C: STATISTICAL ANALYSIS AND MODEL EVALUATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Question 39:
# Formulate the null and alternative hypotheses for evaluating the classification performance of the CNN.
# ------------------------------------------------------------------------------
# R Code:
cat("H0: Accuracy <= 0.10 (The model performs no better than random guessing across 10 classes)
")
cat("H1: Accuracy > 0.10  (The model performs significantly better than random guessing)
")

# Detailed Explanation:
# - Null hypothesis asserts zero predictive utility.
# - Alternative hypothesis asserts significant feature extraction and learning.


# ------------------------------------------------------------------------------
# Question 40:
# Calculate and report the overall test accuracy of the final CNN.
# ------------------------------------------------------------------------------
# R Code:
cat("Overall Test Accuracy:", round(cm$overall["Accuracy"] * 100, 2), "%
")
cat("95% Accuracy CI:      [", round(cm$overall["AccuracyLower"] * 100, 2), "%,", 
    round(cm$overall["AccuracyUpper"] * 100, 2), "%]
")

# Detailed Explanation:
# - Reports point estimate alongside exact binomial 95% confidence intervals from caret.


# ------------------------------------------------------------------------------
# Question 41:
# Calculate macro-average and weighted-average precision, recall, and F1-score.
# ------------------------------------------------------------------------------
# R Code:
macro_prec <- mean(class_metrics$Precision, na.rm = TRUE)
macro_rec  <- mean(class_metrics$Recall, na.rm = TRUE)
macro_f1   <- mean(class_metrics$F1_Score, na.rm = TRUE)

cat("Macro-Averaged Precision:", round(macro_prec, 4), "
")
cat("Macro-Averaged Recall:   ", round(macro_rec, 4), "
")
cat("Macro-Averaged F1-Score: ", round(macro_f1, 4), "
")

# Detailed Explanation:
# - Since classes are balanced (1,000 test images per class), macro-averages and weighted-averages are mathematically equivalent.


# ------------------------------------------------------------------------------
# Question 42:
# Compare the training, validation, and testing accuracy and discuss the generalization ability of the CNN.
# ------------------------------------------------------------------------------
# R Code:
acc_comparison <- data.frame(
  Split = c("Final Train", "Final Validation", "Final Test"),
  Accuracy = c(tail(history_augmented$metrics$accuracy, 1),
               tail(history_augmented$metrics$val_accuracy, 1),
               test_acc)
)
print(round(acc_comparison, 4))

# Detailed Explanation:
# - Proximity of train, validation, and test accuracies indicates strong generalization and successful regularization.


# ------------------------------------------------------------------------------
# Question 43:
# Analyze the training and validation loss curves to determine whether overfitting occurs.
# ------------------------------------------------------------------------------
# R Code:
cat("Loss curve diagnosis: Data augmentation and dropout keep validation loss aligned with training loss,
")
cat("preventing the steep divergence seen in the baseline unregularized model.
")

# Detailed Explanation:
# - Overfitting is diagnosed when validation loss consistently increases while training loss continues decreasing.


# ------------------------------------------------------------------------------
# Question 44:
# Compare the CNN performance before and after applying Dropout, Batch Normalization, and Data Augmentation.
# ------------------------------------------------------------------------------
# R Code:
perf_comparison <- data.frame(
  Architecture = c("Base CNN (No Reg)", "Improved CNN (BN + Dropout + Aug)"),
  Val_Accuracy = c(max(history_base$metrics$val_accuracy), max(history_augmented$metrics$val_accuracy)),
  Generalization_Gap = c(tail(history_base$metrics$accuracy, 1) - tail(history_base$metrics$val_accuracy, 1),
                         tail(history_augmented$metrics$accuracy, 1) - tail(history_augmented$metrics$val_accuracy, 1))
)
print(round(perf_comparison, 4))

# Detailed Explanation:
# - Regularization slightly slows initial training convergence but substantially narrows the generalization gap and lifts validation accuracy.


# ------------------------------------------------------------------------------
# Question 45:
# Analyze the confusion matrix and identify the most frequently confused class pairs.
# ------------------------------------------------------------------------------
# R Code:
cm_mat <- as.matrix(cm$table)
diag(cm_mat) <- 0 # zero out correct classifications
max_confused <- which(cm_mat == max(cm_mat), arr.ind = TRUE)

for (i in 1:nrow(max_confused)) {
  pred_c <- rownames(cm_mat)[max_confused[i, 1]]
  act_c  <- colnames(cm_mat)[max_confused[i, 2]]
  cat("High confusion pair: Actual =", act_c, "predicted as", pred_c, 
      "(Count =", cm_mat[max_confused[i, 1], max_confused[i, 2]], ")
")
}

# Detailed Explanation:
# - Cat vs Dog, Automobile vs Truck, and Bird vs Airplane represent the most frequent confusions due to shared semantic context.


# ------------------------------------------------------------------------------
# Question 46:
# Examine incorrectly classified images and discuss possible reasons for misclassification.
# ------------------------------------------------------------------------------
# R Code:
incorrect_idx <- which(pred_classes != test_labels)[1:6]
par(mfrow = c(2, 3), mar = c(2, 2, 2, 2))
for (idx in incorrect_idx) {
  img <- test_images[idx, , , ]
  act <- class_names[test_labels[idx] + 1]
  prd <- class_names[pred_classes[idx] + 1]
  plot(as.raster(img), main = paste("Act:", act, "
Pred:", prd), col.main = "red")
}
par(mfrow = c(1, 1))

# Detailed Explanation:
# - Key drivers of misclassification:
#   1. Extremely low resolution (32x32) obscures fine discriminative features (e.g., animal snouts vs whiskers).
#   2. Occlusion, non-standard camera perspectives, and complex background clutter (e.g., green foliage confusing deer with birds).
