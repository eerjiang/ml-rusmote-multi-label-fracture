library(dplyr)
library(keras3)
library(mldr)
library(utiml)
library(mldr.resampling)
library(pROC)
library(PRROC)
library(tibble)

# Import pre-written ML-RUSMOTE algorithm functions
source("fun_ml_rusmote.R")

# -------------------------- Global Hyperparameter Configuration --------------------------
# DNN network layer unit settings
u1 <- 256
u2 <- 128
u3 <- 64
# Batch size for DNN training
bs <- 128
# Proportion of validation split during DNN training
vs <- 0.1
# Control flag to decide which models to run (set 1 to enable, 0 to skip)
model_set <- c(0, 0, 1)
# Fixed random seed for 100% result reproducibility
GLOBAL_SEED <- 12345

# -------------------------- Full Pipeline Execution --------------------------
# Load and preprocess full training dataset
tbl_orig_train <- as.data.frame(tbl_orig_2)
tbl_orig_train <- mldr_from_dataframe(tbl_orig_train, labelIndices = c(n1:n2), name = "trainMLDR")

# Print original dataset imbalance statistics for reference
cat("Original dataset label count distribution:\n")
print(table(tbl_orig_train$dataset$.labelcount))
cat("Original dataset per-label Imbalance Ratio (IRLbl):\n")
print(tbl_orig_train$labels$IRLbl)

# Execute ML-RUSMOTE resampling to mitigate multi-label class imbalance
set.seed(GLOBAL_SEED)  # Fixed random seed for result reproducibility
tbl_orig_train_x <- tbl_orig_2[,1:n0]
tbl_orig_train_y <- tbl_orig_2[,n1:n2]

# Record and print time consumption of resampling
start_time <- Sys.time()
a1 <- ML_RUS_SMOTE(
  X = as.matrix(tbl_orig_train_x),
  y = as.matrix(tbl_orig_train_y),
  nsr = 0.5, 
  usr = 1.25, 
  k = 5,
  f_cont = c(1:6),
  f_cat = c(7:n0)
)
resample_time <- Sys.time() - start_time
cat("ML-RUSMOTE resampling finished, time cost:", round(resample_time, 2), units(resample_time), "\n")

# Combine resampled features and labels into a complete data frame
tbl_mlrusmote_train_0 <- as.data.frame(cbind(a1[[1]], a1[[2]]))
# Verify imbalance level after resampling
post_irlbl <- IRLbl(tbl_mlrusmote_train_0[,n1:n2])
post_meanir <- MeanIR(tbl_mlrusmote_train_0[,n1:n2])
cat("\nPost-resample Mean Imbalance Ratio (MeanIR):", round(post_meanir, 3), "\n")
cat("Post-resample per-label Imbalance Ratio (IRLbl):\n")
print(post_irlbl)

# Print post-resample label distribution details for validation
post_label_sum <- apply(tbl_mlrusmote_train_0[,n1:n2], 1, sum)
cat("\nPost-resample number of labels per sample distribution:\n")
print(table(post_label_sum))
cat("Post-resample single-label positive sample counts:\n")
for (col_idx in n1:n2) {
  cat(paste0(colnames(tbl_mlrusmote_train_0)[col_idx], ": ", sum(tbl_mlrusmote_train_0[, col_idx] == 1), "\n"))
}
# Compare with original training set label distribution
orig_label_sum <- apply(tbl_orig_train_y, 1, sum)
cat("\nOriginal training set number of labels per sample distribution:\n")
print(table(orig_label_sum))

# Convert resampled dataset to standard mldr format for multi-label model training
tbl_mlrusmote_train_2 <- mldr_from_dataframe(tbl_mlrusmote_train_0, labelIndices = c(n1:n2), name = "trainMLDR")

# Prepare formatted input for deep learning model
tbl_mlrusmote_train_x <- as.matrix(tbl_mlrusmote_train_2$dataset[,1:n0])
tbl_mlrusmote_train_y <- as.matrix(tbl_mlrusmote_train_2$dataset[,n1:n2])

# -------------------------- Train Three Final Models --------------------------
# Train Binary Relevance (BR) Model
if (model_set == 1) {
  cat("\n=== Training BR Model ===\n")
  model_br_mlrusmote <- br(tbl_mlrusmote_train_2)
}

# Train ML-kNN Model
if (model_set == 1) {
  cat("\n=== Training ML-kNN Model ===\n")
  model_mlknn_mlrusmote <- mlknn(tbl_mlrusmote_train_2, k = 3)
}

# Train ML-DNN Deep Model
if (model_set == 1) {
  cat("\n=== Training DNN Model ===\n")
  feature_dim <- ncol(tbl_mlrusmote_train_x)
  label_dim <- ncol(tbl_mlrusmote_train_y)
  
  # Construct sequential DNN architecture
  model_dnn_mlrusmote <- keras_model_sequential()  %>%
    layer_dense(units = u1, activation = 'relu', input_shape = feature_dim) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = u2, activation = 'relu') %>%
    layer_dense(units = u3, activation = 'relu') %>%
    layer_dense(units = label_dim, activation = 'sigmoid') # Use sigmoid for independent multi-label classification
  
  # Compile DNN training configuration
  model_dnn_mlrusmote %>% compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  # Execute model training
  history_mlrusmote <- model_dnn_mlrusmote %>% fit(
    tbl_mlrusmote_train_x,
    tbl_mlrusmote_train_y,
    epochs = 50,
    batch_size = bs,
    validation_split = vs,
    verbose = FALSE
  )
  
  # Plot and save DNN training curve to local file
  p <- plot(history_mlrusmote)
  ggsave("output/holdout/dnn_training_curve.png", p, width = 8, height = 5)
  cat("DNN training curve saved to output/holdout/dnn_training_curve.png\n")
}

# -------------------------- Calculate Optimal Classification Threshold for Each Model & Label --------------------------
cat("\n=== Calculating optimal classification thresholds based on training set predictions ===\n")
# Generate prediction results on the training set for threshold calculation
pred_br_mlrusmote_train <- if (model_set == 1) predict(model_br_mlrusmote, tbl_mlrusmote_train_2) else NULL
pred_mlknn_mlrusmote_train <- if (model_set == 1) predict(model_mlknn_mlrusmote, tbl_mlrusmote_train_2) else NULL
pred_dnn_mlrusmote_train <- if (model_set == 1) predict(model_dnn_mlrusmote, tbl_mlrusmote_train_x, verbose = 0) else NULL

# Collect only the enabled model predictions
all_train_preds <- compact(list(pred_br_mlrusmote_train, pred_mlknn_mlrusmote_train, pred_dnn_mlrusmote_train))
model_names_enabled <- c("BR", "ML-kNN", "DNN")[as.logical(model_set)]

# Initialize threshold matrix with proper dimension
tbl_threshold_mlrusmote_train <- matrix(NA, nrow = length(all_train_preds), ncol = n_labels)
rownames(tbl_threshold_mlrusmote_train) <- model_names_enabled
colnames(tbl_threshold_mlrusmote_train) <- colnames(tbl_mlrusmote_train_y)

# Loop through each model and each label to find optimal threshold
for (model_idx in 1:length(all_train_preds)) {
  y_pred_all <- all_train_preds[[model_idx]]
  for (label_idx in 1:n_labels) {
    y_pred <- as.numeric(y_pred_all[, label_idx])
    y_true <- as.numeric(tbl_mlrusmote_train_2$dataset[, n0 + label_idx])
    
    # Skip labels with only single class to avoid roc function error
    if (length(unique(y_true)) < 2) {
      warning(paste0("Label ", colnames(tbl_mlrusmote_train_y)[label_idx], " in model ", model_names_enabled[model_idx], " only contains one class, threshold will be set to 0.5 as default"))
      tbl_threshold_mlrusmote_train[model_idx, label_idx] <- 0.5
      next
    }
    
    # Calculate optimal threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)
    roc_lg <- pROC::roc(y_true, y_pred, quiet = TRUE)
    best_cutoff <- coords(roc_lg, x = "best", transpose = FALSE)
    tbl_threshold_mlrusmote_train[model_idx, label_idx] <- best_cutoff$threshold
  }
}

# Export threshold table for later reuse
write.csv(tbl_threshold_mlrusmote_train, file = "output/holdout/optimal_classification_thresholds.csv", row.names = TRUE)
cat("Optimal threshold calculation finished, full table exported to output/holdout/optimal_classification_thresholds.csv\n")

# -------------------------- Run Prediction on Independent Test Set --------------------------
cat("\n=== Running prediction on the test set ===\n")
# BR model prediction and base evaluation
pred_br_mlrusmote_val <- NULL
me_br_mlrusmote_val <- NULL
if (model_set == 1) {
  pred_br_mlrusmote_val <- predict(model_br_mlrusmote, val_set_2)
  me_br_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_br_mlrusmote_val)
  cat("BR model standard evaluation metrics:\n")
  print(me_br_mlrusmote_val)
}

# ML-kNN model prediction and base evaluation
pred_mlknn_mlrusmote_val <- NULL
me_mlknn_mlrusmote_val <- NULL
if (model_set == 1) {
  pred_mlknn_mlrusmote_val <- predict(model_mlknn_mlrusmote, val_set_2)
  me_mlknn_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_mlknn_mlrusmote_val)
  cat("\nML-kNN model standard evaluation metrics:\n")
  print(me_mlknn_mlrusmote_val)
}

# DNN model prediction and base evaluation
pred_dnn_mlrusmote_val <- NULL
me_dnn_mlrusmote_val <- NULL
if (model_set == 1) {
  pred_dnn_mlrusmote_val <- model_dnn_mlrusmote %>% predict(val_set_1_x, verbose = 0)
  pred_dnn_mlrusmote_val <- as.mlresult(pred_dnn_mlrusmote_val)
  me_dnn_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_dnn_mlrusmote_val)
  cat("\nDNN model standard evaluation metrics:\n")
  print(me_dnn_mlrusmote_val)
  # Clean up Keras session to release memory
  gc()
  keras3::k_clear_session()
}

# Export raw prediction results of all models
if (!is.null(pred_br_mlrusmote_val)) write.csv(pred_br_mlrusmote_val, file = "output/holdout/pred_br_mlrusmote_val.csv", row.names = FALSE)
if (!is.null(pred_mlknn_mlrusmote_val)) write.csv(pred_mlknn_mlrusmote_val, file = "output/holdout/pred_mlknn_mlrusmote_val.csv", row.names = FALSE)
if (!is.null(pred_dnn_mlrusmote_val)) write.csv(pred_dnn_mlrusmote_val, file = "output/holdout/pred_dnn_mlrusmote_val.csv", row.names = FALSE)
cat("\nAll raw test set prediction results saved.\n")

# -------------------------- Custom Metric Calculation (Macro F1, MCC, PRAUC) --------------------------
cat("\n=== Calculating custom evaluation metrics: Macro F1, MCC, PRAUC ===\n")
# Collect all valid test set prediction results
all_test_preds <- compact(list(pred_br_mlrusmote_val, pred_mlknn_mlrusmote_val, pred_dnn_mlrusmote_val))
# Get ground truth label matrix of test set
tbl_temp_true_base <- val_set_1

# Initialize final metric storage data frame
tbl_f1_mcc_prauc_val <- tibble()

for (model_idx in 1:length(all_test_preds)) {
  tbl_temp_pred <- all_test_preds[[model_idx]]
  model_name_current <- model_names_enabled[model_idx]
  # Initialize temporary matrix to store metrics for each label
  tbl_f1_training <- matrix(NA, nrow = 1, ncol = n_labels)
  tbl_prauc_training <- matrix(NA, nrow = 1, ncol = n_labels)
  tbl_mcc_training <- matrix(NA, nrow = 1, ncol = n_labels)
  
  for (label_idx in 1:n_labels) {
    y_pred <- as.numeric(tbl_temp_pred[,label_idx])
    y_true <- as.numeric(tbl_temp_true_base[, 23 + label_idx])
    
    # Get pre-calculated optimal threshold for current model and label
    t <- tbl_threshold_mlrusmote_train[model_idx, label_idx]
    # Binarize prediction scores using the optimal threshold
    b0 <- ifelse(y_pred > t, 1, 0)
    
    # Calculate confusion matrix elements
    cm <- table(factor(b0, levels = c(0,1)), factor(y_true, levels = c(0,1)))
    TP <- ifelse(nrow(cm) >=2 && ncol(cm)>=2, cm[2, 2], 0)  # Predict 1, ground truth 1
    TN <- ifelse(nrow(cm) >=1 && ncol(cm)>=1, cm[1, 1], 0)  # Predict 0, ground truth 0
    FP <- ifelse(nrow(cm) >=2 && ncol(cm)>=1, cm[2, 1], 0)  # Predict 1, ground truth 0
    FN <- ifelse(nrow(cm) >=1 && ncol(cm)>=2, cm[1, 2], 0)  # Predict 0, ground truth 1
    
    # Calculate Matthews Correlation Coefficient (MCC), handle zero denominator case
    denominator_mcc <- sqrt(as.numeric((TP + FP) * (TP + FN)) * as.numeric((TN + FP) * (TN + FN)))
    mcc <- ifelse(denominator_mcc == 0, 0, (TP * TN - FP * FN) / denominator_mcc)
    tbl_mcc_training[, label_idx] <- mcc
    
    # Calculate F1-score, handle zero denominator case
    precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    recall    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
    f1 <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
    tbl_f1_training[, label_idx] <- f1
    
    # Calculate Precision-Recall AUC (PRAUC), skip labels with only single class
    if (length(unique(y_true)) < 2) {
      prauc <- 1.0
    } else {
      prauc_obj <- pr.curve(
        scores.class0 = y_pred[y_true == 0],  # Prediction scores for negative samples
        scores.class1 = y_pred[y_true == 1],  # Prediction scores for positive samples
        curve = TRUE  # Return full PR curve data
      )
      prauc <- prauc_obj$auc.integral
    }
    tbl_prauc_training[, label_idx] <- prauc
  }
  
  # Combine per-label metrics for current model
  temp_0 <- rbind(tbl_f1_training, tbl_mcc_training, tbl_prauc_training)
  temp_0 <- as.data.frame(temp_0)
  colnames(temp_0) <- colnames(tbl_mlrusmote_train_y)
  
  # Calculate average metric value across all labels
  temp_0$mean_values <- apply(temp_0, 1, mean, na.rm = TRUE)
  # Add metric type identification column
  temp_0$m_metrics <- c("Macro F1", "MCC", "PRAUC")
  # Add model name identification column
  temp_0$model <- rep(model_name_current, 3)
  
  tbl_f1_mcc_prauc_val <- bind_rows(tbl_f1_mcc_prauc_val, temp_0)
}

# Reorder columns for better readability
tbl_f1_mcc_prauc_val <- tbl_f1_mcc_prauc_val[, c("model", "m_metrics", colnames(tbl_mlrusmote_train_y), "mean_values")]

# Print and export final custom metric table
cat("\nFinal custom evaluation metrics for all models:\n")
print(tbl_f1_mcc_prauc_val)
write.csv(tbl_f1_mcc_prauc_val, file = "output/holdout/tbl_f1_mcc_prauc_val_mlrusmote_threshold.csv", row.names = FALSE)
cat("\nFull custom evaluation result exported to output/holdout/tbl_f1_mcc_prauc_val_mlrusmote_threshold.csv\n")
