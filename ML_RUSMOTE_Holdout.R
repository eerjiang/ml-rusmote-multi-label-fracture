library(dplyr)
library(keras3)
library(mldr)
library(utiml)
library(mldr.resampling)
library(pROC)
library(PRROC)
library(tibble)

# Import pre-written ML-RUSMOTE algorithm functions
source("ML_RUSMOTE_Function.R")

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
model_set <- c(1, 1, 1) #BR, ML-kNN, DNN
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
tbl_orig_train_x <- tbl_orig_2[,1:n_features]
tbl_orig_train_y <- tbl_orig_2[,n1:n2]

# Record and print time consumption of resampling
start_time <- Sys.time()
a1 <- ML_RUS_SMOTE(
  X = as.matrix(tbl_orig_train_x),
  y = as.matrix(tbl_orig_train_y),
  nsr = 0.5, 
  usr = 1.25, 
  k = 5,
  f_cont = c(1:n_features_cont),
  f_cat = c(n3:n_features)
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
tbl_mlrusmote_train_x <- as.matrix(tbl_mlrusmote_train_2$dataset[,1:n_features])
tbl_mlrusmote_train_y <- as.matrix(tbl_mlrusmote_train_2$dataset[,n1:n2])

# -------------------------- Train Three Final Models --------------------------
# Train Binary Relevance (BR) Model
if (model_set[1] == 1) {
  cat("\n=== Training BR Model ===\n")
  model_br_mlrusmote <- br(tbl_mlrusmote_train_2)
}

# Train ML-kNN Model
if (model_set[2] == 1) {
  cat("\n=== Training ML-kNN Model ===\n")
  model_mlknn_mlrusmote <- mlknn(tbl_mlrusmote_train_2, k = 3)
}

# Train ML-DNN Deep Model
if (model_set[3] == 1) {
  cat("\n=== Training DNN Model ===\n")
  feature_dim <- ncol(tbl_mlrusmote_train_x)
  label_dim <- ncol(tbl_mlrusmote_train_y)
  
  # Construct sequential DNN architecture
  model_dl_mlrusmote <- keras_model_sequential()  %>%
    layer_dense(units = u1, activation = 'relu', input_shape = feature_dim) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = u2, activation = 'relu') %>%
    layer_dense(units = u3, activation = 'relu') %>%
    layer_dense(units = label_dim, activation = 'sigmoid')
  
  # Compile DNN training configuration
  model_dl_mlrusmote %>% compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  # Execute model training
  history_mlrusmote <- model_dl_mlrusmote %>% fit(
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
pred_br_mlrusmote_train <- if (model_set[1] == 1) predict(model_br_mlrusmote, tbl_mlrusmote_train_2) else NULL
pred_mlknn_mlrusmote_train <- if (model_set[2] == 1) predict(model_mlknn_mlrusmote, tbl_mlrusmote_train_2) else NULL
pred_dl_mlrusmote_train <- if (model_set[3] == 1) predict(model_dl_mlrusmote, tbl_mlrusmote_train_x, verbose = 0) else NULL

# Collect only the enabled model predictions
all_train_preds <- compact(list(pred_br_mlrusmote_train, pred_mlknn_mlrusmote_train, pred_dl_mlrusmote_train))
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
    y_true <- as.numeric(tbl_mlrusmote_train_2$dataset[, n_features + label_idx])
    
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

# -------------------------- Run Prediction on Hold-out Set --------------------------
cat("\n=== Running prediction on the hold-out set ===\n")
# BR model prediction and base evaluation
pred_br_mlrusmote_val <- NULL
me_br_mlrusmote_val <- NULL
if (model_set[1] == 1) {
  pred_br_mlrusmote_val <- predict(model_br_mlrusmote, val_set_2)
  me_br_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_br_mlrusmote_val)
  cat("BR model standard evaluation metrics:\n")
  print(me_br_mlrusmote_val)
}

# ML-kNN model prediction and base evaluation
pred_mlknn_mlrusmote_val <- NULL
me_mlknn_mlrusmote_val <- NULL
if (model_set[2] == 1) {
  pred_mlknn_mlrusmote_val <- predict(model_mlknn_mlrusmote, val_set_2)
  me_mlknn_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_mlknn_mlrusmote_val)
  cat("\nML-kNN model standard evaluation metrics:\n")
  print(me_mlknn_mlrusmote_val)
}

# DNN model prediction and base evaluation
pred_dl_mlrusmote_val <- NULL
me_dl_mlrusmote_val <- NULL
if (model_set[3] == 1) {
  pred_dl_mlrusmote_val <- model_dl_mlrusmote %>% predict(val_set_1_x, verbose = 0)
  pred_dl_mlrusmote_val <- as.mlresult(pred_dl_mlrusmote_val)
  me_dl_mlrusmote_val <- multilabel_evaluate(val_set_2, pred_dl_mlrusmote_val)
  cat("\nDNN model standard evaluation metrics:\n")
  print(me_dl_mlrusmote_val)
  # Clean up Keras session to release memory
  gc()
  keras3::k_clear_session()
}

# Export raw prediction results of all models
if (!is.null(pred_br_mlrusmote_val)) write.csv(pred_br_mlrusmote_val, file = "output/holdout/pred_br_mlrusmote_val.csv", row.names = FALSE)
if (!is.null(pred_mlknn_mlrusmote_val)) write.csv(pred_mlknn_mlrusmote_val, file = "output/holdout/pred_mlknn_mlrusmote_val.csv", row.names = FALSE)
if (!is.null(pred_dl_mlrusmote_val)) write.csv(pred_dl_mlrusmote_val, file = "output/holdout/pred_dl_mlrusmote_val.csv", row.names = FALSE)
cat("\nAll raw test set prediction results saved.\n")

# -------------------------- Custom Metric Calculation (Macro F1, MCC, PRAUC) --------------------------
cat("\n=== Calculating custom evaluation metrics: Macro F1, MCC, PRAUC ===\n")
# Collect all valid test set prediction results
all_test_preds <- compact(list(pred_br_mlrusmote_val, pred_mlknn_mlrusmote_val, pred_dl_mlrusmote_val))
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
    y_true <- as.numeric(tbl_temp_true_base[, n_features + label_idx])
    
    # Get pre-calculated optimal threshold for current model and label
    t <- tbl_threshold_mlrusmote_train[model_idx, label_idx]
    # Binarize prediction scores using the optimal threshold
    b0 <- ifelse(y_pred > t, 1, 0)
    
    # Calculate confusion matrix elements
    TN <- length(which(b0==0 & y_true==0)) # Predict 0, ground truth 0
    FN <- length(which(b0==0 & y_true==1)) # Predict 0, ground truth 1
    FP <- length(which(b0==1 & y_true==0)) # Predict 1, ground truth 0
    TP <- length(which(b0==1 & y_true==1)) # Predict 1, ground truth 1
    
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


# =============================================================================
# ML-RUSMOTE Bootstrap Evaluation (1000 iterations)
# =============================================================================


# -------------------------- Global Hyperparameter Configuration --------------
# Number of bootstrap resampling iterations
n_bootstrap <- 1000
# Fixed seed: makes the whole bootstrap procedure 100% reproducible
GLOBAL_SEED <- 12345

# Model names, kept in the same order as model_set
model_names_all     <- c("BR", "ML-kNN", "DNN")
model_names_enabled <- model_names_all[as.logical(model_set)]
n_models_enabled    <- length(model_names_enabled)

# Sanity checks on the objects inherited from the upstream pipeline
stopifnot(length(model_set) == length(model_names_all))
stopifnot(nrow(threshold_tbl) >= n_models_enabled)

# Output directory and file-name tag (aligned with the hold-out script)
out_dir  <- "output/holdout"
file_tag <- sprintf("bootstrap%d_mlrusmote_threshold", n_bootstrap)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("Bootstrap iterations :", n_bootstrap, "\n")
cat("Models to evaluate   :", paste(model_names_enabled, collapse = ", "), "\n")
cat("Hold-out sample size :", nrow(val_set_1), "\n\n")

# -------------------------- Result Containers --------------------------------
# One row per bootstrap iteration, one column per enabled model
macro_f1_set <- matrix(NA_real_, nrow = n_bootstrap, ncol = n_models_enabled)
mcc_set      <- matrix(NA_real_, nrow = n_bootstrap, ncol = n_models_enabled)
prauc_set    <- matrix(NA_real_, nrow = n_bootstrap, ncol = n_models_enabled)
colnames(macro_f1_set) <- model_names_enabled
colnames(mcc_set)      <- model_names_enabled
colnames(prauc_set)    <- model_names_enabled

# -------------------------- Bootstrap Loop -----------------------------------
# Setting the seed once here guarantees that the n_bootstrap resamples, and
# therefore the whole result table, can be reproduced exactly.
set.seed(GLOBAL_SEED)

# Work on a plain data frame copy so that row names can be reset safely,
# regardless of whether the upstream object is a data.frame or a tibble.
val_df <- as.data.frame(val_set_1)
n_obs  <- nrow(val_df)

for (bts in seq_len(n_bootstrap)) {
  
  # ---- 1. Draw one bootstrap sample (with replacement) from the hold-out set
  boot_idx <- sample(seq_len(n_obs), size = n_obs, replace = TRUE)
  boot_df  <- val_df[boot_idx, , drop = FALSE]
  rownames(boot_df) <- NULL
  
  # Feature matrix required by the DNN model
  boot_x <- as.matrix(boot_df[, 1:n_features])
  # mldr object required by the BR and ML-kNN models
  boot_mldr <- mldr_from_dataframe(boot_df, labelIndices = c(n1:n2), name = "bootMLDR")
  
  # ---- 2. Generate predictions with every enabled model ---------------------
  # pred_list[[m]] corresponds to model_names_enabled[m], which also matches
  # row m of threshold_tbl (thresholds are stored in the same model order).
  pred_list <- list()
  if (model_set[1] == 1) pred_list[[length(pred_list) + 1]] <- predict(model_br_mlrusmote, boot_mldr)
  if (model_set[2] == 1) pred_list[[length(pred_list) + 1]] <- predict(model_mlknn_mlrusmote, boot_mldr)
  if (model_set[3] == 1) pred_list[[length(pred_list) + 1]] <- predict(model_dl_mlrusmote, boot_x, verbose = 0)
  
  # ---- 3. Compute Macro F1, MCC and PRAUC per model and per label ----------
  for (m in seq_along(pred_list)) {
    
    y_pred_mat <- pred_list[[m]]
    
    f1_per_label    <- numeric(n_labels)
    mcc_per_label   <- numeric(n_labels)
    prauc_per_label <- numeric(n_labels)
    
    for (i in seq_len(n_labels)) {
      
      y_pred <- as.numeric(y_pred_mat[, i])
      y_true <- as.numeric(boot_df[, n_features + i])
      
      # Binarise the continuous scores with the pre-computed optimal threshold
      t_opt <- tbl_threshold_mlrusmote_train[m, i]
      b0 <- ifelse(y_pred > t_opt, 1, 0)
      
      # Confusion-matrix elements
      TN <- sum(b0 == 0 & y_true == 0)  # predicted 0, ground truth 0
      FN <- sum(b0 == 0 & y_true == 1)  # predicted 0, ground truth 1
      FP <- sum(b0 == 1 & y_true == 0)  # predicted 1, ground truth 0
      TP <- sum(b0 == 1 & y_true == 1)  # predicted 1, ground truth 1
      
      # Matthews Correlation Coefficient (MCC); 0 when the denominator is 0
      mcc_den <- sqrt(as.numeric((TP + FP) * (TP + FN)) * as.numeric((TN + FP) * (TN + FN)))
      mcc_per_label[i] <- if (mcc_den == 0) 0 else (TP * TN - FP * FN) / mcc_den
      
      # F1-score; precision and recall fall back to 0 on empty denominators
      precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
      recall    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
      f1_per_label[i] <- ifelse((precision + recall) == 0, 0,
                                2 * precision * recall / (precision + recall))
      
      # Precision-Recall AUC; undefined when the bootstrap sample contains a
      # single class, which is set to 1.0 by convention (same rule as in the
      # hold-out script)
      if (length(unique(y_true)) < 2) {
        prauc_per_label[i] <- 1.0
      } else {
        prauc_per_label[i] <- pr.curve(
          scores.class0 = y_pred[y_true == 0],  # scores of the negative samples
          scores.class1 = y_pred[y_true == 1],  # scores of the positive samples
          curve = TRUE                          # also return the full PR curve
        )$auc.integral
      }
    }
    
    # Macro averaging: unweighted mean over all labels
    macro_f1_set[bts, m] <- mean(f1_per_label,    na.rm = TRUE)
    mcc_set[bts, m]      <- mean(mcc_per_label,   na.rm = TRUE)
    prauc_set[bts, m]    <- mean(prauc_per_label, na.rm = TRUE)
  }
  
  # Light-weight progress report and memory release
  if (bts == 1 || bts %% 100 == 0) {
    cat(sprintf("Bootstrap iteration %d / %d finished\n", bts, n_bootstrap))
  }
  if (bts %% 100 == 0) gc()
}

# -------------------------- Summarise and Export Results ---------------------
cat("\n=== Bootstrap summary: Macro F1 ===\n")
print(summary(macro_f1_set))
cat("\n=== Bootstrap summary: MCC ===\n")
print(summary(mcc_set))
cat("\n=== Bootstrap summary: PRAUC ===\n")
print(summary(prauc_set))

# Compact summary table: mean, SD and 95% percentile interval of every metric
bootstrap_summary <- data.frame(
  model          = model_names_enabled,
  macro_f1_mean  = colMeans(macro_f1_set),
  macro_f1_sd    = apply(macro_f1_set, 2, sd),
  macro_f1_low   = apply(macro_f1_set, 2, quantile, probs = 0.025, na.rm = TRUE),
  macro_f1_high  = apply(macro_f1_set, 2, quantile, probs = 0.975, na.rm = TRUE),
  mcc_mean       = colMeans(mcc_set),
  mcc_sd         = apply(mcc_set, 2, sd),
  mcc_low        = apply(mcc_set, 2, quantile, probs = 0.025, na.rm = TRUE),
  mcc_high       = apply(mcc_set, 2, quantile, probs = 0.975, na.rm = TRUE),
  prauc_mean     = colMeans(prauc_set),
  prauc_sd       = apply(prauc_set, 2, sd),
  prauc_low      = apply(prauc_set, 2, quantile, probs = 0.025, na.rm = TRUE),
  prauc_high     = apply(prauc_set, 2, quantile, probs = 0.975, na.rm = TRUE),
  row.names      = NULL
)

cat("\n=== Bootstrap summary table (mean / SD / 95% interval) ===\n")
print(bootstrap_summary)

write.csv(macro_f1_set, file = file.path(out_dir, sprintf("tbl_macro_f1_%s.csv", file_tag)), row.names = FALSE)
write.csv(mcc_set,      file = file.path(out_dir, sprintf("tbl_mcc_%s.csv", file_tag)),      row.names = FALSE)
write.csv(prauc_set,    file = file.path(out_dir, sprintf("tbl_prauc_%s.csv", file_tag)),    row.names = FALSE)
write.csv(bootstrap_summary, file = file.path(out_dir, sprintf("tbl_summary_%s.csv", file_tag)), row.names = FALSE)

cat(sprintf("\nAll bootstrap result tables exported to %s/\n", out_dir))
