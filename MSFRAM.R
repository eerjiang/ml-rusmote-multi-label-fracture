# Comparison pipeline: Benchmark FRAX clinical fracture risk tool against MSFRAM (ML-RUSMOTE based multi-label deep learning model)
library(pROC)
library(dplyr)
library(tibble)

# -------------------------- Global Initialization & Dataset Preprocessing --------------------------
# Load full raw dataset containing both FRAX predicted scores and patient identifiers
data_final2 <- read.csv("input/data_final2.csv", sep = ",", header = TRUE)

# Align patient IDs to match the multi-label preprocessed dataset (only retain patients that exist in both datasets)
matched_indices <- match(data_final2$pat_handle, data_final_mul_2$pat_handle, nomatch = 0)
data_final3 <- data_final2[which(matched_indices != 0), ]

# Keep only required core columns: patient ID, hip fracture ground truth, major fracture ground truth, FRAX hip risk, FRAX major risk
data_final3 <- data_final3[, c(1, 4, 5, 13, 14)]

# Pre-allocate empty columns to store MSFRAM predicted risk scores for 4 different fracture sites
data_final3$forearm_pred <- NA
data_final3$hip_pred <- NA
data_final3$humerus_pred <- NA
data_final3$vertebrae_pred <- NA

# Reset row indices to consecutive integers for easier subset slicing
rownames(data_final3) <- 1:nrow(data_final3)

# Split full dataset into training set and independent holdout validation set using pre-defined split indices
# (Consistent with the train-test split in your previous data preprocessing pipeline)
data_final3_train <- data_final3[dev_index, ]
data_final3_val <- data_final3[-dev_index, ]

# Verify dimension of MSFRAM prediction matrix to ensure alignment with sample count
cat("Dimension of MSFRAM prediction result matrix:\n")
print(dim(pred_dnn_mlrusmote_val))

# -------------------------- Holdout Independent Test Set Evaluation --------------------------
if(TRUE){
  # Assign 4 MSFRAM fracture site prediction scores to corresponding columns in validation set
  for(i in 1:4){
    data_final3_val[, 5 + i] <- pred_dnn_mlrusmote_val[, i]
  }

  # -------------------------- Hip Fracture Prediction Performance Comparison --------------------------
  cat("\n=== Hip Fracture Prediction Performance (Holdout Set) ===\n")
  # Calculate ROC curve, AUC and 95% confidence interval for FRAX hip risk score
  roc_frax_hip <- pROC::roc(data_final3_val$hip_fx, data_final3_val$frax_hip, ci = TRUE, quiet = TRUE)
  cat("FRAX Hip Fracture Prediction AUC:", round(roc_frax_hip$auc, 3), 
      " | 95% CI: [", round(roc_frax_hip$ci, 3), ",", round(roc_frax_hip$ci, 3), "]\n")
  
  # Calculate ROC curve, AUC and 95% confidence interval for MSFRAM hip risk score
  roc_ml_hip <- pROC::roc(data_final3_val$hip_fx, data_final3_val$hip_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Hip Fracture Prediction AUC:", round(roc_ml_hip$auc, 3), 
      " | 95% CI: [", round(roc_ml_hip$ci, 3), ",", round(roc_ml_hip$ci, 3), "]\n")
  
  # Run paired DeLong's test to check if the AUC difference between two models is statistically significant
  delong_hip <- roc.test(roc_frax_hip, roc_ml_hip, paired = TRUE)
  cat("DeLong Test for Hip Fracture AUC Comparison: p-value =", round(delong_hip$p.value, 5), "\n")
  print(delong_hip)

  # -------------------------- Major Osteoporotic Fracture Preprocessing --------------------------
  # Generate two aggregated MSFRAM risk scores for major fracture prediction
  data_final3_val$max_pred <- NA
  data_final3_val$mean_pred <- NA
  
  # Vectorized calculation to replace slow original loop
  data_final3_val <- data_final3_val %>%
    rowwise() %>%
    mutate(
      max_pred = max(c_across(6:9)),
      mean_pred = mean(c_across(6:9))
    ) %>% ungroup()
  
  # -------------------------- Major Fracture Performance Comparison (Max Aggregation Strategy) --------------------------
  cat("\n=== Major Osteoporotic Fracture Prediction Performance (Holdout Set, Max Aggregation) ===\n")
  roc_frax_major <- pROC::roc(data_final3_val$major_fx, data_final3_val$frax_major, ci = TRUE, quiet = TRUE)
  cat("FRAX Major Fracture Prediction AUC:", round(roc_frax_major$auc, 3), 
      " | 95% CI: [", round(roc_frax_major$ci, 3), ",", round(roc_frax_major$ci, 3), "]\n")
  
  roc_ml_major_max <- pROC::roc(data_final3_val$major_fx, data_final3_val$max_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Major Fracture Prediction AUC (Max Aggregation):", round(roc_ml_major_max$auc, 3), 
      " | 95% CI: [", round(roc_ml_major_max$ci, 3), ",", round(roc_ml_major_max$ci, 3), "]\n")
  
  delong_major_max <- roc.test(roc_frax_major, roc_ml_major_max, paired = TRUE)
  cat("DeLong Test for Major Fracture AUC Comparison (Max Aggregation): p-value =", round(delong_major_max$p.value, 5), "\n")
  print(delong_major_max)
  
  # -------------------------- Major Fracture Performance Comparison (Mean Aggregation Strategy) --------------------------
  cat("\n=== Major Osteoporotic Fracture Prediction Performance (Holdout Set, Mean Aggregation) ===\n")
  roc_ml_major_mean <- pROC::roc(data_final3_val$major_fx, data_final3_val$mean_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Major Fracture Prediction AUC (Mean Aggregation):", round(roc_ml_major_mean$auc, 3), 
      " | 95% CI: [", round(roc_ml_major_mean$ci, 3), ",", round(roc_ml_major_mean$ci, 3), "]\n")
  
  delong_major_mean <- roc.test(roc_frax_major, roc_ml_major_mean, paired = TRUE)
  cat("DeLong Test for Major Fracture AUC Comparison (Mean Aggregation): p-value =", round(delong_major_mean$p.value, 5), "\n")
  print(delong_major_mean)
  
  # Export full holdout set evaluation results to local file for later report generation
  write.csv(data_final3_val, file = "output/validation/frax_msfram_comparison_holdout_set.csv", row.names = FALSE)
}

# -------------------------- Cross Validation Training Set Evaluation --------------------------
for(n_r in 1:n_folds){
  # Get training and validation split indices for current cross validation fold
  train_index <- zz1[zz2 != n_r]
  
  # Extract true fracture ground truth of current fold's validation subset
  current_test_y <- tbl_orig_test[,n1:n2]
  
  # Calculate start and end row index to extract corresponding MSFRAM predictions from full cross validation prediction matrix
  r0 <- (n_r - 1) * nrow(tbl_orig_test) + 1
  r1 <- n_r * nrow(tbl_orig_test)
  current_msfram_pred <- pred_dnn_mlrusmote_set[r0:r1, ]
  
  # Assign predictions to reserved position in full training dataset
  data_final3_train[-train_index, 6:9] <- current_msfram_pred
}

if(TRUE){
  # Vectorized calculation of max and mean aggregated major fracture prediction scores
  data_final3_train <- data_final3_train %>%
    rowwise() %>%
    mutate(
      max_pred = max(c_across(6:9)),
      mean_pred = mean(c_across(6:9))
    ) %>% ungroup()
  
  # -------------------------- Hip Fracture Prediction Performance Comparison --------------------------
  cat("\n=== Hip Fracture Prediction Performance (Cross Validation Training Set) ===\n")
  roc_frax_hip_train <- pROC::roc(data_final3_train$hip_fx, data_final3_train$frax_hip, ci = TRUE, quiet = TRUE)
  cat("FRAX Hip Fracture Prediction AUC:", round(roc_frax_hip_train$auc, 3), 
      " | 95% CI: [", round(roc_frax_hip_train$ci, 3), ",", round(roc_frax_hip_train$ci, 3), "]\n")
  
  roc_ml_hip_train <- pROC::roc(data_final3_train$hip_fx, data_final3_train$hip_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Hip Fracture Prediction AUC:", round(roc_ml_hip_train$auc, 3), 
      " | 95% CI: [", round(roc_ml_hip_train$ci, 3), ",", round(roc_ml_hip_train$ci, 3), "]\n")
  
  delong_hip_train <- roc.test(roc_frax_hip_train, roc_ml_hip_train, paired = TRUE)
  cat("DeLong Test for Hip Fracture AUC Comparison: p-value =", round(delong_hip_train$p.value, 5), "\n")
  print(delong_hip_train)

  # -------------------------- Major Fracture Performance Comparison --------------------------
  cat("\n=== Major Osteoporotic Fracture Prediction Performance (Cross Validation Training Set) ===\n")
  roc_frax_major_train <- pROC::roc(data_final3_train$major_fx, data_final3_train$frax_major, ci = TRUE, quiet = TRUE)
  cat("FRAX Major Fracture Prediction AUC:", round(roc_frax_major_train$auc, 3), 
      " | 95% CI: [", round(roc_frax_major_train$ci, 3), ",", round(roc_frax_major_train$ci, 3), "]\n")
  
  # Evaluate max aggregation strategy
  roc_ml_major_max_train <- pROC::roc(data_final3_train$major_fx, data_final3_train$max_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Major Fracture Prediction AUC (Max Aggregation):", round(roc_ml_major_max_train$auc, 3), 
      " | 95% CI: [", round(roc_ml_major_max_train$ci, 3), ",", round(roc_ml_major_max_train$ci, 3), "]\n")
  
  delong_major_max_train <- roc.test(roc_frax_major_train, roc_ml_major_max_train, paired = TRUE)
  cat("DeLong Test for Major Fracture AUC Comparison (Max Aggregation): p-value =", round(delong_major_max_train$p.value, 5), "\n")
  print(delong_major_max_train)
  
  # Evaluate mean aggregation strategy
  roc_ml_major_mean_train <- pROC::roc(data_final3_train$major_fx, data_final3_train$mean_pred, ci = TRUE, quiet = TRUE)
  cat("MSFRAM Major Fracture Prediction AUC (Mean Aggregation):", round(roc_ml_major_mean_train$auc, 3), 
      " | 95% CI: [", round(roc_ml_major_mean_train$ci, 3), ",", round(roc_ml_major_mean_train$ci, 3), "]\n")
  
  delong_major_mean_train <- roc.test(roc_frax_major_train, roc_ml_major_mean_train, paired = TRUE)
  cat("DeLong Test for Major Fracture AUC Comparison (Mean Aggregation): p-value =", round(delong_major_mean_train$p.value, 5), "\n")
  print(delong_major_mean_train)
  
  # Export full cross validation evaluation results to local file
  write.csv(data_final3_train, file = "output/validation/frax_msfram_comparison_cross_validation_set.csv", row.names = FALSE)
}

