library(dplyr)
library(keras3)
library(mldr)
library(utiml)
library(purrr)

# Import pre-written ML-RUSMOTE algorithm functions
source("ML_RUSMOTE_Function.R")


# -------------------------- Global Hyperparameter Configuration --------------------------
# ML-DNN network layer unit settings
u1 <- 256
u2 <- 128
u3 <- 64
# Batch size for DNN training
bs <- 128
# Proportion of validation split during DNN training
vs <- 0.1
# Number of cross-validation folds
n_folds <- 3
# Fixed random seed for full reproducibility
GLOBAL_SEED <- 100

# Define the candidate parameter search grid to find optimal values
# Adjust this list to expand your parameter search range
param_grid <- expand.grid(
  nsr = c(0.3, 0.4, 0.5, 0.6, 0.7),
  usr = c(0.5, 0.75, 1, 1.25, 2),
  smote_k = 5,
  mlknn_k = c(2, 3, 4, 5),
  stringsAsFactors = FALSE
)

# Initialize storage structures for evaluation metrics and prediction results
metric_names <- NULL
me_br_set_list <- list()
me_mlknn_set_list <- list()
me_dnn_set_list <- list()
full_pred_results <- list()

# Cross-validation partition pre-allocation
n <- nrow(tbl_orig)
zz1 <- 1:n
zz2 <- rep(1:n_folds, ceiling(n/n_folds))[1:n]
set.seed(GLOBAL_SEED)
zz2 <- sample(zz2, n)

# -------------------------- Auto Tuning & Cross Validation Core Logic --------------------------
for (param_idx in 1:nrow(param_grid)) {
  current_params <- param_grid[param_idx, ]
  cat("\n=== Running parameter combination", param_idx, "/", nrow(param_grid), 
      " | nsr:", current_params$nsr, "usr:", current_params$usr, 
      "mlknn_k:", current_params$mlknn_k, " ===\n")
  
  # Initialize metric storage matrix for current parameter group
  me_br_mlrusmote_set <- matrix(NA, nrow = 22, ncol = n_folds)
  me_mlknn_mlrusmote_set <- matrix(NA, nrow = 22, ncol = n_folds)
  me_dnn_mlrusmote_set <- matrix(NA, nrow = 22, ncol = n_folds)
  
  pred_br_mlrusmote_set <- c()
  pred_mlknn_mlrusmote_set <- c()
  pred_dnn_mlrusmote_set <- c()
  
  # Run each fold of cross validation
  for(n_r in 1:n_folds) {
    set.seed(GLOBAL_SEED + n_r)
    train_index <- zz1[zz2 != n_r]
    
    # Preprocess training and test datasets to mldr format
    tbl_orig_train <- as.data.frame(tbl_orig_2[train_index, ])
    tbl_orig_train <- mldr_from_dataframe(tbl_orig_train, labelIndices = c(n1:n2), name = "trainMLDR")
    tbl_orig_test <- as.data.frame(tbl_orig_test <- as.data.frame(tbl_orig_2[-train_index, ]))
    tbl_orig_test <- mldr_from_dataframe(tbl_orig_test, labelIndices = c(n1:n2), name = "testMLDR")
    
    # Execute ML-RUSMOTE oversampling + undersampling processing
    set.seed(123)
    tbl_orig_train_x <- tbl_orig_train$dataset[,1:n_features]
    tbl_orig_train_y <- tbl_orig_train$dataset[,n1:n2]
    a1 <- ML_RUS_SMOTE(
      X = as.matrix(tbl_orig_train_x),
      y = as.matrix(tbl_orig_train_y),
      nsr = current_params$nsr,
      usr = current_params$usr,
      k = current_params$smote_k,
      f_cont = c(1:n_features_cont),
      f_cat = c(n3:n_features)
    )
    
    # Verify imbalance ratio after resampling
    tbl_mlrusmote_train_0 <- as.data.frame(cbind(a1[[1]], a1[[2]]))
    processed_irlbl <- IRLbl(tbl_mlrusmote_train_0[,n1:n2])
    processed_meanir <- MeanIR(tbl_mlrusmote_train_0[,n1:n2])
    cat("Fold", n_r, " | Post-resample MeanIR:", round(processed_meanir, 3), "\n")
    
    tbl_mlrusmote_train_2 <- mldr_from_dataframe(tbl_mlrusmote_train_0, labelIndices = c(n1:n2), name = "trainMLDR")
    
    # Prepare formatted DNN training and test input data
    tbl_mlrusmote_train_x <- as.matrix(tbl_mlrusmote_train_2$dataset[,1:n_features])
    tbl_mlrusmote_train_y <- as.matrix(tbl_mlrusmote_train_2$dataset[,n1:n2])
    tbl_orig_test_x <- as.matrix(tbl_orig_test$dataset[,1:n_features])
    tbl_orig_test_y <- as.matrix(tbl_orig_test$dataset[,n1:n2])
    
    # -------------------------- Binary Relevance (BR) Model Training & Evaluation --------------------------
    model_br_mlrusmote <- br(tbl_mlrusmote_train_2)
    pred_br_mlrusmote <- predict(model_br_mlrusmote, tbl_orig_test)
    
    # Store prediction results
    pred_br_mlrusmote_set <- if (is.null(dim(pred_br_mlrusmote_set))) as.data.frame(pred_br_mlrusmote) else bind_rows(pred_br_mlrusmote_set, as.data.frame(pred_br_mlrusmote))
    
    # Calculate multi-label evaluation metrics
    me_br_mlrusmote <- multilabel_evaluate(tbl_orig_test, pred_br_mlrusmote)
    me_br_mlrusmote_set[,n_r] <- me_br_mlrusmote
    
    # Save metric names for first iteration
    if (is.null(metric_names)) metric_names <- names(me_br_mlrusmote)
    
    # -------------------------- ML-kNN Model Training & Evaluation --------------------------
    model_mlknn_mlrusmote <- mlknn(tbl_mlrusmote_train_2, k = current_params$mlknn_k)
    pred_mlknn_mlrusmote <- predict(model_mlknn_mlrusmote, tbl_orig_test)
    
    # Store prediction results
    pred_mlknn_mlrusmote_set <- if (is.null(dim(pred_mlknn_mlrusmote_set))) as.data.frame(pred_mlknn_mlrusmote) else bind_rows(pred_mlknn_mlrusmote_set, as.data.frame(pred_mlknn_mlrusmote))
    
    # Calculate multi-label evaluation metrics
    me_mlknn_mlrusmote <- multilabel_evaluate(tbl_orig_test, pred_mlknn_mlrusmote)
    me_mlknn_mlrusmote_set[,n_r] <- me_mlknn_mlrusmote
    
    # -------------------------- ML-DNN Deep Model Training & Evaluation --------------------------
    feature_dim <- dim(tbl_mlrusmote_train_x)
    label_dim <- dim(tbl_mlrusmote_train_y)
    
    # Construct sequential DNN model
    model_dnn_mlrusmote <- keras_model_sequential()  %>%
      layer_dense(units = u1, activation = 'relu', input_shape = feature_dim) %>%
      layer_dropout(rate = 0.1) %>%
      layer_dense(units = u2, activation = 'relu') %>%
      layer_dense(units = u3, activation = 'relu') %>%
      layer_dense(units = label_dim, activation = 'sigmoid') # Use sigmoid for multi-label independent classification
    
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
    
    # Evaluate on test set and generate predictions
    results <- model_dnn_mlrusmote %>% evaluate(tbl_orig_test_x, tbl_orig_test_y, verbose = 0)
    pred_dnn_mlrusmote <- model_dnn_mlrusmote %>% predict(tbl_orig_test_x, verbose = 0)
    
    # Store prediction results
    pred_dnn_mlrusmote_set <- if (is.null(dim(pred_dnn_mlrusmote_set))) as.data.frame(pred_dnn_mlrusmote) else bind_rows(pred_dnn_mlrusmote_set, as.data.frame(pred_dnn_mlrusmote))
    
    # Format predictions to mldr result standard and calculate metrics
    pred_dnn_mlrusmote <- as.mlresult(pred_dnn_mlrusmote)
    me_dnn_mlrusmote <- multilabel_evaluate(tbl_orig_test, pred_dnn_mlrusmote)
    me_dnn_mlrusmote_set[,n_r] <- me_dnn_mlrusmote
    
    # Clean up temporary Keras session to avoid memory leak
    gc()
    rm(model_dnn_mlrusmote)
    keras3::k_clear_session()
    
    cat("Fold", n_r, "completed\n")
  }
  
  # Save results for current parameter combination
  rownames(me_br_mlrusmote_set) <- metric_names
  rownames(me_mlknn_mlrusmote_set) <- metric_names
  rownames(me_dnn_mlrusmote_set) <- metric_names
  
  me_br_set_list[[param_idx]] <- me_br_mlrusmote_set
  me_mlknn_set_list[[param_idx]] <- me_mlknn_mlrusmote_set
  me_dnn_set_list[[param_idx]] <- me_dnn_mlrusmote_set
  
  full_pred_results[[param_idx]] <- list(
    params = current_params,
    br_pred = pred_br_mlrusmote_set,
    mlknn_pred = pred_mlknn_mlrusmote_set,
    dnn_pred = pred_dnn_mlrusmote_set
  )
}

# -------------------------- Post-processing: Calculate Optimal Parameters & Export Results --------------------------
# Helper function to calculate cross-validation mean and standard deviation for all parameter combinations
get_summarized_metrics <- function(metric_list, param_grid, metric_names) {
  map_dfr(1:length(metric_list), function(i) {
    current_mat <- metric_list[[i]]
    metric_mean <- round(apply(current_mat, 1, mean), 3)
    metric_sd <- round(apply(current_mat, 1, sd), 3)
    res_df <- data.frame(t(metric_mean), t(metric_sd))
    colnames(res_df) <- paste0(rep(metric_names, 2), c(rep("_mean", length(metric_names)), rep("_sd", length(metric_names))))
    cbind(param_grid[i, ], res_df)
  })
}

# Generate full metric summary table for all models
br_summary <- get_summarized_metrics(me_br_set_list, param_grid, metric_names)
mlknn_summary <- get_summarized_metrics(me_mlknn_set_list, param_grid, metric_names)
dnn_summary <- get_summarized_metrics(me_dnn_set_list, param_grid, metric_names)

# Find optimal parameter set based on selected target metric (we use F1-score as default here)
best_br_params <- br_summary[which.max(br_summary$f1_mean), ]
best_mlknn_params <- mlknn_summary[which.max(mlknn_summary$f1_mean), ]
best_dnn_params <- dnn_summary[which.max(dnn_summary$f1_mean), ]

# Print optimal parameter results
cat("\n=== Optimal Parameters Found ===\n")
cat("Best BR Model Params:\n")
print(best_br_params)
cat("\nBest ML-kNN Model Params:\n")
print(best_mlknn_params)
cat("\nBest DNN Model Params:\n")
print(best_dnn_params)

# Export all training results to corresponding output files
write.csv(br_summary, file = "output/training/me_br_mlrusmote_full_summary.csv")
write.csv(mlknn_summary, file = "output/training/me_mlknn_mlrusmote_full_summary.csv")
path_name <- paste0("output/training/me_dnn_mlrusmote_set_", u1, "_", u2, "_", u3, "_", bs, "_", vs, ".csv")
write.csv(dnn_summary, file = path_name)

# Export separate optimal parameter result files
write.csv(best_br_params, file = "output/training/best_br_mlrusmote_params.csv")
write.csv(best_mlknn_params, file = "output/training/best_mlknn_mlrusmote_params.csv")
write.csv(best_dnn_params, file = "output/training/best_dnn_mlrusmote_params.csv")
