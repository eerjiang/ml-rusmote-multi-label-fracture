# Data preprocessing script: split dataset into 80% training set and 20% hold-out set, perform standardization
# Load required dependent packages
library(caret)
library(dplyr)
library(mldr)

# Load original raw dataset
data <- read.csv("NHANES/NHANES dataset.csv", header = TRUE, sep = ",")

# Delete Id column
data <- data[,-1]

# Define number of features and labels
n_features <- 10 # number of features
n_labels <- 3 # number of labels
n_features_cont <- 9 # number of continuous features
n1 <- n_features + 1
n2 <- n_features + n_labels
n3 <- n_features_cont + 1


# Calculate total number of fracture sites per sample
data$fx_count <- apply(data[,n1:n2], 1, sum)

# Define stratification variable to preserve label distribution consistency after splitting
stratified_var <- "fx_count"


# Set fixed random seed to ensure 100% reproducibility of data splitting results
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)

# Perform stratified random sampling to split dataset
# 80% of samples are assigned to training set, the remaining 20% are assigned to independent test/validation set
dev_index <- createDataPartition(
  y = data[[stratified_var]],  # Stratify based on the number of fracture sites per sample
  p = 0.8,                     # Proportion of training set in total samples
  times = 1,                   # Run one-time split instead of multiple repeated splits
  list = FALSE                 # Return split result in matrix format instead of nested list
)


# Generate training set and validation set based on split index
dev_set <- data[dev_index, ]
val_set  <- data[-dev_index, ]

# Helper function: calculate percentage distribution of specified stratification variable in input dataset
prop_table <- function(df) prop.table(table(df[[stratified_var]])) * 100

# Verify stratification effect to check if distribution of fracture site counts is consistent between two sets
cat("Training set class distribution of fracture site counts:\n"); print(round(prop_table(dev_set), 1))
cat("\nValidation set class distribution of fracture site counts:\n"); print(round(prop_table(val_set), 1))

# Print total positive sample count for each label in training set
cat("\nPer-label positive sample count in training set:\n")
print(apply(dev_set[,n1:ncol(dev_set)], 2, sum))
# Print total positive sample count for each label in validation set
cat("\nPer-label positive sample count in validation set:\n")
print(apply(val_set[,n1:ncol(val_set)], 2, sum))


# -------------------------- Training Set Standardization Processing --------------------------
if(TRUE){
  # Remove the auxiliary fx_count column which is only used for stratification
  tbl_orig <- dev_set[, -ncol(dev_set)]
  
  # Convert all categorical feature columns to numeric data type
  for(i in n3:n_features){
    tbl_orig[, i] <- as.numeric(tbl_orig[, i])
  }
  
  # Convert feature data frame to matrix format for faster arithmetic operation
  tbl_orig_matrix <- as.matrix(tbl_orig)
  
  # Perform Min-Max normalization on the continuous numerical features (scale values to range [0,1])
  # Calculate maximum, minimum and value difference of each continuous feature based on training set only
  train_feature_max <- apply(tbl_orig_matrix[, 1:n_features_cont], 2, max)
  train_feature_min <- apply(tbl_orig_matrix[, 1:n_features_cont], 2, min)
  train_feature_range <- train_feature_max - train_feature_min
  
  # Add protection for features with zero value range (all samples share the same value)
  train_feature_range[train_feature_range == 0] <- 1
  
  # Execute normalization calculation
  for(i in 1:n3){
    tbl_orig_matrix[, i] <- (tbl_orig_matrix[, i] - train_feature_min[i]) / train_feature_range[i]
  }
  
  # Split processed training data into independent feature subset and label subset
  tbl_orig_x <- tbl_orig_matrix[, 1:total_feature_num]
  tbl_orig_y <- tbl_orig_matrix[, c(total_feature_num + 1):ncol(tbl_orig_matrix)]
}

# Export fully processed training set data to local file
write.csv(as.data.frame(tbl_orig_matrix), file = "tbl_orig_1.csv", row.names = FALSE)


# -------------------------- Hold-out Set Standardization Processing --------------------------
if(TRUE){
  # Remove redundant fx_count auxiliary column from original hold-out set
  val_set_1 <- val_set %>% select(-fx_count)
  
  # Convert all categorical feature columns to numeric data type
  for(i in n3:n_features){
    val_set_1[, i] <- as.numeric(val_set_1[, i])
  }
  
  # Convert validation set data frame to matrix format for faster arithmetic operation
  val_set_1_matrix <- as.matrix(val_set_1)
  
  # Execute Min-Max normalization on validation set using maximum/minimum values calculated from TRAINING SET only
  # This prevents data leakage which will lead to over-optimistic evaluation results
  for(i in 1:n_features_cont){
    val_set_1_matrix[, i] <- (val_set_1_matrix[, i] - train_feature_min[i]) / train_feature_range[i]
  }
  
  # Split processed hold-out data into independent feature subset and standard mldr multi-label format dataset
  val_set_1_x <- val_set_1_matrix[, 1:n_features]
  val_set_1 <- as.data.frame(val_set_1_matrix)
  val_set_2 <- mldr_from_dataframe(val_set_1, labelIndices = c(n1:ncol(val_set_1)), name = "testMLDR")
}

# Export fully processed hold-out set data to local file
write.csv(val_set_1, file = "val_set_1.csv", row.names = FALSE)
