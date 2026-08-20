library(dplyr)

#' Calculate Imbalance Ratio per Label (IRLbl)
#'
#' Computes the imbalance ratio for each label in multi-label datasets,
#' which is defined as the maximum number of positive samples across all labels
#' divided by the number of positive samples of the current label.
#' @param labels Binary 0/1 matrix, rows represent samples, columns represent labels
#' @return Named numeric vector of IRLbl values for each label
IRLbl <- function(labels) {
  # Validate input label matrix
  if (!is.matrix(labels) && !is.data.frame(labels)) {
    stop("Input 'labels' must be a matrix or data frame with only 0/1 values")
  }
  if (!all(as.matrix(labels) %in% c(0, 1))) {
    stop("Input 'labels' can only contain 0 or 1 values")
  }
  
  pos_nums_per_label <- colSums(labels)
  max_pos_nums <- max(pos_nums_per_label)
  # Replace labels with zero positive samples with 1 to avoid division by zero error
  pos_nums_per_label[pos_nums_per_label == 0] <- 1
  return(max_pos_nums / pos_nums_per_label)
}

#' Calculate Mean Imbalance Ratio (MeanIR)
#'
#' Computes the average value of all label-specific imbalance ratios (IRLbl)
#' @param labels Binary 0/1 matrix, rows represent samples, columns represent labels
#' @return Numeric scalar of MeanIR value
MeanIR <- function(labels) {
  irlbl_value <- IRLbl(labels)
  return(mean(irlbl_value))
}

#' Identify Tail (Minority) Labels
#'
#' Selects labels whose imbalance ratio is greater than the MeanIR, which are
#' defined as the minority/long-tailed labels in the multi-label dataset
#' @param y Binary 0/1 label matrix/data frame
#' @return Character vector of tail label names
get_tail_label <- function(y) {
  if (is.null(colnames(y))) {
    colnames(y) <- paste0("label_", seq_len(ncol(y)))
  }
  columns <- colnames(y)
  pos_counts <- colSums(y)
  irpl <- max(pos_counts) / pos_counts
  mir <- mean(irpl)
  tail_label <- columns[irpl > mir]
  
  if (length(tail_label) == 0) {
    warning("No tail labels identified: all labels have imbalance ratio <= MeanIR")
  }
  return(tail_label)
}

#' Get Index List of Samples with Positive Labels for Each Tail Label
#'
#' Returns a list where each element stores the row indices of samples that have
#' the corresponding tail label marked as positive (value = 1)
#' @param y Binary 0/1 label matrix/data frame
#' @return List of integer vectors, each element corresponds to one tail label
get_index <- function(y) {
  tail_labels <- get_tail_label(y)
  index <- lapply(tail_labels, function(lbl) which(y[, lbl] == 1))
  names(index) <- tail_labels
  return(index)
}

#' Get Indices of Samples with Zero Labels
#'
#' Returns row indices of samples that have no positive labels (sum of all labels = 0)
#' @param y Binary 0/1 label matrix/data frame
#' @return Integer vector of zero-label sample indices
get_index_zero <- function(y) {
  s_df <- rowSums(y)
  index <- which(s_df == 0)
  return(index)
}

#' Get Indices of Samples with At Least One Positive Label
#'
#' Returns row indices of samples that have sum of all labels greater than 0
#' @param y Binary 0/1 label matrix/data frame
#' @return Integer vector of non-zero label sample indices
get_index_nonzero <- function(y) {
  s_df <- rowSums(y)
  index <- which(s_df != 0)
  return(index)
}

#' Extract Minority Label Corresponding Feature and Label Subsets
#'
#' Returns paired feature and label subsets only containing samples with
#' positive tail labels
#' @param X Feature matrix/data frame
#' @param y Binary 0/1 label matrix/data frame
#' @return List with two elements: subset of minority features, subset of minority labels
get_minority_instace <- function(X, y) {
  if (nrow(X) != nrow(y)) {
    stop("Number of rows of X and y must be consistent")
  }
  index <- get_index(y)
  X_sub <- X[unlist(index), , drop = FALSE]
  y_sub <- y[unlist(index), , drop = FALSE]
  return(list(X_sub, y_sub))
}

#' Compute k Nearest Neighbors Indices for All Samples
#'
#' Uses Euclidean distance to find the k nearest neighbors for each sample,
#' excluding the sample itself
#' @param X Numeric feature matrix/data frame (only continuous features for distance calculation)
#' @param k Number of nearest neighbors to return
#' @return Numeric matrix with nrow(X) rows and k columns, stores neighbor indices
nearest_neighbour <- function(X, k) {
  if (k < 1 || k != as.integer(k)) {
    stop("k must be a positive integer greater than or equal to 1")
  }
  if (k >= nrow(X)) {
    stop("k cannot be greater than or equal to total number of samples")
  }
  
  m_dist <- as.matrix(dist(X))
  diag(m_dist) <- Inf
  indices <- t(apply(m_dist, 1, order)[1:k, ])
  return(indices)
}

#' Filter Valid Seed Samples for Synthetic Sample Generation
#'
#' Selects qualified minority samples as seed points, excluding samples whose
#' k-neighbors contain too many zero-label noise samples
#' @param X Feature matrix/data frame
#' @param y Binary 0/1 label matrix/data frame
#' @param k Number of nearest neighbors used for validation
#' @return List of seed sample indices for each tail label
get_seed_index <- function(X, y, k) {
  s_df <- rowSums(y)
  index_zero <- which(s_df == 0)
  
  sample_knn <- nearest_neighbour(X, k)
  index_minor <- get_index(y)
  
  index_seed <- lapply(seq_along(index_minor), function(i) {
    index_minor_one <- index_minor[[i]]
    keep_mask <- vapply(index_minor_one, function(a0) {
      a1 <- sample_knn[a0, ]
      b0 <- length(intersect(a1, index_zero))
      # Mark as valid seed only if the number of all-zero neighbors <= 2k/3
      b0 <= (2 * k / 3)
    }, FUN.VALUE = logical(1))
    index_minor_one[keep_mask]
  })
  names(index_seed) <- names(index_minor)
  
  empty_seed_check <- vapply(index_seed, length, FUN.VALUE = integer(1)) == 0
  if (any(empty_seed_check)) {
    warning(paste0("Tail labels [", paste(names(index_seed)[empty_seed_check], collapse = ", "), "] have no valid seed samples, no synthetic samples will be generated for these labels"))
  }
  return(index_seed)
}

#' Calculate Synthetic Sample Generation Threshold for Each Tail Label
#'
#' Computes how many new synthetic samples need to be generated for each tail label
#' based on the oversampling ratio parameter nsr
#' @param y Binary 0/1 label matrix/data frame
#' @param nsr Numeric oversampling ratio, controls target positive sample count relative to majority label
#' @return Integer vector of required synthetic sample counts for each tail label
get_sample_thr <- function(y, nsr) {
  if (nsr <= 0) {
    stop("nsr (oversampling ratio) must be a positive number")
  }
  
  num_label <- colSums(y)
  max_num <- max(num_label)
  sample_thr <- floor(max_num * nsr - num_label)
  
  tail_label <- get_tail_label(y)
  sample_thr <- sample_thr[names(num_label) %in% tail_label]
  sample_thr[sample_thr < 0] <- 0
  
  return(sample_thr)
}

#' ML-RUS-SMOTE Multi-Label Imbalance Processing Algorithm
#'
#' Combined algorithm that performs synthetic minority oversampling on tail labels
#' and random undersampling on all-zero noise samples to balance the multi-label dataset
#' @param X Feature matrix/data frame
#' @param y Binary 0/1 label matrix/data frame
#' @param nsr Numeric oversampling ratio for minority labels
#' @param usr Numeric undersampling ratio for all-zero noise samples
#' @param k Number of nearest neighbors used for SMOTE generation
#' @param f_cont Integer/character vector of indices/names of continuous features
#' @param f_cat Integer/character vector of indices/names of categorical features
#' @return List with two elements: balanced final feature set, balanced final label set
ML_RUSMOTE <- function(X, y, nsr, usr, k, f_cont, f_cat) {
  # Full input validation
  if (!is.data.frame(X) && !is.matrix(X)) stop("X must be a matrix or data frame")
  if (!is.data.frame(y) && !is.matrix(y)) stop("y must be a matrix or data frame")
  if (nrow(X) != nrow(y)) stop("Number of rows of X and y must match")
  if (usr < 0) stop("usr (undersampling ratio) cannot be negative")
  
  sample_knn <- nearest_neighbour(X[, f_cont, drop = FALSE], k)
  sample_seed <- get_seed_index(X, y, k)
  
  tail_label <- get_tail_label(y)
  label_minor <- which(colnames(y) %in% tail_label)
  if (length(label_minor) == 0) {
    warning("No tail labels found, no oversampling will be performed")
  }
  
  sample_thr <- get_sample_thr(y, nsr)
  num_minor_lb <- length(sample_thr)
  total_synthetic <- sum(sample_thr)
  
  # Preallocate empty result frames with proper column types
  X_new <- if (total_synthetic > 0) {
    X[seq_len(total_synthetic), , drop = FALSE]
  } else {
    X[0, , drop = FALSE]
  }
  y_new <- if (total_synthetic > 0) {
    y[seq_len(total_synthetic), , drop = FALSE]
  } else {
    y[0, , drop = FALSE]
  }
  
  j <- 0
  sum_new_y <- rep(0, num_minor_lb)
  
  # Perform synthetic sample generation
  for (i in seq_len(num_minor_lb)) {
    current_seeds <- sample_seed[[i]]
    required_samples <- sample_thr[i]
    if (length(current_seeds) == 0 || required_samples <= 0) next
    
    while (sum_new_y[i] < required_samples) {
      j <- j + 1
      seed_one <- sample(current_seeds, 1)
      neighbor_k <- sample_knn[seed_one, ]
      neighbor_one <- sample(neighbor_k, 1)
      
      nn_df <- y[neighbor_k, , drop = FALSE]
      sum_col_k <- colSums(nn_df)
      
      # Assign label set for new synthetic sample
      y_new[j, ] <- y[seed_one, ]
      y_new[j, which(sum_col_k > (k / 2))] <- 1
      
      # Assign continuous features for new synthetic sample
      ratio <- runif(1)
      gap <- X[neighbor_one, f_cont, drop = FALSE] - X[seed_one, f_cont, drop = FALSE]
      X_new[j, f_cont] <- X[seed_one, f_cont] + ratio * gap
      
      # Assign categorical features for new synthetic sample
      for (nf in seq_along(f_cat)) {
        r0 <- f_cat[nf]
        # Combine seed sample feature, neighbor sample feature and all k neighbors' features
        nk0 <- c(X[seed_one, r0], X[neighbor_k, r0])
        nk1 <- table(nk0)
        X_new[j, r0] <- as.numeric(names(which.max(nk1)))
      }
      
      sum_new_y <- colSums(y_new[, label_minor, drop = FALSE])
    }
  }
  
  # Perform random undersampling on all-zero label samples
  index_zero <- get_index_zero(y)
  non_zero_count <- nrow(y) - length(index_zero)
  # Ensure undersampling count does not exceed total number of all-zero samples
  under_sample <- min(length(index_zero), floor(non_zero_count * usr))
  u_s <- if (under_sample > 0) sample(index_zero, under_sample) else integer(0)
  
  X_zero_us <- X[u_s, , drop = FALSE]
  y_zero_us <- y[u_s, , drop = FALSE]
  
  index_nonzero <- get_index_nonzero(y)
  X_nonzero <- X[index_nonzero, , drop = FALSE]
  y_nonzero <- y[index_nonzero, , drop = FALSE]
  
  # Combine all parts to get final balanced dataset
  X_final <- dplyr::bind_rows(X_zero_us, X_nonzero, X_new[1:j, , drop = FALSE])
  y_final <- dplyr::bind_rows(y_zero_us, y_nonzero, y_new[1:j, , drop = FALSE])
  
  rownames(X_final) <- NULL
  rownames(y_final) <- NULL
  
  return(list(X_final, y_final))
}
