#setWD to current file
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#load packages
pkgs <- c(
  "tidyverse",
  "caret"
)
lapply(pkgs, library, character.only=TRUE)

##### Load Datasets

# Original Data
#binary criterion == diabetes_binary$Diabetes_binary (badly balanced)
#pred == all others (all binary)
diabetes_binary <- read.csv("..\\data\\raw_data\\original_data\\diabetes_binary.csv")
PRED_diabetes_binary <- diabetes_binary %>% select(-Diabetes_binary)
CRIT_diabetes_binary <- diabetes_binary %>% select(Diabetes_binary)
NROW_diabetes_binary <- nrow(diabetes_binary)
NCOL_diabetes_binary <- ncol(diabetes_binary)

# Anonymized Data
diabetes_binary_anon <- read.csv("..\\data\\raw_data\\anonymized_data\\anonymized_diabetes_binary.csv")
PRED_diabetes_binary_anon <- diabetes_binary_anon %>% select(-Diabetes_binary)
CRIT_diabetes_binary_anon <- diabetes_binary_anon %>% select(Diabetes_binary)

assertthat::are_equal(ncol(PRED_diabetes_binary)+1, NCOL_diabetes_binary)
assertthat::assert_that(all(table(diabetes_binary$Diabetes_binary) > 10000)) 
assertthat::assert_that(NCOL_diabetes_binary > 20)

assertthat::assert_that(nrow(diabetes_binary) == nrow(diabetes_binary_anon))
assertthat::assert_that(ncol(diabetes_binary) == ncol(diabetes_binary_anon))

#loopvars
N_features <- c(10, 15, 20)
N_rows <- c(1000, 5000, 10000)
reps <- 2

for (rep in 1:reps){
  for(N_feature in N_features) {
    for(N_row in N_rows){
      
      # N_feature = 10
      # N_row = 1000
      
      # Derive Row Indices -------------------------------------------------------
      
      index_0 <- which(CRIT_diabetes_binary == 0)
      index_1 <- which(CRIT_diabetes_binary == 1)
      
      sampled_indices_0 <- sample(index_0, ceiling(N_row/2))
      sampled_indices_1 <- sample(index_1, ceiling(N_row/2))
      
      balanced_sampled_indices <- c(sampled_indices_0, sampled_indices_1) %>% sample()
      
      # Derive Col Indices -------------------------------------------------------
      
      sampled_col_indices <- sample(c(1:ncol(PRED_diabetes_binary)), N_feature, replace=FALSE)
      
      # Reduce Data --------------------------------------------------------------
      
      original_crit_reduced <- CRIT_diabetes_binary[balanced_sampled_indices, ]
      original_pred_reduced <- PRED_diabetes_binary[balanced_sampled_indices, sampled_col_indices]
      
      anonymized_crit_reduced <- CRIT_diabetes_binary_anon[balanced_sampled_indices, ]
      anonymized_pred_reduced <- PRED_diabetes_binary_anon[balanced_sampled_indices, sampled_col_indices]
      
      class(original_pred_reduced)
      lapply(anonymized_pred_reduced, class)
      
      # One hot encode -----------------------------------------------------------
      
      string_columns <- sapply(anonymized_pred_reduced, is.character)
      string_columns <- names(string_columns[string_columns])
      
      numeric_data <- anonymized_pred_reduced[setdiff(names(anonymized_pred_reduced), string_columns)]
      
      dummy_data <- dummyVars(formula = ~ ., data = anonymized_pred_reduced[, string_columns], fullRank = TRUE)
      dummy_data <- data.frame(predict(dummy_data, newdata = anonymized_pred_reduced))
      
      # Combine the numeric data with the one-hot encoded data
      anonymized_pred_reduced <- cbind(numeric_data, dummy_data)

      # Save Data to Corresponding Folder ----------------------------------------
      
      folder_name_original <- paste0(N_row,"r_",N_feature,"f")
      
      # Original
      write.csv(
        original_pred_reduced, 
        file = paste0("..\\data\\original_training_data\\", folder_name_original, "\\X_reduced_diabetes_binary", rep, ".csv"),
        row.names = FALSE
      )
      
      write.csv(
        data.frame(y = original_crit_reduced), 
        file = paste0("..\\data\\original_training_data\\", folder_name_original, "\\y_reduced_diabetes_binary", rep, ".csv"),
        row.names = FALSE
      )
      
      # Anonymized
      write.csv(
        anonymized_pred_reduced, 
        file = paste0("..\\data\\anonymized_training_data\\", folder_name_original, "\\X_reduced_diabetes_binary_anon", rep, ".csv"),
        row.names = FALSE
      )
      
      write.csv(
        data.frame(y = anonymized_crit_reduced), 
        file = paste0("..\\data\\anonymized_training_data\\", folder_name_original, "\\y_reduced_diabetes_binary_anon", rep, ".csv"),
        row.names = FALSE
      )
    }
  }
  
}


