#setWD to current file
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#load packages
pkgs <- c(
  "tidyverse"
)
lapply(pkgs, library, character.only=TRUE)

##### Load Datasets

#binary criterion == naticusDROID$Result (well balanced)
#pred == all others (all binary)
naticusDROID <- read.csv("..\\data\\training_data\\original_data\\naticusDROID.csv")
PRED_naticusDROID <- naticusDROID %>% select(-Result)
CRIT_naticusDROID <- naticusDROID %>% select(Result)
NROW_naticusDROID <- nrow(naticusDROID)
NCOL_naticusDROID <- ncol(naticusDROID)

assertthat::are_equal(ncol(PRED_naticusDROID)+1, NCOL_naticusDROID)
assertthat::assert_that(all(table(naticusDROID$Result) > 10000)) 
assertthat::assert_that(NCOL_naticusDROID > 20)


#binary criterion == diabetes_binary$Diabetes_binary (badly balanced)
#pred == all others (all binary)
diabetes_binary <- read.csv("..\\data\\training_data\\original_data\\diabetes_binary.csv")
PRED_diabetes_binary <- diabetes_binary %>% select(-Diabetes_binary)
CRIT_diabetes_binary <- diabetes_binary %>% select(Diabetes_binary)
NROW_diabetes_binary <- nrow(diabetes_binary)
NCOL_diabetes_binary <- ncol(diabetes_binary)

assertthat::are_equal(ncol(PRED_diabetes_binary)+1, NCOL_diabetes_binary)
assertthat::assert_that(all(table(diabetes_binary$Diabetes_binary) > 10000)) 
assertthat::assert_that(NCOL_diabetes_binary > 20)

#hypervars
PRED <- PRED_diabetes_binary
CRIT <- CRIT_diabetes_binary
NAME <- "diabetes_binary"


#loopvars
N_features <- c(10, 15, 20)
N_rows <- c(1000, 5000, 10000)

for(N_feature in N_features) {
  for(N_row in N_rows){
    
    # Derive Row Indices -------------------------------------------------------
    
    index_0 <- which(CRIT == 0)
    index_1 <- which(CRIT == 1)
    
    sampled_indices_0 <- sample(index_0, ceiling(N_row/2))
    sampled_indices_1 <- sample(index_1, ceiling(N_row/2))
    
    balanced_sampled_indices <- c(sampled_indices_0, sampled_indices_1) %>% sample()
    
    # Derive Col Indices -------------------------------------------------------
    
    sampled_col_indices <- sample(c(1:ncol(PRED)), N_feature, replace=FALSE)
    
    # Reduce Data --------------------------------------------------------------
    
    crit_reduced <- CRIT[balanced_sampled_indices, ]
    pred_reduced <- PRED[balanced_sampled_indices, sampled_col_indices]
    
    # Save Data to Corresponding Folder ----------------------------------------
    
    folder_name <- paste0(N_row,"r_",N_feature,"f")
    
    write.csv(
      pred_reduced, 
      file = paste0("..\\data\\training_data\\", folder_name, "\\X_reduced_", NAME, ".csv"),
      row.names = FALSE
    )
    
    write.csv(
      data.frame(y = crit_reduced), 
      file = paste0("..\\data\\training_data\\", folder_name, "\\y_reduced_", NAME, ".csv"),
      row.names = FALSE
    )
  }
}
