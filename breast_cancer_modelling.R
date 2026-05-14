# -----------------------------------------------------
# Predictive Modelling of Breast Cancer Diagnosis
# -------- Wisconsin Breast Cancer Dataset --------
# -----------------------------------------------------

# 1. Load packages

library(caret)
library(ggplot2)
library(scales)
library(tidyr)
library(randomForest)
library(pROC)
library(corrplot)
library(kernlab)
library(forcats)

# 2. Create output folders

dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# 3. Import dataset

# Import dataset and treat "?" as missing values
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
wbc_data <- read.csv(url, header = FALSE, na.strings = "?")

# Inspect the dataset structure before preprocessing
str(wbc_data)
summary(wbc_data)
head(wbc_data)

# 4. Assign variable names

# Assign variable names from the UCI documentation
colnames(wbc_data) <- c(
  "ID",
  "Clump.thickness",
  "Cell.size",
  "Cell.shape",
  "Marginal.adhesion",
  "Single.epithelial.cell.size",
  "Bare.nuclei",
  "Bland.chromatin",
  "Normal.nucleoli",
  "Mitoses",
  "Class"
)

# 5. Pre-process data

# Remove non-predictive ID and recode variables for classification
wbc_data$ID <- NULL
wbc_data$Bare.nuclei <- as.numeric(wbc_data$Bare.nuclei)
wbc_data$Class <- ifelse(wbc_data$Class == 2, "benign", "malignant")
wbc_data$Class <- factor(wbc_data$Class, levels = c("benign", "malignant"))

# Check missing values before imputation
missing_before <- colSums(is.na(wbc_data))
missing_before

# Impute missing Bare.nuclei values using the median
wbc_data$Bare.nuclei[is.na(wbc_data$Bare.nuclei)] <-
  median(wbc_data$Bare.nuclei, na.rm = TRUE)

# Check missing values after imputation
missing_after <- colSums(is.na(wbc_data))
missing_after

# Check class distribution
class_distribution <- table(wbc_data$Class)
class_distribution
prop.table(class_distribution)

# 6. Create train-test split

# Create a stratified train-test split for reproducibility
set.seed(123)

train_index <- createDataPartition(
  wbc_data$Class,
  p = 0.7,
  list = FALSE
)

train_data <- wbc_data[train_index, ]
test_data <- wbc_data[-train_index, ]

# 7. Standardise predictors for SVM

train_scaled <- train_data
test_scaled <- test_data

# Standardise predictors using training data parameters to avoid leakage
scaling_values <- scale(train_data[, -ncol(train_data)])

train_scaled[, -ncol(train_scaled)] <- scaling_values

test_scaled[, -ncol(test_scaled)] <- scale(
  test_data[, -ncol(test_data)],
  center = attr(scaling_values, "scaled:center"),
  scale = attr(scaling_values, "scaled:scale")
)

# 8. Correlation analysis

# Compute correlation matrix on predictors only
cor_matrix <- cor(train_data[, -ncol(train_data)])

# View and plot matrix
round(cor_matrix, 2)

png("outputs/figures/correlation_matrix.png", width = 1000, height = 800)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  order = "hclust",
  tl.col = "black",
  tl.cex = 0.8,
  tl.srt = 45,
  addCoef.col = "black",
  number.cex = 0.6,
  col = colorRampPalette(c("firebrick", "white", "steelblue"))(200),
  title = "Correlation Matrix of Predictor Variables",
  mar = c(0, 0, 2, 0)
)

dev.off()

# 9. Evaluation functions

# Calculate classification metrics from a confusion matrix
evaluate_model <- function(cm) {
  
  true_pos <- cm["malignant", "malignant"]
  false_neg <- cm["benign", "malignant"]
  true_neg <- cm["benign", "benign"]
  false_pos <- cm["malignant", "benign"]
  
  accuracy <- (true_pos + true_neg) / sum(cm)
  sensitivity <- true_pos / (true_pos + false_neg)
  specificity <- true_neg / (true_neg + false_pos)
  precision <- true_pos / (true_pos + false_pos)
  f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))
  
  return(c(
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Precision = precision,
    F1_Score = f1_score
  ))
}

# Generate confusion matrix and apply shared evaluation function
get_metrics <- function(predicted, actual) {
  
  cm <- table(
    Predicted = predicted,
    Actual = actual
  )
  
  print(cm)
  evaluate_model(cm)
}

# 10. Cross-validation settings

# Define common 5-fold cross-validation settings
cv_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE
)

# 11. Train models

# Train logistic regression
set.seed(123)

cv_log <- train(
  Class ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  trControl = cv_control
)

# Train radial SVM
set.seed(123)

cv_svm <- train(
  Class ~ .,
  data = train_scaled,
  method = "svmRadial",
  trControl = cv_control,
  tuneLength = 5
)

# Train random forest
set.seed(123)

cv_rf <- train(
  Class ~ .,
  data = train_data,
  method = "rf",
  tuneGrid = data.frame(mtry = c(2, 3, 4)),
  ntree = 500,
  importance = TRUE,
  trControl = cv_control
)

# 12. Test-set predictions

# Generate logistic regression class predictions from the trained caret model
pred_log_class <- predict(cv_log, test_data, type = "raw")
pred_log_prob <- predict(cv_log, test_data, type = "prob")[, "malignant"]

log_metrics <- get_metrics(pred_log_class, test_data$Class)

# Generate SVM predictions
pred_svm_class <- predict(cv_svm, test_scaled, type = "raw")
pred_svm_prob <- predict(cv_svm, test_scaled, type = "prob")[, "malignant"]

svm_metrics <- get_metrics(pred_svm_class, test_data$Class)

# Generate random forest predictions
pred_rf_class <- predict(cv_rf, test_data, type = "raw")
pred_rf_prob <- predict(cv_rf, test_data, type = "prob")[, "malignant"]

rf_metrics <- get_metrics(pred_rf_class, test_data$Class)

# 13. Compare model metrics

# Combine test-set metrics for model comparison
results <- data.frame(
  Model = c("Logistic Regression", "Support Vector Machine", "Random Forest"),
  rbind(log_metrics, svm_metrics, rf_metrics)
)

results[, -1] <- round(results[, -1], 3)
results

write.csv(results, "outputs/model_metrics.csv", row.names = FALSE)

# 14. Class distribution plot

# Prepare class distribution data for visualisation
class_df <- as.data.frame(table(wbc_data$Class))
colnames(class_df) <- c("Class", "Count")

class_df$Class <- factor(
  class_df$Class,
  levels = c("benign", "malignant"),
  labels = c("Benign", "Malignant")
)

class_df$Percentage <- round((class_df$Count / sum(class_df$Count)) * 100, 1)
class_df$Label <- paste0(class_df$Count, " (", class_df$Percentage, "%)")

# Plot class distribution
class_plot <- ggplot(class_df, aes(x = Class, y = Count, fill = Class)) +
  geom_col(width = 0.55, colour = "grey30", linewidth = 0.3) +
  geom_text(aes(label = Label), vjust = -0.5, size = 4) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Class Distribution",
    subtitle = "Wisconsin Breast Cancer Dataset",
    x = "Diagnosis Class",
    y = "Number of Observations"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

class_plot

ggsave(
  filename = "outputs/figures/class_distribution.png",
  plot = class_plot,
  width = 7,
  height = 5
)

# 15. Model performance plot

# Reshape model metrics into long format for plotting
results_long <- pivot_longer(
  results,
  cols = c(Accuracy, Sensitivity, Specificity, Precision, F1_Score),
  names_to = "Metric",
  values_to = "Score"
)

results_long$Metric <- factor(
  results_long$Metric,
  levels = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Score"),
  labels = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1-score")
)

results_long$Model <- factor(
  results_long$Model,
  levels = c("Logistic Regression", "Support Vector Machine", "Random Forest")
)

# Plot comparative model performance
performance_plot <- ggplot(
  results_long,
  aes(x = Model, y = Score, fill = Metric)
) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Model Performance Comparison",
    subtitle = "Test-set performance across classification metrics",
    x = "Model",
    y = "Score",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1),
    panel.grid.minor = element_blank()
  )

performance_plot

ggsave(
  filename = "outputs/figures/model_performance.png",
  plot = performance_plot,
  width = 8,
  height = 5
)

# 16. Cross-validation accuracy

# Extract cross-validation accuracy for comparison with test performance
cv_results <- data.frame(
  Model = c("Logistic Regression", "Support Vector Machine", "Random Forest"),
  CV_Accuracy = c(
    max(cv_log$results$Accuracy),
    max(cv_svm$results$Accuracy),
    max(cv_rf$results$Accuracy)
  )
)

cv_results$CV_Accuracy <- round(cv_results$CV_Accuracy, 3)
cv_results

write.csv(cv_results, "outputs/cross_validation_accuracy.csv", row.names = FALSE)

# 17. ROC curves and AUC

# Generate ROC curves using predicted probabilities
roc_log <- roc(
  response = test_data$Class,
  predictor = pred_log_prob,
  levels = c("benign", "malignant"),
  direction = "<"
)

roc_svm <- roc(
  response = test_data$Class,
  predictor = pred_svm_prob,
  levels = c("benign", "malignant"),
  direction = "<"
)

roc_rf <- roc(
  response = test_data$Class,
  predictor = pred_rf_prob,
  levels = c("benign", "malignant"),
  direction = "<"
)

# Summarise AUC values numerically
auc_results <- data.frame(
  Model = c("Logistic Regression", "Support Vector Machine", "Random Forest"),
  AUC = c(
    as.numeric(auc(roc_log)),
    as.numeric(auc(roc_svm)),
    as.numeric(auc(roc_rf))
  )
)

auc_results$AUC <- round(auc_results$AUC, 3)
auc_results

write.csv(auc_results, "outputs/auc_results.csv", row.names = FALSE)

# Add AUC values to ROC curve labels
roc_labels <- c(
  paste0("Logistic Regression (AUC = ", auc_results$AUC[1], ")"),
  paste0("SVM (AUC = ", auc_results$AUC[2], ")"),
  paste0("Random Forest (AUC = ", auc_results$AUC[3], ")")
)

roc_list <- list(roc_log, roc_svm, roc_rf)
names(roc_list) <- roc_labels

# Plot ROC curves for threshold-independent comparison
roc_plot <- ggroc(
  roc_list,
  legacy.axes = TRUE
) +
  geom_line(linewidth = 1.0) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title = "ROC Curves",
    subtitle = "Comparison of classifier discrimination performance",
    x = "False Positive Rate",
    y = "True Positive Rate",
    colour = "Model"
  ) +
  theme_minimal()

roc_plot

ggsave(
  filename = "outputs/figures/roc_curves.png",
  plot = roc_plot,
  width = 8,
  height = 5
)

# 18. Random forest variable importance

# Assess random forest variable importance
rf_importance <- varImp(cv_rf)
rf_importance

importance_df <- data.frame(
  Variable = rownames(rf_importance$importance),
  Importance = rf_importance$importance[, 1]
)

importance_df$Variable <- gsub("\\.", " ", importance_df$Variable)

importance_plot <- ggplot(
  importance_df,
  aes(x = reorder(Variable, Importance), y = Importance)
) +
  geom_col(width = 0.65) +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance",
    subtitle = "Relative importance of predictors in the random forest model",
    x = "Predictor",
    y = "Importance"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

importance_plot

ggsave(
  filename = "outputs/figures/random_forest_variable_importance.png",
  plot = importance_plot,
  width = 8,
  height = 5
)

# 19. Save model objects

saveRDS(cv_log, "outputs/logistic_regression_model.rds")
saveRDS(cv_svm, "outputs/svm_model.rds")
saveRDS(cv_rf, "outputs/random_forest_model.rds")

# 20. Session information

sessionInfo()
