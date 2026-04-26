#' Prepare Data for Discriminant Analysis
#'
#' Splits a dataset into training and testing sets and optionally standardizes predictors.
#'
#' @param data A data frame.
#' @param target Character string giving the name of the categorical outcome variable.
#' @param train_prop Proportion of data used for training. Default is 0.7.
#' @param scale Logical. If TRUE, numeric predictors are centered and scaled.
#' @param seed Random seed for reproducibility.
#'
#' @return A list containing train_data, test_data, and preprocess_obj.
#' @export

#first function - For preparing data for discriminant analysis

prepare_disc_data <- function(data, target, train_prop, scale = TRUE, seed = 123) {
  set.seed(seed)

  data <- data |>
    tidyr::drop_na()

  data[[target]] <- as.factor(data[[target]])

  train_index <- caret::createDataPartition(
    data[[target]],
    p = train_prop,
    list = FALSE
  )

  train_raw <- data[train_index, ]
  test_raw  <- data[-train_index, ]

  train_raw[[target]] <- droplevels(train_raw[[target]])
  test_raw[[target]]  <- droplevels(test_raw[[target]])

  if (scale) {
    preprocess_obj <- caret::preProcess(
      train_raw |> dplyr::select(-dplyr::all_of(target)),
      method = c("center", "scale")
    )

    train_scaled <- predict(
      preprocess_obj,
      train_raw |> dplyr::select(-dplyr::all_of(target))
    )

    test_scaled <- predict(
      preprocess_obj,
      test_raw |> dplyr::select(-dplyr::all_of(target))
    )

    train_data <- train_scaled
    train_data[[target]] <- train_raw[[target]]

    test_data <- test_scaled
    test_data[[target]] <- test_raw[[target]]
  } else {
    preprocess_obj <- NULL
    train_data <- train_raw
    test_data <- test_raw
  }

  return(list(
    train_data = train_data,
    test_data = test_data,
    preprocess_obj = preprocess_obj
  ))
}

#' Run Linear Discriminant Analysis with Visualization
#'
#' Fits an LDA model, evaluates it on test data, and generates a 2D visualization
#' of the first two linear discriminants (LD1 and LD2).
#'
#' @param train_data A data frame containing training data.
#' @param test_data A data frame containing test data.
#' @param target A character string specifying the target (categorical) variable name.
#'
#' @return A list containing:
#' \item{model}{The fitted LDA model}
#' \item{predictions}{Predicted class labels for test data}
#' \item{confusion_matrix}{Confusion matrix object}
#' \item{accuracy}{Test set accuracy}
#' \item{coefficients}{LDA scaling coefficients}
#' \item{plot}{ggplot object of LD1 vs LD2}
#'
#' @examples
#' result <- run_lda_model(train_data, test_data, "genre_group")
#' result$accuracy
#' result$plot
#'
#' @export

# Running LDA model with visualization of LD1 vs LD2

run_lda_model <- function(train_data, test_data, target) {

  # Create formula
  formula <- stats::as.formula(paste(target, "~ ."))

  # Fit model
  model <- MASS::lda(formula, data = train_data)

  # Predict on test data
  pred <- predict(model, newdata = test_data)

  # Confusion matrix
  conf <- caret::confusionMatrix(
    pred$class,
    test_data[[target]]
  )

  # Accuracy
  acc <- conf$overall["Accuracy"]

  # -----------------------------
  # LDA Visualization (LD1 vs LD2)
  # -----------------------------

  lda_values <- predict(model, newdata = train_data)

  lda_df <- as.data.frame(lda_values$x)
  lda_df[[target]] <- train_data[[target]]

  plot <- ggplot2::ggplot(
    lda_df,
    ggplot2::aes(x = LD1, y = LD2, color = .data[[target]])
  ) +
    ggplot2::geom_point(alpha = 0.6, size = 0.8) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "LDA Visualization (LD1 vs LD2)",
      x = "Linear Discriminant 1",
      y = "Linear Discriminant 2",
      color = target
    )

  # Return all outputs
  return(list(
    model = model,
    predictions = pred,
    confusion_matrix = conf,
    accuracy = acc,
    coefficients = model$scaling,
    plot = plot
  ))
}


#' Run Quadratic Discriminant Analysis
#'
#' Fits a QDA model and evaluates it on test data.
#'
#' @param train_data Training data frame.
#' @param test_data Test data frame.
#' @param target Character string giving the outcome variable.
#'
#' @return A list containing model, predictions, confusion matrix, and accuracy.
#' @export

# 3rd function for running QDA model

run_qda_model <- function(train_data, test_data, target) {
  formula <- stats::as.formula(paste(target, "~ ."))

  model <- MASS::qda(formula, data = train_data)

  pred <- predict(model, newdata = test_data)

  conf <- caret::confusionMatrix(
    pred$class,
    test_data[[target]]
  )

  return(list(
    model = model,
    predictions = pred,
    confusion_matrix = conf,
    accuracy = conf$overall["Accuracy"]
  ))
}

#' Run Flexible Discriminant Analysis
#'
#' Fits an FDA model and evaluates it on test data.
#'
#' @param train_data Training data frame.
#' @param test_data Test data frame.
#' @param target Character string giving the outcome variable.
#'
#' @return A list containing model, predictions, confusion matrix, and accuracy.
#' @export

# 4th function for running FDA model

run_fda_model <- function(train_data, test_data, target) {
  formula <- stats::as.formula(paste(target, "~ ."))

  model <- mda::fda(formula, data = train_data)

  pred <- predict(model, newdata = test_data)

  conf <- caret::confusionMatrix(
    as.factor(pred),
    test_data[[target]]
  )

  return(list(
    model = model,
    predictions = pred,
    confusion_matrix = conf,
    accuracy = conf$overall["Accuracy"]
  ))
}

#' Plot Confusion Matrix Heatmap
#'
#' Creates a heatmap from a caret confusionMatrix object.
#'
#' @param conf_mat A caret confusionMatrix object.
#' @param title Plot title.
#'
#' @return A ggplot object.
#' @export

# 5th function for plotting confusion matrix heatmap

plot_confusion_matrix <- function(conf_mat, title = "Confusion Matrix Heatmap") {
  conf_df <- as.data.frame(conf_mat$table)

  ggplot2::ggplot(conf_df, ggplot2::aes(x = Reference, y = Prediction, fill = Freq)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = Freq), size = 3) +
    ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = title,
      x = "True Class",
      y = "Predicted Class"
    )
}

