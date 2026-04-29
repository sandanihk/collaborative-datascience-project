#' @title Preparing data for Discriminant Function Analysis
#' @description This function splits a dataset into training and testing data, and also optionally standardizes the predictor variables, when the scaling is used
#' ,the centering and the scaling parameters are estimated from the training data and then aopplied to both training and test data after splitting to avoid data leakage.
#' @param data a data frame
#' @param target_column_name A character string stating the categorical response variable name
#' @param training_propotion_size The proportion of data used for training.
#' @param scale A logical value (TRUE or FALSE); if TRUE, the numeric predictors are centered and scaled
#' @param seed A random seed for reproducibility of results
#' @keywords discriminant-function-analysis
#' @return A list containing:
#' \item{training_data}{The processed training data.}
#' \item{test_data}{The processed test data.}
#' \item{preprocess_object}{The preprocessing object used for scaling. NULL if scale = FALSE.}
#' @export
#' @examples
#' prepare_data(data = iris, target_column_name = "Species", training_propotion_size = 0.8)
#' prepare_data(data = iris, target_column_name = "Species",training_propotion_size = 0.8, scale = FALSE)

prepare_data <- function(data, target_column_name, training_propotion_size, scale = TRUE, seed = 123) {

  set.seed(seed)

  # First removing missing values in the data
  data <- tidyr::drop_na(data)

  # Keeping the target variable as a factor
  data[[target_column_name]] <- as.factor(data[[target_column_name]])

  # Then splitting the dataset to Train and test sets
  train_index <- caret::createDataPartition(
    data[[target_column_name]],
    p = training_propotion_size,
    list = FALSE
  )

  training_set_raw <- data[train_index, ]
  test_set_raw  <- data[-train_index, ]

  # After that cleaning the unused factor levels in the target variable for both train and test sets

  training_set_raw[[target_column_name]] <- droplevels(training_set_raw[[target_column_name]])
  test_set_raw[[target_column_name]]  <- droplevels(test_set_raw[[target_column_name]])

  # Scaling the variable values if the user provided the scale argument is TRUE.
  # Here, Caret's preProcess function is used to center and scale the numeric predictors based on the training data,
  # and then apply the same transformations to the test data.

  if (scale) {

    training_predictors <- dplyr::select(
      training_set_raw,
      -dplyr::all_of(target_column_name)
    )

    test_predictors <- dplyr::select(
      test_set_raw,
      -dplyr::all_of(target_column_name)
    )

    preprocess_object <- caret::preProcess(
      training_predictors,
      method = c("center", "scale")
    )

    training_set_scaled <- predict(preprocess_object, training_predictors)
    test_set_scaled <- predict(preprocess_object, test_predictors)

    final_training_data <- training_set_scaled
    final_training_data[[target_column_name]] <- training_set_raw[[target_column_name]]

    final_test_data <- test_set_scaled
    final_test_data[[target_column_name]] <- test_set_raw[[target_column_name]]

  } else {
    final_training_data <- training_set_raw
    final_test_data  <- test_set_raw
    preprocess_object <- NULL
  }

  return(list(
    training_data = final_training_data,
    test_data = final_test_data,
    preprocess_object = preprocess_object
  ))
}

#' @title Run Linear Discriminant Analysis (LDA) with Cross-Validation and Visualization
#'
#' @description
#' Fits a Linear Discriminant Analysis (LDA) model, evaluates its classification
#' performance on test data, performs Leave-One-Out Cross-Validation (LOOCV) on
#' the training data, and creates an LD1 vs. LD2 visualization when at least two
#' discriminant axes are available.
#'
#' @param training_data A data frame containing the training data.
#' @param test_data A data frame containing the test data.
#' @param target_column_name A character string specifying the categorical response variable name.
#'
#' @keywords lda visualization cross-validation classification
#'
#' @return A list containing:
#' \item{model}{The fitted LDA model.}
#' \item{predictions}{The prediction object returned by predict().}
#' \item{confusion_matrix}{A caret confusion matrix object.}
#' \item{accuracy}{The test set accuracy.}
#' \item{cv_model}{The LOOCV model output.}
#' \item{cv_accuracy}{The Leave-One-Out Cross-Validation accuracy.}
#' \item{coefficients}{The LDA scaling coefficients.}
#' \item{plot}{An LD1 vs. LD2 ggplot object if available. NULL for two-class problems.}
#'
#' @export
#'
#' @examples
#' prepared <- prepare_data(iris, "Species", training_propotion_size = 0.7)
#'
#' lda_result <- run_lda_model(
#'   training_data = prepared$training_data,
#'   test_data = prepared$test_data,
#'   target_column_name = "Species"
#' )
#'
#' lda_result$accuracy
#' lda_result$cv_accuracy
#' lda_result$plot


run_lda_model <- function(training_data, test_data, target_column_name) {

  # Creating the formula for the model
  formula <- stats::as.formula(paste(target_column_name, "~ ."))

  # Fitting the LDA model
  model <- MASS::lda(formula, data = training_data)

  # Leave-One-Out Cross-Validation on training data
  cv_model <- MASS::lda(
    formula,
    data = training_data,
    CV = TRUE
  )

  cv_accuracy <- mean(cv_model$class == training_data[[target_column_name]])

  # Predict on test data
  predictions <- predict(model, newdata = test_data)

  # Accuracy evaluation using confusion matrix
  confusion_matrix <- caret::confusionMatrix(
    predictions$class,
    test_data[[target_column_name]]
  )

  accuracy <- confusion_matrix$overall["Accuracy"]

  # Visualization
  lda_values <- predict(model, newdata = training_data)
  lda_df <- as.data.frame(lda_values$x)
  lda_df[[target_column_name]] <- training_data[[target_column_name]]

  if (ncol(lda_values$x) >= 2) {

    plot <- ggplot2::ggplot(
      lda_df,
      ggplot2::aes(x = LD1, y = LD2, color = .data[[target_column_name]])
    ) +
      ggplot2::geom_point(alpha = 0.6, size = 0.8) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "LDA Visualization (LD1 vs LD2)",
        x = "Linear Discriminant 1",
        y = "Linear Discriminant 2",
        color = target_column_name
      )

  } else {
    plot <- NULL
  }

  return(list(
    model = model,
    predictions = predictions,
    confusion_matrix = confusion_matrix,
    accuracy = accuracy,
    cv_model = cv_model,
    cv_accuracy = cv_accuracy,
    coefficients = model$scaling,
    plot = plot
  ))
}


#' @title Run Quadratic Discriminant Analysis (QDA) with Cross-Validation
#'
#' @description
#' Fits a Quadratic Discriminant Analysis (QDA) model, evaluates its classification
#' performance on test data, and performs Leave-One-Out Cross-Validation (LOOCV)
#' on the training data. Unlike LDA, QDA allows each class to have its own
#' covariance structure, which enables more flexible decision boundaries.
#'
#' @param training_data A data frame containing the training data.
#' @param test_data A data frame containing the test data.
#' @param target_column_name A character string specifying the categorical response variable.
#'
#' @keywords qda classification cross-validation
#'
#' @return A list containing:
#' \item{model}{The fitted QDA model.}
#' \item{predictions}{The prediction object returned by predict().}
#' \item{confusion_matrix}{A caret confusion matrix object.}
#' \item{accuracy}{The test set accuracy.}
#' \item{cv_model}{The LOOCV model output.}
#' \item{cv_accuracy}{The Leave-One-Out Cross-Validation accuracy.}
#'
#' @export
#'
#' @examples
#' prepared <- prepare_data(iris, "Species", training_propotion_size = 0.7)
#'
#' qda_result <- run_qda_model(
#'   training_data = prepared$training_data,
#'   test_data = prepared$test_data,
#'   target_column_name = "Species"
#' )
#'
#' qda_result$accuracy
#' qda_result$cv_accuracy

run_qda_model <- function(training_data, test_data, target_column_name) {

  # Creating the formula for the model
  formula <- stats::as.formula(paste(target_column_name, "~ ."))

  # Fitting the QDA model
  model <- MASS::qda(formula, data = training_data)

  # Leave-One-Out Cross-Validation on training data
  cv_model <- MASS::qda(
    formula,
    data = training_data,
    CV = TRUE
  )

  cv_accuracy <- mean(cv_model$class == training_data[[target_column_name]])

  # Predict on test data
  predictions <- predict(model, newdata = test_data)

  # Accuracy evaluation using confusion matrix
  confusion_matrix <- caret::confusionMatrix(
    data = predictions$class,
    reference = test_data[[target_column_name]]
  )

  accuracy <- confusion_matrix$overall["Accuracy"]

  return(list(
    model = model,
    predictions = predictions,
    confusion_matrix = confusion_matrix,
    accuracy = accuracy,
    cv_model = cv_model,
    cv_accuracy = cv_accuracy
  ))
}

#' @title Run Flexible Discriminant Analysis (FDA) with Cross-Validation
#'
#' @description
#' Fits a Flexible Discriminant Analysis model, evaluates its classification
#' performance on test data, and performs k-fold cross-validation on the
#' training data. FDA is a more flexible extension of discriminant analysis
#' that can model more complex relationships between predictors and class
#' membership.
#'
#' @param training_data A data frame containing the training data.
#' @param test_data A data frame containing the test data.
#' @param target_column_name A character string specifying the categorical response variable name.
#' @param cv_folds Number of cross-validation folds. Default is 10.
#' @param seed An integer used to ensure reproducibility
#' @keywords fda classification cross-validation
#'
#' @return A list containing:
#' \item{model}{The fitted FDA model.}
#' \item{predictions}{The predicted class labels.}
#' \item{confusion_matrix}{A caret confusion matrix object.}
#' \item{accuracy}{The test set accuracy.}
#' \item{cv_model}{The caret cross-validation model output.}
#' \item{cv_accuracy}{The cross-validation accuracy.}

#' @export
#'
#' @examples
#' prepared <- prepare_data(iris, "Species", training_propotion_size = 0.7)
#'
#' fda_result <- run_fda_model(
#'   training_data = prepared$training_data,
#'   test_data = prepared$test_data,
#'   target_column_name = "Species"
#' )
#'
#' fda_result$accuracy
#' fda_result$cv_accuracy


run_fda_model <- function(training_data, test_data, target_column_name, cv_folds = 10, seed = 123) {

  set.seed(seed)

  formula <- stats::as.formula(paste(target_column_name, "~ ."))

  # Fit FDA model on full training data
  model <- mda::fda(formula, data = training_data)

  # Create folds
  folds <- caret::createFolds(
    training_data[[target_column_name]],
    k = cv_folds,
    list = TRUE
  )

  fold_accuracy <- numeric(length(folds))

  for (i in seq_along(folds)) {

    valid_index <- folds[[i]]

    cv_train <- training_data[-valid_index, ]
    cv_valid <- training_data[valid_index, ]

    cv_model <- mda::fda(formula, data = cv_train)

    cv_predictions <- predict(cv_model, newdata = cv_valid)

    fold_accuracy[i] <- mean(
      as.factor(cv_predictions) == cv_valid[[target_column_name]]
    )
  }

  cv_accuracy <- mean(fold_accuracy)

  # Predict on test data
  predictions <- predict(model, newdata = test_data)

  confusion_matrix <- caret::confusionMatrix(
    data = as.factor(predictions),
    reference = test_data[[target_column_name]]
  )

  accuracy <- confusion_matrix$overall["Accuracy"]

  return(list(
    model = model,
    predictions = predictions,
    confusion_matrix = confusion_matrix,
    accuracy = accuracy,
    cv_accuracy = cv_accuracy,
    cv_fold_accuracy = fold_accuracy
  ))
}

#' @title Plotting a confusion matrix heatmap
#' @description
#' Creates a heatmap visualization from a caret confusionMatrix object.
#' The diagonal cells represent correct classifications, while off-diagonal
#' cells represent misclassifications
#' @param confusion_matrix a caret confusionMatrix object created using caret::confusionMatrix()
#' @param title the plot title
#' @keywords heatmap visualization confusion-matrix
#' @return A ggplot object
#' @export
#' @examples
#' # Example using iris dataset
#' cm <- caret::confusionMatrix(
#'   data = iris$Species,
#'   reference = iris$Species
#' )
#'
#' plot_confusion_matrix(cm, title = "Example Confusion Matrix")

plot_confusion_matrix <- function(confusion_matrix, title = "Confusion Matrix Heatmap") {
  conf_df <- as.data.frame(confusion_matrix$table)

  ggplot2::ggplot(conf_df, ggplot2::aes(x = Reference, y = Prediction, fill = Freq)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = Freq), size = 3) +
    ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    ) +
    ggplot2::labs(
      title = title,
      x = "True Class",
      y = "Predicted Class"
    )
}

#' Spotify Dataset
#'
#' A dataset containing Spotify track features used for discriminant analysis.
#'
#' @format A data frame with multiple variables:
#' \describe{
#'   \item{acousticness}{Numeric (0–1)}
#'   \item{album_name}{Character}
#'   \item{artists}{Character}
#'   \item{danceability}{Numeric (0–1)}
#'   \item{duration_ms}{Numeric}
#'   \item{energy}{Numeric (0–1)}
#'   \item{explicit}{Logical}
#'   \item{instrumentalness}{Numeric (0–1)}
#'   \item{key}{Integer}
#'   \item{liveness}{Numeric}
#'   \item{loudness}{Numeric (dB)}
#'   \item{mode}{Integer}
#'   \item{popularity}{Integer (0–100)}
#'   \item{speechiness}{Numeric (0–1)}
#'   \item{tempo}{Numeric}
#'   \item{time_signature}{Integer}
#'   \item{track_genre}{Character}
#'   \item{track_id}{Character}
#'   \item{track_name}{Character}
#'   \item{valence}{Numeric}
#' }
#'
"spotify_data"

#' LazyData: true
#' LazyDataCompression: xz
#' VignetteBuilder: quarto
#' Suggests: quarto
