cat("Jupyter Notebook: An interactive web-based interface that allows you to combine code, text, and visualizations in a single document.\n")
cat("NumPy: A foundational Python library used for numerical computing and working with large arrays/matrices.\n")
cat("SciPy: A Python library built on NumPy used for scientific and technical computing.\n")
cat("Statsmodels: A Python module used for estimating statistical models and performing statistical tests.\n")
cat("Pandas: A popular Python library for data manipulation and analysis, offering data structures like DataFrames.\n")
cat("RStudio: The primary Integrated Development Environment (IDE) for R, which makes writing R code, viewing plots, and managing workspaces much easier.\n")

cat("readr: Fast and friendly way to read rectangular data (like CSV files).\n")
cat("readxl: Used to import data from Excel spreadsheets (.xls and .xlsx).\n")
cat("dplyr: A grammar of data manipulation, providing consistent verbs to filter, select, and modify data.\n")
cat("tidyr: Tools to clean and organize data into a tidy format.\n")
cat("ggplot2: A powerful system for declaratively creating graphics and data visualizations.\n")
cat("psych: Functions for multivariate analysis and descriptive statistics, heavily used in psychology research.\n")

library(readr)

my_data <- read_csv(file.choose(), comment = "#")

head(my_data)
str(my_data)
summary(my_data)

A <- matrix(c(4, 2, 7, 6), nrow = 2, byrow = TRUE)
B <- matrix(c(1, 3, 5, 2), nrow = 2, byrow = TRUE)

print("Matrix A:")
print(A)

print("Matrix B:")
print(B)

print("1. Matrix Addition:")
print(A + B)

print("2. Matrix Subtraction:")
print(A - B)

print("3. Matrix Multiplication:")
print(A %*% B)

print("4. Transpose of Matrix A:")
print(t(A))

print("5. Inverse of Matrix A:")
print(solve(A))

print("6. Determinant of Matrix A:")
print(det(A))

print("7. Rank of Matrix A:")
print(qr(A)$rank)

eigen_A <- eigen(A)

print("8. Eigenvalues of Matrix A:")
print(eigen_A$values)

print("9. Eigenvectors of Matrix A:")
print(eigen_A$vectors)

sepal_len <- iris$Sepal.Length

get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mean_absolute_deviation <- function(v) {
  mean(abs(v - mean(v)))
}

cat("1. Mean:", mean(sepal_len), "\n")
cat("2. Median:", median(sepal_len), "\n")
cat("3. Mode:", get_mode(sepal_len), "\n")
cat("4. Minimum:", min(sepal_len), "\n")
cat("5. Maximum:", max(sepal_len), "\n")
cat("6. Range:", range(sepal_len)[1], "to", range(sepal_len)[2], "\n")
cat("7. Interquartile Range (IQR):", IQR(sepal_len), "\n")
cat("8. Mean Absolute Deviation:", mean_absolute_deviation(sepal_len), "\n")
cat("9. Variance:", var(sepal_len), "\n")
cat("10. Standard Deviation:", sd(sepal_len), "\n")
