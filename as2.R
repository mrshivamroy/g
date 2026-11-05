1.
cat("Enter marks of 5 subjects:\n")
sub1 <- as.numeric(readline(prompt = "Sub 1: "))
sub2 <- as.numeric(readline(prompt = "Sub 2: "))
sub3 <- as.numeric(readline(prompt = "Sub 3: "))
sub4 <- as.numeric(readline(prompt = "Sub 4: "))
sub5 <- as.numeric(readline(prompt = "Sub 5: "))

marks <- c(sub1, sub2, sub3, sub4, sub5)

total_marks <- sum(marks)
avg <- mean(marks)
percentage <- (total_marks / 500) * 100

cat("Total marks: ", total_marks, "\n")
cat("Avg marks: ", avg, "\n")
cat("Percentage of marks: ", percentage, "\n")


2.
calculate_salary <- function(basic_pay) {
  hra <- 0.20 * basic_pay
  da <- 0.15 * basic_pay
  tax <- 0.10 * basic_pay

  gross <- basic_pay + hra + da
  net <- gross - tax

  return(list(
    "Basic Pay:" = basic_pay,
    "HRA: " = hra,
    "DA: " = da,
    "Tax: " = tax,
    "Gross Salary: " = gross,
    "Net Salary: " = net
  ))
}

s <- calculate_salary(62400)
print(s)


3.
km <- 2L
m <- km * 1000L
cm <- m * 100L
dm <- m * 10L

ct <- data.frame(
  Unit = c("KM", "M", "CM", "DM"),
  Distance = c(km, m, cm, dm)
)

print(ct)


4.
inr <- 5000
usdr <- 0.010503
eurr <- 0.0091
gbpr <- 0.007804

usd <- inr * usdr
euro <- inr * eurr
gbp <- inr * gbpr

currency_table <- data.frame(
  Currency = c("US Dollar (USD)", "Euro (EUR)", "British Pound (GBP)"),
  Rate = c(usdr, eurr, gbpr),
  Converted_Amount = c(usd, euro, gbp)
)

print(currency_table)


5.
units_consumed <- 350
rate_per_unit <- 6.50

total_bill <- units_consumed * rate_per_unit

cat("Total Electricity Bill: ", total_bill, "\n")


6.
weight_kg <- 70
height_m <- 1.75

bmi <- weight_kg / (height_m^2)

print(round(bmi, 1))


7.
num <- -5

if (num > 0) {
  print("Positive")
} else if (num < 0) {
  print("Negative")
} else {
  print("Zero")
}


8.
a <- 15
b <- 42
c <- 23

largest <- max(a, b, c)
print(largest)


9.
age <- 20

if (age >= 18) {
  print("Eligible to Vote")
} else {
  print("Not Eligible to Vote")
}


10.
cat("Enter marks for 5 subjects separated by spaces (Press Enter twice):\n")
marks <- scan()

if (length(marks) != 5) {
  stop("Error: Please enter exactly 5 marks.")
}

if (any(marks < 35)) {
  result <- "Fail (Failed in one or more subjects)"
} else {
  total_marks <- sum(marks)
  percentage <- (total_marks / 500) * 100

  cat("\nTotal Marks:", total_marks, "/ 500\n")
  cat("Percentage:", round(percentage, 2), "%\n")

  if (percentage >= 75) {
    result <- "Distinction"
  } else if (percentage >= 60) {
    result <- "First Class"
  } else if (percentage >= 50) {
    result <- "Second Class"
  } else {
    result <- "Pass"
  }
}

cat("Final Result Classification:", result, "\n")
