cat("Q1\n")
x <- 0:12
pmf <- dbinom(x, size = 12, prob = 0.25)
cdf <- pbinom(x, size = 12, prob = 0.25)
par(mfrow = c(1, 2))
plot(x, pmf, type = "h", lwd = 2, col = "steelblue", main = "PMF of Binomial(12, 0.25)", xlab = "x", ylab = "P(X = x)")
points(x, pmf, pch = 16, col = "steelblue")
plot(x, cdf, type = "s", lwd = 2, col = "darkred", main = "CDF of Binomial(12, 0.25)", xlab = "x", ylab = "P(X <= x)")
print(paste("P(X = 4) =", dbinom(4, 12, 0.25)))
print(paste("P(X <= 4) =", pbinom(4, 12, 0.25)))
print(paste("P(X >= 5) =", 1 - pbinom(4, 12, 0.25)))
par(mfrow = c(1, 1))

cat("Q2\n")
n <- 20
p <- 0.6
x <- 0:n
pmf <- dbinom(x, n, p)
plot(x, pmf, type = "h", lwd = 2, col = "steelblue", main = "PMF of Binomial(20, 0.6)", xlab = "x", ylab = "P(X = x)")
points(x, pmf, pch = 16, col = "steelblue")
abline(v = n * p, col = "red", lwd = 2, lty = 2)
print(paste("Mean np =", n * p))

cat("Q3\n")
x <- 0:5
f <- c(4, 15, 30, 28, 17, 6)
N <- sum(f)
n <- 5
mean_x <- sum(x * f) / N
p_hat <- mean_x / n
fitted_prob <- dbinom(x, n, p_hat)
expected <- fitted_prob * N
print(paste("Estimated p =", round(p_hat, 4)))
print(data.frame(x = x, observed = f, fitted_prob = round(fitted_prob, 4), expected = round(expected, 2)))

cat("Q4\n")
lambda <- 5
print(paste("P(X = 3) =", dpois(3, lambda)))
print(paste("P(X <= 3) =", ppois(3, lambda)))
print(paste("P(X > 7) =", 1 - ppois(7, lambda)))
print(paste("P(4 <= X <= 8) =", ppois(8, lambda) - ppois(3, lambda)))

cat("Q5\n")
x <- 0:6
observed <- c(10, 25, 30, 20, 10, 4, 1)
N <- sum(observed)
lambda_hat <- sum(x * observed) / N
fitted_prob <- dpois(x, lambda_hat)
expected <- fitted_prob * N
print(paste("Estimated lambda =", round(lambda_hat, 4)))
print(data.frame(x = x, observed = observed, fitted_prob = round(fitted_prob, 4), expected = round(expected, 2)))
plot(x, observed, type = "h", lwd = 5, col = "steelblue", main = "Observed vs Fitted Poisson Frequencies", xlab = "x", ylab = "Frequency", ylim = c(0, max(observed, expected) + 5))
points(x + 0.12, expected, type = "h", lwd = 5, col = "red")
legend("topright", legend = c("Observed", "Fitted"), col = c("steelblue", "red"), lwd = 3)

cat("Q6\n")
set.seed(123)
b <- rbinom(1000, size = 10, prob = 0.5)
g <- rgeom(1000, prob = 0.3)
p <- rpois(1000, lambda = 4)
print(paste("Binomial -> empirical mean =", round(mean(b), 3), ", theoretical mean =", 10 * 0.5))
print(paste("Geometric -> empirical mean =", round(mean(g), 3), ", theoretical mean =", round((1 - 0.3) / 0.3, 3)))
print(paste("Poisson -> empirical mean =", round(mean(p), 3), ", theoretical mean =", 4))

cat("Q7\n")
set.seed(123)
obs <- rbinom(1000, size = 20, prob = 0.3)
print(table(obs))
print(paste("Sample mean =", round(mean(obs), 3)))
barplot(table(obs), col = "steelblue", main = "Frequency Distribution of Binomial(20, 0.3)", xlab = "Number of Successes", ylab = "Frequency")

cat("Q8\n")
lambda <- 4.5
print(paste("P(X < 2) =", ppois(1, lambda)))
print(paste("P(X >= 2) =", 1 - ppois(1, lambda)))
print(paste("P(X <= 6) =", ppois(6, lambda)))
print(paste("P(X > 8) =", 1 - ppois(8, lambda)))

cat("Q9\n")
lambda <- 6
print(paste("P(exactly 4 calls) =", dpois(4, lambda)))
print(paste("P(at most 5 calls) =", ppois(5, lambda)))
print(paste("P(more than 8 calls) =", 1 - ppois(8, lambda)))

cat("Q10\n")
x <- 0:5
f <- c(6, 16, 29, 30, 14, 5)
N <- sum(f)
n <- 5
mean_x <- sum(x * f) / N
p_hat <- mean_x / n
fitted_prob <- dbinom(x, n, p_hat)
expected <- fitted_prob * N
print(paste("Estimated p =", round(p_hat, 4)))
print(data.frame(x = x, observed = f, fitted_prob = round(fitted_prob, 4), expected = round(expected, 2)))

cat("Q11\n")
x <- 0:7
observed <- c(4, 10, 21, 29, 20, 10, 4, 2)
N <- sum(observed)
lambda_hat <- sum(x * observed) / N
fitted_prob <- dpois(x, lambda_hat)
expected <- fitted_prob * N
print(paste("Estimated lambda =", round(lambda_hat, 4)))
print(data.frame(x = x, observed = observed, fitted_prob = round(fitted_prob, 4), expected = round(expected, 2)))

cat("Q12\n")
joint <- matrix(c(10, 15, 5, 20, 25, 10, 5, 10, 15), nrow = 3, byrow = TRUE)
rownames(joint) <- c("X1", "X2", "X3")
colnames(joint) <- c("Y1", "Y2", "Y3")
print(joint)
print("Marginal distribution of X (row sums):")
print(rowSums(joint))
print("Marginal distribution of Y (column sums):")
print(colSums(joint))
print("Joint probability distribution:")
print(prop.table(joint))

cat("Q13\n")
x <- 0:15
binom_pmf <- dbinom(x, 10, 0.5)
pois_pmf <- dpois(x, 5)
plot(x, binom_pmf, type = "h", lwd = 5, col = "steelblue", main = "PMF of Binomial(10, 0.5) vs Poisson(5)", xlab = "x", ylab = "P(X = x)", ylim = c(0, max(binom_pmf, pois_pmf) + 0.03))
points(x + 0.15, pois_pmf, type = "h", lwd = 5, col = "red")
legend("topright", legend = c("Binomial(10, 0.5)", "Poisson(5)"), col = c("steelblue", "red"), lwd = 3)
print("Both distributions have mean 5 and very similar shapes.")

cat("Q14\n")
print(paste("Sum of Binomial(12, 0.25) PMF =", sum(dbinom(0:12, 12, 0.25))))
print(paste("Sum of Poisson(5) PMF (x = 0 to 50) =", sum(dpois(0:50, 5))))

cat("Q15\n")
print(paste("(a) P(X < 45) =", pnorm(45, mean = 50, sd = 8)))
print(paste("(b) P(X > 60) =", 1 - pnorm(60, mean = 50, sd = 8)))
print(paste("(c) P(40 < X < 55) =", pnorm(55, mean = 50, sd = 8) - pnorm(40, mean = 50, sd = 8)))
print(paste("(d) 90th percentile =", qnorm(0.90, mean = 50, sd = 8)))

cat("Q16\n")
x <- seq(50, 150, length.out = 500)
y <- dnorm(x, mean = 100, sd = 15)
plot(x, y, type = "l", lwd = 2, col = "steelblue", main = "Density of N(100, 15^2)", xlab = "x", ylab = "Density")
abline(v = 100, col = "red", lwd = 2, lty = 2)
print("Density curve plotted with mean marked by a vertical dashed line.")

cat("Q17\n")
x <- seq(-8, 8, length.out = 500)
plot(x, dnorm(x, 0, 1), type = "l", lwd = 2, col = "black", main = "Normal Density Curves", xlab = "x", ylab = "Density", ylim = c(0, 0.45))
lines(x, dnorm(x, 0, 2), lwd = 2, col = "red")
lines(x, dnorm(x, 2, 1), lwd = 2, col = "blue")
legend("topright", legend = c("N(0, 1)", "N(0, 2^2)", "N(2, 1)"), col = c("black", "red", "blue"), lwd = 2)
print("Comment: Increasing the standard deviation from 1 to 2 makes the curve flatter and more spread out around the centre, while changing the mean from 0 to 2 shifts the entire curve to the right without changing its shape.")

cat("Q18\n")
set.seed(123)
obs <- rnorm(500, mean = 25, sd = 5)
print(paste("Sample mean =", round(mean(obs), 3), ", theoretical mean = 25"))
print(paste("Sample standard deviation =", round(sd(obs), 3), ", theoretical standard deviation = 5"))
