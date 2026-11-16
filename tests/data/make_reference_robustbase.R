# Reference values for tests/test_estimators.py, produced with robustbase 0.99-2:
#   Rscript make_reference_robustbase.R   (run inside tests/data)
library(robustbase)
for (k in c(0, 1, 3, 4)) {
  x <- as.matrix(read.csv(sprintf("x%d.csv", k), header = FALSE))
  set.seed(123)
  f <- covMcd(x, alpha = 0.75, nsamp = 500)
  write.table(f$center, sprintf("r_center%d.txt", k), row.names = FALSE, col.names = FALSE)
  write.table(f$cov, sprintf("r_cov%d.txt", k), row.names = FALSE, col.names = FALSE)
  write.table(f$raw.weights, sprintf("r_weights%d.txt", k), row.names = FALSE, col.names = FALSE)
}
