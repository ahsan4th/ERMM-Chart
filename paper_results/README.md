# Complete ARL tables of the simulation study

These tables accompany Section 5 of the article (they replace the former
appendix). Every cell gives ARL (SDRL).

| File | Content |
|---|---|
| Table_A1.csv – Table_A3.csv | mean, variability and simultaneous shifts by number of characteristics p (N = 1000, rho = 0.3, lambda = 0.1) |
| Table_A4.csv – Table_A6.csv | the same shifts by weighting constant lambda (N = 1000, p = 3, rho = 0.3) |
| Table_A7.csv – Table_A9.csv | the same shifts by correlation rho (N = 1000, p = 3, lambda = 0.1) |
| Table_A10.csv – Table_A12.csv | case design (N = 100, p = 5, plant correlation), lambda = 0.1, 0.3, 0.5, 0.7 |

A mean shift moves every characteristic by `a` standard deviations, a
variability shift multiplies every variance by `b`, and a simultaneous shift
combines both with b = 1 + a. Out-of-control ARLs use 1000 replications
(N = 1000) or 2000 replications (case design) with common random numbers;
limits were calibrated for ARL0 = 370 with 10,000 replications. These values
were computed with the authors' R code.
