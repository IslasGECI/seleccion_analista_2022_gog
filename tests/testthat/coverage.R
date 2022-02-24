library(covr)
library(testthat)
cobertura <- file_coverage(
  c(
    "R/do_nothing.R",
    "R/dummy_model.R"
  ),
  c(
    "tests/testthat/test_nothing.R",
    "tests/testthat/test_dummy_model.R"
  )
)
covr::codecov(coverage = cobertura, token = "21a75c47-3624-40d2-9f9c-262a51e35d50")
