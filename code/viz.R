# ---------------------------------------------------------------
# compare_models.R
# ---------------------------------------------------------------
# This script:
#   1. Builds a grouped bar plot of accuracy (KNN, GPT4o, Claude)
#      at Top-10 vs. Top-20 for two cohorts (PubMed, Clinical).
#   2. Performs two‐proportion tests (prop.test) between KNN vs. GPT4o
#      and KNN vs. Claude, for each threshold and cohort.
# ---------------------------------------------------------------

# 1) load required packages
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

library(ggplot2)
library(dplyr)

# 2) define the raw accuracy data
#    Columns: model, threshold (top N), cohort, accuracy (proportion)
data <- data.frame(
  model     = c(
    rep("KNN",     4),
    rep("GPT4o",   4),
    rep("Claude",  4)
  ),
  threshold = rep(c(10, 20), times = 2 * 3 / 2),  # will be recycled below
  cohort    = c(
    "PubMed",   "PubMed",   "Clinical", "Clinical",  # KNN rows
    "PubMed",   "PubMed",   "Clinical", "Clinical",  # GPT4o rows
    "PubMed",   "PubMed",   "Clinical", "Clinical"   # Claude rows
  ),
  accuracy  = c(
    # KNN:
    0.169491525, 0.237288136, 0.276923077, 0.323076923,
    # GPT4o:
    0.344827586, 0.327586207, 0.281250000, 0.218750000,
    # Claude:
    0.362068966, 0.465517241, 0.187500000, 0.171875000
  ),
  stringsAsFactors = FALSE
)
	
# (Double‐check that the ordering of "threshold" matches the four accuracy values per model:)
#    For each model, the first two rows correspond to PubMed (Top-10, Top-20),
#    and the next two rows correspond to Clinical (Top-10, Top-20).

# 3) convert some columns to factors for plotting
data$threshold <- factor(data$threshold, levels = c("10", "20"))
data$model     <- factor(data$model, levels    = c("KNN", "GPT4o", "Claude"))
data$cohort    <- factor(data$cohort, levels   = c("PubMed", "Clinical"))

# 4) build a grouped bar chart
plot <- ggplot(data, aes(x = model, y = accuracy, fill = threshold)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75), width = 0.6) +
  facet_wrap(~ cohort) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title    = "Top-N Diagnostic Accuracy: KNN vs. GPT4o vs. Claude",
    x        = "Model",
    y        = "Accuracy (%)",
    fill     = "Top N"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(hjust = 0.5, face = "bold"),
    axis.text.x     = element_text(face = "bold")
  )

# 5) print (or save) the plot
print(plot)

# If you want to save to a PDF or PNG, uncomment one of these lines:
# ggsave("model_comparison_plot.pdf", plot = plot, width = 8, height = 5)
# ggsave("model_comparison_plot.png", plot = plot, width = 8, height = 5, dpi = 300)

# 6) append sample sizes and compute raw counts of successes/failures
data_counts <- data %>%
  mutate(
    n = ifelse(cohort == "PubMed", 59, 65),
    # round() is used to convert proportion × n into an integer count
    successes = round(accuracy * n),
    failures   = n - successes
  )

# 7) define a helper to run pairwise proportion tests for each (cohort, threshold)
run_prop_tests <- function(cohort_name, thr) {
  # subset to the given cohort & threshold
  df_sub <- data_counts %>%
    filter(cohort == cohort_name, threshold == thr)

  n_cohort <- df_sub$n[1]
  # extract successes for each model
  knn_succ   <- df_sub$successes[df_sub$model == "KNN"]
  gpt4o_succ <- df_sub$successes[df_sub$model == "GPT4o"]
  cla_succ   <- df_sub$successes[df_sub$model == "Claude"]

  cat("--------------------------------------------------------------\n")
  cat("Cohort =", cohort_name, " | Top", thr, "\n\n")

  # KNN vs GPT4o
  test1 <- prop.test(
    x = c(knn_succ, gpt4o_succ),
    n = c(n_cohort, n_cohort),
    alternative = "two.sided",
    correct = FALSE
  )
  cat("KNN vs GPT4o:\n")
  cat("    KNN successes =", knn_succ, "of", n_cohort, "\n")
  cat("    GPT4o successes =", gpt4o_succ, "of", n_cohort, "\n")
  cat("    prop.test p-value =", signif(test1$p.value, 3), "\n\n")

  # KNN vs Claude
  test2 <- prop.test(
    x = c(knn_succ, cla_succ),
    n = c(n_cohort, n_cohort),
    alternative = "two.sided",
    correct = FALSE
  )
  cat("KNN vs Claude:\n")
  cat("    KNN successes =", knn_succ, "of", n_cohort, "\n")
  cat("    Claude successes =", cla_succ, "of", n_cohort, "\n")
  cat("    prop.test p-value =", signif(test2$p.value, 3), "\n\n")
}

# 8) run the tests for all cohorts × thresholds
for (co in levels(data_counts$cohort)) {
  for (th in levels(data_counts$threshold)) {
    run_prop_tests(cohort_name = co, thr = as.numeric(as.character(th)))
  }
}
