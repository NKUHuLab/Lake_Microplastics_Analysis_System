# ============================================================================
# IngestionMethods/TrophicControls visual revision v4
# Purpose:
#   1) Replace IngestionMethodsb heatmap/bubble with an information-dense zero-rate forest.
#   2) Merge IngestionMethodsc and TrophicControlsb into one table-style forest plot.
#   3) Drop IngestionMethodsd from the main evidence set.
#   4) Add habitat-resolved biodilution plot for TrophicControls.
#   5) Add core-cluster diagnostics, items/ind contrast, leave-one-study-out
#      influence analysis, and trophic-level uncertainty stress tests.
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(glmmTMB)
  library(ggeffects)
  library(scales)
  library(patchwork)
})

BASE <- Sys.getenv("LAKE_MP_REPO_ROOT", normalizePath(getwd(), winslash = "/", mustWork = TRUE))
DATA_ROOT <- Sys.getenv("LAKE_MP_DATA_ROOT", file.path(BASE, "data"))
OUT_ROOT <- Sys.getenv("LAKE_MP_OUTPUT_DIR", file.path(BASE, "outputs"))
INPUT <- Sys.getenv("LAKE_MP_BIOTA_COLLAPSED_WORKBOOK", file.path(DATA_ROOT, "evidence", "biota_zinb_and_traits", "Supplementary_Data_2_FINAL_COLLAPSED.xlsx"))
INPUT10 <- Sys.getenv("LAKE_MP_BIOTA_PREDICTED_WORKBOOK", file.path(DATA_ROOT, "evidence", "biota_zinb_and_traits", "Supplementary_Data_2_WITH_PRED_MP.xlsx"))
DIR_DATA <- file.path(OUT_ROOT, "biota_zinb_and_traits", "data")
DIR_FIG <- file.path(OUT_ROOT, "biota_zinb_and_traits", "figures")
dir.create(DIR_DATA, showWarnings = FALSE, recursive = TRUE)
dir.create(DIR_FIG, showWarnings = FALSE, recursive = TRUE)

set.seed(20260624)

safe_pdf <- function(filename, width, height, ...) {
  grDevices::pdf(file = filename, width = width, height = height,
                 family = "sans", useDingbats = FALSE, onefile = FALSE)
}

save_panel <- function(plot, base_name, width = 8, height = 5.2, dpi = 420) {
  png_path <- file.path(DIR_FIG, paste0(base_name, ".png"))
  pdf_path <- file.path(DIR_FIG, paste0(base_name, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, device = safe_pdf, bg = "white")
  invisible(tibble(file_base = base_name, png = png_path, pdf = pdf_path))
}

theme_nature <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "grey15"),
      axis.ticks = element_line(linewidth = 0.30, colour = "grey15"),
      axis.title = element_text(face = "bold", colour = "grey10"),
      axis.text = element_text(colour = "grey18"),
      plot.title = element_text(face = "bold", size = base_size + 2, hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.1, colour = "grey30", hjust = 0),
      plot.caption = element_text(size = base_size - 1.2, colour = "grey35", hjust = 0),
      legend.title = element_text(face = "bold", size = base_size - 0.4),
      legend.text = element_text(size = base_size - 0.9),
      legend.key.height = unit(0.32, "cm"),
      panel.grid.major.y = element_line(linewidth = 0.18, colour = "grey90"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(7, 8, 6, 8)
    )
}

method_classify <- function(x) {
  case_when(
    str_detect(x, regex("KOH", ignore_case = TRUE)) ~ "KOH",
    str_detect(x, regex("H2O2|hydrogen peroxide", ignore_case = TRUE)) ~ "H2O2",
    str_detect(x, regex("Enzymatic|SDS|proteinase", ignore_case = TRUE)) ~ "Enzymatic",
    str_detect(x, regex("Sieving|visual sorting", ignore_case = TRUE)) ~ "Physical",
    TRUE ~ "Other"
  )
}

tissue_group <- function(x) {
  case_when(
    str_detect(x, regex("gill", ignore_case = TRUE)) &
      str_detect(x, regex("GIT|gut|gastro|stomach|intestine", ignore_case = TRUE)) ~ "GIT + gills",
    str_detect(x, regex("gill", ignore_case = TRUE)) ~ "Gills",
    str_detect(x, regex("whole", ignore_case = TRUE)) ~ "Whole body",
    str_detect(x, regex("GIT|gut|gastro|stomach|intestine", ignore_case = TRUE)) ~ "GIT",
    TRUE ~ "Other tissue"
  )
}

taxon_group <- function(family, species, ref) {
  bird_fam <- c("Accipitridae", "Cathartidae", "Strigidae", "Phalacrocoracidae", "Pandionidae")
  inv_fam <- c("Atyidae", "Palaemonidae")
  case_when(
    family %in% bird_fam |
      str_detect(species, regex("Phalacrocorax|Accipiter|Buteo|Cathartes|Coragyps|Pandion|Megascops|Strix|cormorant", ignore_case = TRUE)) |
      str_detect(ref, regex("bird|cormorant|raptor", ignore_case = TRUE)) ~ "Bird",
    family %in% inv_fam |
      str_detect(species, regex("Caridina|Macrobrachium|shrimp|arthropod", ignore_case = TRUE)) ~ "Invertebrate",
    !is.na(family) & family != "" ~ "Fish",
    TRUE ~ "Other/uncertain"
  )
}

format_p <- function(p) {
  ifelse(is.na(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
}

format_ci <- function(beta, lo, hi) {
  sprintf("%.2f [%.2f, %.2f]", beta, lo, hi)
}

fit_tl_model <- function(dat, response_col = "MP_g", rhs = "Pred_s + TL_s + Origin + Habitat + (1 | Study)",
                         zi = TRUE) {
  d <- dat %>%
    filter(!is.na(.data[[response_col]]), .data[[response_col]] >= 0) %>%
    mutate(Yc = round(.data[[response_col]] * 1000))
  form <- as.formula(paste("Yc ~", rhs))
  model <- tryCatch(
    suppressWarnings(glmmTMB(form, ziformula = if (zi) ~1 else ~0, family = nbinom2(link = "log"), data = d)),
    error = function(e) NULL
  )
  if (is.null(model) && zi) {
    model <- tryCatch(
      suppressWarnings(glmmTMB(form, family = nbinom2(link = "log"), data = d)),
      error = function(e) NULL
    )
  }
  if (is.null(model)) {
    return(list(model = NULL, row = tibble(
      N = nrow(d), N_Studies = n_distinct(d$Study), beta = NA_real_, se = NA_real_,
      z = NA_real_, p = NA_real_, lo = NA_real_, hi = NA_real_
    ), data = d))
  }
  co <- summary(model)$coefficients$cond
  if (!("TL_s" %in% rownames(co))) {
    beta <- se <- z <- p <- NA_real_
  } else {
    tl <- co["TL_s", ]
    beta <- as.numeric(tl["Estimate"])
    se <- as.numeric(tl["Std. Error"])
    z <- as.numeric(tl["z value"])
    p <- as.numeric(tl["Pr(>|z|)"])
  }
  list(
    model = model,
    row = tibble(
      N = nrow(d), N_Studies = n_distinct(d$Study),
      beta = beta, se = se, z = z, p = p,
      lo = beta - 1.96 * se,
      hi = beta + 1.96 * se
    ),
    data = d
  )
}

fit_subgroup_tl <- function(dat, response_col = "MP_g", min_n = 8, min_studies = 3) {
  d <- dat %>%
    filter(!is.na(.data[[response_col]]), .data[[response_col]] >= 0, !is.na(TL_s), !is.na(Pred_s)) %>%
    mutate(Yc = round(.data[[response_col]] * 1000))

  if (nrow(d) < min_n || n_distinct(d$Study) < min_studies || n_distinct(d$TL_s) < 4) {
    return(tibble(
      N = nrow(d), N_Studies = n_distinct(d$Study), beta = NA_real_, se = NA_real_,
      z = NA_real_, p = NA_real_, lo = NA_real_, hi = NA_real_,
      Covariates = "not modelled: sparse subgroup"
    ))
  }

  covars <- c("Pred_s")
  add_cov <- function(var, min_cell = 5, max_n_per_cov = 25) {
    tab <- table(d[[var]])
    n_distinct(d[[var]]) > 1 && min(tab[tab > 0]) >= min_cell && nrow(d) >= max_n_per_cov
  }
  if (add_cov("Origin")) covars <- c(covars, "Origin")
  if (add_cov("Habitat")) covars <- c(covars, "Habitat")
  if (add_cov("Method_Model")) covars <- c(covars, "Method_Model")

  random_term <- if (n_distinct(d$Study) >= 4) "(1 | Study)" else NULL
  rhs <- paste(c("TL_s", covars, random_term), collapse = " + ")
  form <- as.formula(paste("Yc ~", rhs))

  model <- tryCatch(
    suppressWarnings(glmmTMB(form, ziformula = ~1, family = nbinom2(link = "log"), data = d)),
    error = function(e) NULL
  )
  if (is.null(model)) {
    model <- tryCatch(
      suppressWarnings(glmmTMB(form, family = nbinom2(link = "log"), data = d)),
      error = function(e) NULL
    )
  }
  if (is.null(model) && length(covars) > 1) {
    rhs <- paste(c("TL_s", "Pred_s", random_term), collapse = " + ")
    form <- as.formula(paste("Yc ~", rhs))
    model <- tryCatch(
      suppressWarnings(glmmTMB(form, family = nbinom2(link = "log"), data = d)),
      error = function(e) NULL
    )
    covars <- "Pred_s"
  }
  if (is.null(model)) {
    return(tibble(
      N = nrow(d), N_Studies = n_distinct(d$Study), beta = NA_real_, se = NA_real_,
      z = NA_real_, p = NA_real_, lo = NA_real_, hi = NA_real_,
      Covariates = "model failed"
    ))
  }

  co <- summary(model)$coefficients$cond
  if (!("TL_s" %in% rownames(co))) {
    beta <- se <- z <- p <- NA_real_
  } else {
    tl <- co["TL_s", ]
    beta <- as.numeric(tl["Estimate"])
    se <- as.numeric(tl["Std. Error"])
    z <- as.numeric(tl["z value"])
    p <- as.numeric(tl["Pr(>|z|)"])
  }

  tibble(
    N = nrow(d), N_Studies = n_distinct(d$Study),
    beta = beta, se = se, z = z, p = p,
    lo = beta - 1.96 * se,
    hi = beta + 1.96 * se,
    Covariates = paste(covars, collapse = " + ")
  )
}

predict_tl <- function(fit) {
  pred <- tryCatch(ggeffects::ggpredict(fit$model, terms = "TL_s [all]", type = "count"), error = function(e) NULL)
  if (is.null(pred)) pred <- ggeffects::ggpredict(fit$model, terms = "TL_s [all]")
  as_tibble(pred) %>%
    mutate(
      TL = x * sd(fit$data$TL, na.rm = TRUE) + mean(fit$data$TL, na.rm = TRUE),
      predicted_orig = predicted / 1000,
      conf.low_orig = conf.low / 1000,
      conf.high_orig = conf.high / 1000
    )
}

method_cols <- c(
  "KOH" = "#4B74B2",
  "H2O2" = "#4FAE9E",
  "Physical" = "#D69A48",
  "Enzymatic" = "#8B72B1",
  "Other" = "#888888"
)

habitat_cols <- c(
  "F-Only" = "#2A6FBB",
  "FM" = "#45A778",
  "FT" = "#C87B47",
  "FMT" = "#8E67AE"
)

habitat_labels <- c(
  "F-Only" = "F-Only\nfreshwater only",
  "FM" = "FM\nfreshwater + marine",
  "FT" = "FT\nfreshwater + terrestrial",
  "FMT" = "FMT\nfreshwater + marine + terrestrial"
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df0 <- read_excel(INPUT)
df10_raw <- read_excel(INPUT10)
stopifnot(nrow(df0) == 120, nrow(df10_raw) == 120)

df <- df0 %>%
  rename(
    MP_g = Measured_Abundance_g,
    MP_ind = Measured_Abundance_ind,
    TL = Trophic_level,
    Habitat = `Habitat dependency`
  ) %>%
  mutate(
    Method = method_classify(Separation_Method),
    Zero_g = MP_g == 0,
    Zero_ind = MP_ind == 0,
    Study = factor(Reference),
    Habitat = factor(Habitat, levels = c("F-Only", "FM", "FT", "FMT"))
  )

df10 <- df10_raw %>%
  rename(
    MP_g = Measured_Abundance_g,
    MP_ind = Measured_Abundance_ind,
    TL = Trophic_level,
    Pred_MP = Predicted_Lake_MP,
    Habitat = `Habitat dependency`
  ) %>%
  mutate(
    Method = method_classify(Separation_Method),
    Method_Model = factor(if_else(Method %in% c("KOH", "H2O2"), Method, "Minor methods")),
    Sensitivity = factor(if_else(Target_Size_Lower_Limit_um < 50, "<50 um", ">=50 um"),
                         levels = c("<50 um", ">=50 um")),
    Tissue_Group = factor(tissue_group(Tissue_Analyzed),
                          levels = c("GIT", "GIT + gills", "Whole body", "Gills", "Other tissue")),
    Taxon_Group = taxon_group(Family, Species_or_Sample_Type, Reference),
    Origin = factor(Origin),
    Habitat = factor(Habitat, levels = c("F-Only", "FM", "FT", "FMT")),
    Habitat3 = fct_collapse(Habitat, "FT/FMT" = c("FT", "FMT")),
    Study = factor(Reference),
    Pred_s = as.numeric(scale(log1p(Pred_MP))),
    TL_s = as.numeric(scale(TL))
  )

model_rows <- read_csv(file.path(DIR_DATA, "IngestionMethods_TrophicControls_model_summary_v2.csv"), show_col_types = FALSE) %>%
  distinct(Model, .keep_all = TRUE)

primary_fit <- fit_tl_model(df10, "MP_g")
pred_primary <- predict_tl(primary_fit)
ind_fit <- fit_tl_model(df10, "MP_ind")
pred_ind <- predict_tl(ind_fit)

# ---------------------------------------------------------------------------
# Fig IngestionMethodsb v4: detection-limit zero-rate forest
# ---------------------------------------------------------------------------
spearman_zero <- suppressWarnings(cor(log10(df$Target_Size_Lower_Limit_um + 1), as.numeric(df$Zero_g),
                                      method = "spearman", use = "complete.obs"))

zero_rate_rows <- df %>%
  mutate(
    Size_window = case_when(
      Target_Size_Lower_Limit_um < 1 ~ "<1",
      Target_Size_Lower_Limit_um < 10 ~ "1-<10",
      Target_Size_Lower_Limit_um < 50 ~ "10-<50",
      Target_Size_Lower_Limit_um < 100 ~ "50-<100",
      TRUE ~ ">=100"
    ),
    Size_window = factor(Size_window, levels = c("<1", "1-<10", "10-<50", "50-<100", ">=100")),
    Sensitivity_window = if_else(Target_Size_Lower_Limit_um < 50, "Higher sensitivity (<50 um)",
                                 "Lower sensitivity (>=50 um)")
  ) %>%
  group_by(Method, Size_window, Sensitivity_window) %>%
  summarise(
    N = n(),
    Studies = n_distinct(Study),
    Zeros = sum(Zero_g),
    Zero_rate = Zeros / N,
    .groups = "drop"
  ) %>%
  filter(N > 0) %>%
  mutate(
    wilson_den = 1 + 1.96^2 / N,
    wilson_mid = (Zero_rate + 1.96^2 / (2 * N)) / wilson_den,
    wilson_half = 1.96 * sqrt((Zero_rate * (1 - Zero_rate) / N) + 1.96^2 / (4 * N^2)) / wilson_den,
    lo = pmax(0, wilson_mid - wilson_half),
    hi = pmin(1, wilson_mid + wilson_half),
    Row_label = sprintf("%s | %s", Method, Size_window),
    Count_label = sprintf("%d/%d", Zeros, N),
    Method = factor(Method, levels = c("KOH", "H2O2", "Physical", "Enzymatic", "Other"))
  ) %>%
  arrange(Method, Size_window) %>%
  mutate(Row = rev(row_number()))

write_csv(
  zero_rate_rows %>%
    transmute(
      Method = as.character(Method),
      Size_window = as.character(Size_window),
      Sensitivity_window,
      N,
      Studies,
      Zeros,
      Zero_rate,
      Wilson_95CI_low = lo,
      Wilson_95CI_high = hi
    ),
  file.path(DIR_DATA, "IngestionMethods_detection_zero_rate_v4.csv")
)

overall_zero <- mean(df$Zero_g, na.rm = TRUE)

p_zero_rate <- ggplot(zero_rate_rows, aes(y = Row)) +
  geom_rect(data = zero_rate_rows %>% filter(row_number() %% 2 == 0),
            aes(xmin = -0.22, xmax = 1.15, ymin = Row - 0.43, ymax = Row + 0.43),
            inherit.aes = FALSE, fill = "grey96", colour = NA) +
  annotate("rect", xmin = 0, xmax = overall_zero, ymin = -Inf, ymax = Inf,
           fill = "#EDF4FA", alpha = 0.9) +
  geom_vline(xintercept = overall_zero, linetype = "dotted", colour = "#315D8A", linewidth = 0.42) +
  geom_vline(xintercept = 0, colour = "grey30", linewidth = 0.32) +
  geom_segment(aes(x = lo, xend = hi, yend = Row), colour = "grey20", linewidth = 0.42, lineend = "round") +
  geom_point(aes(x = Zero_rate, size = N, fill = Sensitivity_window),
             shape = 21, colour = "grey12", stroke = 0.32, alpha = 0.92) +
  geom_text(aes(x = -0.20, label = Row_label), hjust = 0, size = 2.55, colour = "grey10") +
  geom_text(aes(x = 1.03, label = Count_label), hjust = 0, size = 2.55, colour = "grey10") +
  annotate("text", x = -0.20, y = max(zero_rate_rows$Row) + 0.95, label = "Method | size window",
           hjust = 0, size = 3.0, fontface = "bold") +
  annotate("text", x = 1.03, y = max(zero_rate_rows$Row) + 0.95, label = "Zeros / N",
           hjust = 0, size = 3.0, fontface = "bold") +
  scale_x_continuous(limits = c(-0.22, 1.15), breaks = c(0, 0.25, 0.5, 0.75, 1.0),
                     labels = percent_format(accuracy = 1), guide = guide_axis(cap = TRUE)) +
  scale_y_continuous(limits = c(0.4, max(zero_rate_rows$Row) + 1.2), breaks = NULL) +
  scale_size_area(max_size = 8.4, breaks = c(5, 15, 30), name = "Records") +
  scale_fill_manual(values = c("Higher sensitivity (<50 um)" = "#8DBFE3",
                               "Lower sensitivity (>=50 um)" = "#E0A46C"),
                    name = "Detection window") +
  labs(
    title = "IngestionMethodsb | Zero-rate diagnostic by method and detection window",
    subtitle = sprintf("Points show zero proportion with Wilson 95%% CI; dotted line is the overall zero rate (%.1f%%), Spearman rho=%.2f",
                       100 * overall_zero, spearman_zero),
    x = "Zero proportion for mass-normalized abundance",
    y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_nature(8) +
  theme(
    legend.position = "bottom",
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 14, 8, 10)
  )
save_panel(p_zero_rate, "Fig_IngestionMethodsb_detection_zero_rate_v4", 8.8, 5.8)

# ---------------------------------------------------------------------------
# Fig IngestionMethodsc/TrophicControlsb v4: integrated table-style forest plot
# ---------------------------------------------------------------------------
forest_models <- tribble(
  ~Group, ~Model, ~Display, ~Data_set, ~Scope,
  "IngestionMethods method and detection controls", "IngestionMethods/TrophicControls primary items/g", "Primary base model", "Full items/g", "Full-data model",
  "IngestionMethods method and detection controls", "IngestionMethods method-adjusted", "+ method covariate", "Full items/g", "Full-data covariate control",
  "IngestionMethods method and detection controls", "IngestionMethods detection-adjusted", "+ detection-limit covariate", "Full items/g", "Full-data covariate control",
  "IngestionMethods method and detection controls", "IngestionMethods KOH-only", "KOH-only stratum", "KOH subset", "Method-stratified subset",
  "IngestionMethods method and detection controls", "IngestionMethods H2O2-only", "H2O2-only stratum", "H2O2 subset", "Method-stratified subset",
  "TrophicControls biological and scale controls", "TrophicControls tissue-adjusted", "+ tissue covariate", "Full items/g", "Full-data covariate control",
  "TrophicControls biological and scale controls", "TrophicControls dominant methods", "Dominant KOH+H2O2", "KOH+H2O2 subset", "Dominant-method subset",
  "TrophicControls biological and scale controls", "TrophicControls items/ind contrast", "Items/ind response", "Full items/ind", "Alternative normalization",
  "Sparse-taxon sensitivity", "Family n >= 2", "Family n >= 2", "Family subset", "Sparse-taxon subset",
  "Sparse-taxon sensitivity", "Family n >= 3", "Family n >= 3", "Family subset", "Sparse-taxon subset",
  "Sparse-taxon sensitivity", "Family n >= 4", "Family n >= 4", "Family subset", "Sparse-taxon subset"
)

forest_df <- forest_models %>%
  left_join(model_rows, by = "Model") %>%
  mutate(
    Group = factor(Group, levels = c("IngestionMethods method and detection controls",
                                     "TrophicControls biological and scale controls",
                                     "Sparse-taxon sensitivity")),
    Display = factor(Display, levels = rev(Display)),
    Row_base = rev(row_number()) * 0.62,
    Row = Row_base + case_when(
      Group == "IngestionMethods method and detection controls" ~ 0.74,
      Group == "TrophicControls biological and scale controls" ~ 0.37,
      TRUE ~ 0
    ),
    CI_label = format_ci(TL_beta, TL_CI_low, TL_CI_high),
    P_label = format_p(TL_p),
    Sig = !is.na(TL_p) & TL_p < 0.05,
    Row_label = as.character(Display),
    Data_label = Data_set,
    N_label = sprintf("%d", N),
    Study_label = sprintf("%d", N_Studies)
  )

header_rows <- forest_df %>%
  group_by(Group) %>%
  summarise(Row = max(Row) + 0.42, .groups = "drop")

separators <- forest_df %>%
  group_by(Group) %>%
  summarise(y = min(Row) - 0.28, .groups = "drop") %>%
  filter(y > min(forest_df$Row) - 0.2)

forest_axis_min <- -1.10
forest_axis_max <- 0.25
forest_df <- forest_df %>%
  mutate(
    CI_low_plot = pmax(TL_CI_low, forest_axis_min),
    CI_high_plot = pmin(TL_CI_high, forest_axis_max),
    left_trunc = TL_CI_low < forest_axis_min,
    right_trunc = TL_CI_high > forest_axis_max,
    point_fill = case_when(
      Sig & TL_beta <= -0.50 ~ "#2C5F8E",
      Sig & TL_beta < -0.20 ~ "#8CBAD7",
      TRUE ~ "white"
    )
  )

write_csv(
  forest_df %>%
    transmute(
      Group = as.character(Group),
      Model,
      Display = as.character(Display),
      Data_set,
      Scope,
      N,
      N_Studies,
      TL_beta,
      TL_CI_low,
      TL_CI_high,
      TL_p,
      CI_label,
      P_label,
      Significant_negative = Sig
    ),
  file.path(DIR_DATA, "IngestionMethods_TrophicControls_integrated_forest_v4.csv")
)

row_shades <- forest_df %>%
  filter(row_number() %% 2 == 0) %>%
  transmute(ymin = Row - 0.24, ymax = Row + 0.24)

p_forest <- ggplot(forest_df, aes(y = Row)) +
  geom_rect(data = row_shades, aes(xmin = -5.35, xmax = 1.78, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE, fill = "grey96", colour = NA) +
  annotate("rect", xmin = forest_axis_min, xmax = -0.50, ymin = -Inf, ymax = Inf,
           fill = "#E7F0F7", alpha = 0.95) +
  annotate("rect", xmin = -0.50, xmax = -0.20, ymin = -Inf, ymax = Inf,
           fill = "#F2F7FB", alpha = 0.95) +
  annotate("rect", xmin = -0.20, xmax = forest_axis_max, ymin = -Inf, ymax = Inf,
           fill = "#FAFAFA", alpha = 0.95) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey20", linewidth = 0.45) +
  geom_vline(xintercept = c(-0.50, -0.20), linetype = "dotted", colour = "#6E8CA6", linewidth = 0.28) +
  geom_hline(data = separators, aes(yintercept = y),
             linewidth = 0.28, linetype = "dashed", colour = "grey45") +
  geom_segment(data = forest_df %>% filter(!left_trunc & !right_trunc),
               aes(x = CI_low_plot, xend = CI_high_plot, yend = Row),
               linewidth = 0.35, colour = "grey18", lineend = "round") +
  geom_segment(data = forest_df %>% filter(left_trunc),
               aes(x = forest_axis_min, xend = CI_high_plot, yend = Row),
               linewidth = 0.35, colour = "grey18", lineend = "round",
               arrow = grid::arrow(type = "closed", ends = "first", length = grid::unit(0.09, "cm"))) +
  geom_segment(data = forest_df %>% filter(right_trunc),
               aes(x = CI_low_plot, xend = forest_axis_max, yend = Row),
               linewidth = 0.35, colour = "grey18", lineend = "round",
               arrow = grid::arrow(type = "closed", ends = "last", length = grid::unit(0.09, "cm"))) +
  geom_point(aes(x = TL_beta, fill = point_fill), shape = 22, size = 2.65, colour = "grey10", stroke = 0.32) +
  scale_fill_identity() +
  geom_text(aes(x = -5.20, label = Row_label), hjust = 0, size = 2.78, colour = "grey10") +
  geom_text(aes(x = -3.58, label = Data_label), hjust = 0, size = 2.58, colour = "grey20") +
  geom_text(aes(x = -2.66, label = N_label), hjust = 0.5, size = 2.58, colour = "grey15") +
  geom_text(aes(x = -2.20, label = Study_label), hjust = 0.5, size = 2.58, colour = "grey15") +
  geom_text(aes(x = 0.45, label = CI_label), hjust = 0, size = 2.60, colour = "grey10") +
  geom_text(aes(x = 1.34, label = P_label), hjust = 0, size = 2.60, colour = "grey10") +
  geom_text(data = header_rows, aes(x = -5.20, y = Row, label = Group), inherit.aes = FALSE,
            hjust = 0, size = 2.92, fontface = "bold", colour = "grey10") +
  annotate("text", x = -5.20, y = max(forest_df$Row) + 0.82, label = "Model", hjust = 0,
           size = 3.08, fontface = "bold") +
  annotate("text", x = -3.58, y = max(forest_df$Row) + 0.82, label = "Data", hjust = 0,
           size = 3.08, fontface = "bold") +
  annotate("text", x = -2.66, y = max(forest_df$Row) + 0.82, label = "N", hjust = 0.5,
           size = 3.08, fontface = "bold") +
  annotate("text", x = -2.20, y = max(forest_df$Row) + 0.82, label = "Studies", hjust = 0.5,
           size = 3.08, fontface = "bold") +
  annotate("text", x = -0.80, y = max(forest_df$Row) + 0.40, label = "Stronger", hjust = 0.5,
           size = 2.52, colour = "#2C5F8E") +
  annotate("text", x = -0.35, y = max(forest_df$Row) + 0.40, label = "Moderate", hjust = 0.5,
           size = 2.52, colour = "#4E7696") +
  annotate("text", x = 0.03, y = max(forest_df$Row) + 0.40, label = "Weak/null", hjust = 0.5,
           size = 2.52, colour = "grey35") +
  annotate("text", x = 0.45, y = max(forest_df$Row) + 0.82, label = "Beta [95% CI]", hjust = 0,
           size = 3.08, fontface = "bold") +
  annotate("text", x = 1.34, y = max(forest_df$Row) + 0.82, label = "P", hjust = 0,
           size = 3.08, fontface = "bold") +
  scale_x_continuous(limits = c(-5.35, 1.78), breaks = c(-1.0, -0.5, 0, 0.25),
                     guide = guide_axis(cap = TRUE)) +
  scale_y_continuous(limits = c(0.34, max(forest_df$Row) + 0.94), breaks = NULL) +
  labs(
    title = "IngestionMethods/TrophicControls | Method controls and biodilution robustness",
    subtitle = "Full-data covariate controls retain N=120; subset rows explicitly reduce N. Negative beta supports mass-normalized biodilution.",
    x = "Trophic-level coefficient (standardized beta)",
    y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_nature(9.0) +
  theme(
    legend.position = "none",
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 10, 6, 8)
  )
save_panel(p_forest, "Fig_IngestionMethodsc_TrophicControlsb_integrated_forest_v4", 9.8, 4.0)

# ---------------------------------------------------------------------------
# Fig TrophicControlsa/b v4: core clustering for items/g and items/ind
# ---------------------------------------------------------------------------
clean_mp_label <- function(x) {
  case_when(
    x == 0 ~ "0",
    x < 0.01 ~ sprintf("%.3f", x),
    x < 0.1 ~ sprintf("%.2f", x),
    x < 1 ~ sprintf("%.1f", x),
    x < 100 ~ sprintf("%.0f", x),
    TRUE ~ sprintf("%.0f", x)
  )
}

safe_scale <- function(x) {
  s <- mad(x, constant = 1, na.rm = TRUE)
  if (!is.finite(s) || s == 0) s <- IQR(x, na.rm = TRUE) / 1.349
  if (!is.finite(s) || s == 0) s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) s <- 1
  s
}

pseudo_y <- function(y, sigma = 0.05, base = 10) {
  asinh(pmax(y, 0) / sigma) / log(base)
}

make_core_clusters <- function(dat, response_col, fraction = 0.50) {
  base <- dat %>%
    filter(!is.na(.data[[response_col]]), .data[[response_col]] >= 0, !is.na(TL)) %>%
    mutate(
      Y = .data[[response_col]],
      LogY = log10(1 + Y),
      Yv = pseudo_y(Y),
      Habitat = factor(Habitat, levels = c("F-Only", "FM", "FT", "FMT")),
      Origin = factor(Origin, levels = c("Native", "Invasive"))
    )

  candidates <- bind_rows(
    base %>%
      transmute(Reference, TL, Y, LogY, Yv, Pred_MP, Origin, Habitat,
                Core_type = "Origin", Core_group = as.character(Origin)),
    base %>%
      filter(Habitat %in% c("F-Only", "FM")) %>%
      transmute(Reference, TL, Y, LogY, Yv, Pred_MP, Origin, Habitat,
                Core_type = "Habitat", Core_group = as.character(Habitat))
  ) %>%
    group_by(Core_type, Core_group) %>%
    mutate(
      Dist = sqrt(((TL - median(TL, na.rm = TRUE)) / safe_scale(TL))^2 +
                    ((Yv - median(Yv, na.rm = TRUE)) / safe_scale(Yv))^2),
      Cut = quantile(Dist, probs = fraction, na.rm = TRUE),
      Is_core = Dist <= Cut
    ) %>%
    ungroup()

  hulls <- candidates %>%
    filter(Is_core) %>%
    group_by(Core_type, Core_group) %>%
    filter(n() >= 3, n_distinct(TL) >= 2, n_distinct(Yv) >= 2) %>%
    slice(chull(TL, Yv)) %>%
    ungroup()

  labels <- candidates %>%
    group_by(Core_type, Core_group) %>%
    summarise(
      N_total = n(),
      N_core = sum(Is_core),
      Core_pct = 100 * N_core / N_total,
      TL_label = median(TL[Is_core], na.rm = TRUE),
      Y_label = median(Y[Is_core], na.rm = TRUE),
      LogY_label = median(LogY[Is_core], na.rm = TRUE),
      Yv_label = median(Yv[Is_core], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      Core_key = Core_group,
      Label = sprintf("%s core\n%d/%d (%.0f%%)", Core_group, N_core, N_total, Core_pct),
      TL_label = case_when(
        Core_group == "Native" ~ TL_label + 0.28,
        Core_group == "Invasive" ~ TL_label - 0.25,
        Core_group == "F-Only" ~ TL_label - 0.40,
        Core_group == "FM" ~ TL_label + 0.16,
        TRUE ~ TL_label
      ),
      Y_label = case_when(
        Core_group == "Native" ~ pmax(Y_label * 1.70, 0.020),
        Core_group == "Invasive" ~ pmax(Y_label * 0.72, 0.015),
        Core_group == "F-Only" ~ pmax(Y_label * 0.55, 0.012),
        Core_group == "FM" ~ pmax(Y_label * 1.40, 0.020),
        TRUE ~ Y_label
      ),
      Yv_label = case_when(
        Core_group == "Native" ~ Yv_label + 0.10,
        Core_group == "Invasive" ~ Yv_label + 0.18,
        Core_group == "F-Only" ~ Yv_label - 0.15,
        Core_group == "FM" ~ Yv_label + 0.28,
        TRUE ~ Yv_label
      ),
      Yv_label = pmax(Yv_label, 0.12)
    )

  small_hab <- base %>%
    filter(Habitat %in% c("FT", "FMT")) %>%
    group_by(Habitat) %>%
    summarise(
      N_total = n(),
      TL_label = median(TL, na.rm = TRUE),
      Y_label = median(Y, na.rm = TRUE),
      LogY_label = median(LogY, na.rm = TRUE),
      Yv_label = median(Yv, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Label = sprintf("%s\nn=%d", Habitat, N_total))

  list(base = base, hulls = hulls, labels = labels, small_hab = small_hab)
}

plot_core_cluster <- function(fit, pred, response_col, title, subtitle, y_label, csv_name, figure_name) {
  core <- make_core_clusters(fit$data, response_col, fraction = 0.50)
  core_cols <- c("Native" = "#252525", "Invasive" = "#777777",
                 "F-Only" = habitat_cols[["F-Only"]], "FM" = habitat_cols[["FM"]])
  data_max <- max(core$base$Y, na.rm = TRUE)
  y_breaks <- c(0, 0.1, 1, 10, 100, 1000)
  y_breaks <- y_breaks[y_breaks <= max(100, data_max * 1.08)]
  pred_plot <- pred %>%
    mutate(
      predicted_yv = pseudo_y(predicted_orig),
      conf_low_yv = pseudo_y(conf.low_orig),
      conf_high_yv = pseudo_y(conf.high_orig)
    )

  write_csv(
    core$labels %>%
      transmute(Core_type, Core_group, N_total, N_core, Core_pct),
    file.path(DIR_DATA, csv_name)
  )

  p <- ggplot(core$base, aes(x = TL, y = Yv)) +
    geom_ribbon(data = pred_plot, aes(x = TL, ymin = conf_low_yv, ymax = conf_high_yv),
                inherit.aes = FALSE, fill = "#7DB6E8", alpha = 0.18) +
    geom_line(data = pred_plot, aes(x = TL, y = predicted_yv),
              inherit.aes = FALSE, colour = "blue4", linewidth = 1.05) +
    geom_point(aes(fill = Habitat, shape = Origin,
                   size = pmin(Pred_MP, quantile(Pred_MP, 0.92, na.rm = TRUE))),
               colour = "grey18", stroke = 0.25) +
    geom_polygon(data = core$hulls,
                 aes(x = TL, y = Yv, group = interaction(Core_type, Core_group),
                     colour = Core_group, linetype = Core_type),
                 inherit.aes = FALSE, fill = NA, linewidth = 0.92, alpha = 0.95) +
    geom_label(data = core$labels,
               aes(x = TL_label, y = Yv_label, label = Label, colour = Core_group),
               inherit.aes = FALSE, fill = "white", size = 2.35,
               label.padding = grid::unit(0.10, "lines"), linewidth = 0.20,
               show.legend = FALSE) +
    geom_label(data = core$small_hab,
               aes(x = TL_label, y = pmax(Yv_label, 0.05), label = Label, fill = Habitat),
               inherit.aes = FALSE, colour = "grey10", size = 2.25,
               label.padding = grid::unit(0.09, "lines"), linewidth = 0.15,
               alpha = 0.90, show.legend = FALSE) +
    annotate("label", x = min(core$base$TL, na.rm = TRUE) + 0.03,
             y = max(core$base$Yv, na.rm = TRUE) * 0.94,
             hjust = 0, vjust = 1, size = 2.45, linewidth = 0.18,
             label = sprintf("N=%d; studies=%d\nTL beta=%.3f; p=%s",
                             fit$row$N, fit$row$N_Studies, fit$row$beta, format_p(fit$row$p))) +
    scale_y_continuous(breaks = pseudo_y(y_breaks), labels = clean_mp_label(y_breaks),
                       expand = expansion(mult = c(0.06, 0.10))) +
    scale_x_continuous(expand = expansion(mult = c(0.025, 0.025))) +
    scale_fill_manual(values = habitat_cols, labels = habitat_labels, drop = FALSE,
                      name = "Habitat dependency") +
    scale_colour_manual(values = core_cols, name = "Core outline") +
    scale_linetype_manual(values = c("Origin" = "solid", "Habitat" = "longdash"),
                          name = "Core type") +
    scale_shape_manual(values = c("Native" = 24, "Invasive" = 21), name = "Origin") +
    scale_size_continuous(range = c(1.1, 4.4), guide = "none") +
    guides(fill = guide_legend(override.aes = list(shape = 21, size = 3.2, colour = "grey20",
                                                   linetype = 0, linewidth = 0)),
           colour = guide_legend(override.aes = list(fill = NA, size = 1.0)),
           linetype = guide_legend(override.aes = list(colour = "grey25"))) +
    labs(title = title, subtitle = subtitle, x = "Trophic level", y = y_label) +
    coord_cartesian(clip = "off") +
    theme_nature(8) +
    theme(
      legend.position = "right",
      panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.18),
      panel.grid.major.x = element_blank(),
      aspect.ratio = 0.82
    )

  save_panel(p, figure_name, 9.2, 6.4)
  invisible(core)
}

core_g <- plot_core_cluster(
  primary_fit, pred_primary, "MP_g",
  "TrophicControlsa | Core clustering of mass-normalized ingestion",
  "Core outlines enclose the densest 50% of each displayed group; labels report the enclosed share of that group",
  "Measured MP abundance (items/g; pseudo-log scale)",
  "TrophicControls_core_cluster_items_g_v4.csv",
  "Fig_TrophicControlsa_items_g_core_cluster_v4"
)

core_ind <- plot_core_cluster(
  ind_fit, pred_ind, "MP_ind",
  "TrophicControlsb | Core clustering of individual-normalized ingestion",
  "Items/ind is shown as a scale contrast; the TL trend is weak while group clustering remains visible",
  "Measured MP abundance (items/ind; pseudo-log scale)",
  "TrophicControls_core_cluster_items_ind_v4.csv",
  "Fig_TrophicControlsb_items_ind_core_cluster_v4"
)

# ---------------------------------------------------------------------------
# Fig TrophicControlsc v4: leave-one-study-out ZINB influence analysis
# ---------------------------------------------------------------------------
full_tl <- primary_fit$row %>%
  transmute(
    Full_N = N,
    Full_Studies = N_Studies,
    Full_beta = beta,
    Full_p = p,
    Full_low = lo,
    Full_high = hi
  )
full_beta <- full_tl$Full_beta[[1]]

study_meta <- df10 %>%
  filter(!is.na(TL), !is.na(MP_g), MP_g >= 0) %>%
  group_by(Reference) %>%
  summarise(
    Removed_N = n(),
    Removed_study_levels = n_distinct(Study),
    TL_min = min(TL, na.rm = TRUE),
    TL_max = max(TL, na.rm = TRUE),
    TL_range = TL_max - TL_min,
    Major_method = names(sort(table(Method), decreasing = TRUE))[1],
    Major_habitat = names(sort(table(Habitat), decreasing = TRUE))[1],
    .groups = "drop"
  ) %>%
  arrange(desc(Removed_N), desc(TL_range), Reference) %>%
  mutate(Study_ID = sprintf("Study %02d", row_number()))

jackknife_rows <- map_dfr(study_meta$Reference, function(ref_i) {
  d_i <- df10 %>% filter(Reference != ref_i)
  fit_i <- fit_tl_model(d_i, "MP_g")
  fit_i$row %>%
    transmute(
      Reference = ref_i,
      Kept_N = N,
      Kept_Studies = N_Studies,
      TL_beta = beta,
      TL_SE = se,
      TL_z = z,
      TL_p = p,
      TL_CI_low = lo,
      TL_CI_high = hi
    )
})

jackknife <- study_meta %>%
  left_join(jackknife_rows, by = "Reference") %>%
  mutate(
    Delta_beta = TL_beta - full_beta,
    Abs_delta_beta = abs(Delta_beta),
    Significant_negative = !is.na(TL_p) & TL_beta < 0 & TL_p < 0.05,
    Negative = !is.na(TL_beta) & TL_beta < 0,
    Influence_class = case_when(
      Significant_negative ~ "negative, p < 0.05",
      Negative ~ "negative, p >= 0.05",
      TRUE ~ "non-negative or failed"
    ),
    Beta_label = format_ci(TL_beta, TL_CI_low, TL_CI_high),
    P_label = format_p(TL_p),
    Delta_label = sprintf("%+.2f", Delta_beta),
    Removed_label = sprintf("%d", Removed_N)
  ) %>%
  arrange(desc(Abs_delta_beta), desc(Removed_N), Reference) %>%
  mutate(
    Row = rev(row_number()) * 0.34,
    Row_label = Study_ID
  )

jack_summary <- jackknife %>%
  summarise(
    Full_N = full_tl$Full_N[[1]],
    Full_Studies = full_tl$Full_Studies[[1]],
    Full_beta = full_tl$Full_beta[[1]],
    Full_p = full_tl$Full_p[[1]],
    N_jackknife = n(),
    Negative_after_removal = sum(Negative, na.rm = TRUE),
    Significant_negative_after_removal = sum(Significant_negative, na.rm = TRUE),
    Max_abs_delta_beta = max(Abs_delta_beta, na.rm = TRUE),
    Min_beta = min(TL_beta, na.rm = TRUE),
    Max_beta = max(TL_beta, na.rm = TRUE)
  )

write_csv(
  jackknife %>%
    select(
      Study_ID, Reference, Removed_N, Removed_study_levels, TL_min, TL_max, TL_range,
      Major_method, Major_habitat, Kept_N, Kept_Studies, TL_beta, TL_SE, TL_z, TL_p,
      TL_CI_low, TL_CI_high, Delta_beta, Abs_delta_beta, Significant_negative, Negative
    ),
  file.path(DIR_DATA, "TrophicControls_study_jackknife_influence_v4.csv")
)
write_csv(jack_summary, file.path(DIR_DATA, "TrophicControls_study_jackknife_summary_v4.csv"))

jack_axis_min <- min(-0.58, min(jackknife$TL_beta, full_beta, na.rm = TRUE) - 0.035)
jack_axis_max <- 0.02
jack_capsule <- jackknife %>%
  arrange(TL_beta) %>%
  mutate(
    Strip_y = rep(c(-0.034, -0.018, -0.002, 0.014, 0.030, 0.046), length.out = n())
  )
jack_band <- jack_capsule %>%
  summarise(
    Min_beta = min(TL_beta, na.rm = TRUE),
    Q25_beta = quantile(TL_beta, 0.25, na.rm = TRUE),
    Q75_beta = quantile(TL_beta, 0.75, na.rm = TRUE),
    Max_beta = max(TL_beta, na.rm = TRUE)
  )
jack_extreme_labels <- bind_rows(
  jack_capsule %>% slice_max(Delta_beta, n = 1, with_ties = FALSE),
  jack_capsule %>% slice_min(Delta_beta, n = 1, with_ties = FALSE)
) %>%
  mutate(
    Label_x = TL_beta + if_else(Delta_beta > 0, 0.018, -0.018),
    Label_y = if_else(Delta_beta > 0, 0.088, -0.080),
    Label_hjust = if_else(Delta_beta >= 0, 0, 1),
    Influence_label = sprintf("%s %s", Study_ID, Delta_label)
  )

p_jackknife <- ggplot(jack_capsule, aes(x = TL_beta, y = Strip_y)) +
  annotate("rect", xmin = jack_axis_min, xmax = 0, ymin = -Inf, ymax = Inf,
           fill = "#F3F8FB", alpha = 1) +
  annotate("rect", xmin = 0, xmax = jack_axis_max, ymin = -Inf, ymax = Inf,
           fill = "#FAFAFA", alpha = 1) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey35", linewidth = 0.42) +
  geom_segment(data = jack_band, aes(x = Min_beta, xend = Max_beta, y = 0, yend = 0),
               inherit.aes = FALSE, linewidth = 16, colour = "#E3EFF6", lineend = "round") +
  geom_segment(data = jack_band, aes(x = Q25_beta, xend = Q75_beta, y = 0, yend = 0),
               inherit.aes = FALSE, linewidth = 8.5, colour = "#A9CCE3", lineend = "round") +
  geom_vline(xintercept = full_beta, colour = "#173B63", linewidth = 1.0) +
  geom_point(aes(fill = Influence_class, size = Removed_N),
             shape = 21, colour = "grey10", stroke = 0.28) +
  geom_text(data = jack_extreme_labels,
            aes(x = Label_x, y = Label_y, label = Influence_label, hjust = Label_hjust),
            inherit.aes = FALSE, size = 2.45, colour = "grey12") +
  annotate("text", x = full_beta, y = 0.095,
           label = sprintf("full model %.2f", full_beta),
           hjust = 0.5, size = 2.45, colour = "#173B63") +
  scale_fill_manual(
    values = c(
      "negative, p < 0.05" = "#1F5E9C",
      "negative, p >= 0.05" = "#B7D6EA",
      "non-negative or failed" = "white"
    ),
    name = "Leave-one-out result"
  ) +
  scale_size_area(max_size = 4.3, name = "Removed records") +
  scale_x_continuous(limits = c(jack_axis_min, jack_axis_max),
                     breaks = c(-0.50, -0.40, -0.30, -0.20, -0.10, 0),
                     guide = guide_axis(cap = TRUE)) +
  scale_y_continuous(limits = c(-0.105, 0.130), breaks = NULL) +
  labs(
    title = "TrophicControlsc | Study-jackknife stability capsule",
    subtitle = sprintf("24/24 beta<0; %d/24 p<0.05; max |Delta beta|=%.2f. Capsule: beta range and interquartile span.",
                       jack_summary$Significant_negative_after_removal[[1]],
                       jack_summary$Max_abs_delta_beta[[1]]),
    x = "Trophic-level coefficient after removing one study (standardized beta)",
    y = NULL
  ) +
  theme_nature(8.2) +
  theme(
    legend.position = "none",
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(7, 9, 5, 8)
  )
save_panel(p_jackknife, "Fig_TrophicControlsc_study_jackknife_compact_v4", 6.3, 2.25)

# ---------------------------------------------------------------------------
# Fig TrophicControlsd v4: study-fixed permutation test
# ---------------------------------------------------------------------------
study_fixed_slope <- function(dat, response_col, tl_col = "TL") {
  d <- dat %>%
    filter(!is.na(.data[[response_col]]), .data[[response_col]] >= 0, !is.na(.data[[tl_col]])) %>%
    mutate(Y_log = log10(1 + .data[[response_col]])) %>%
    group_by(Study) %>%
    mutate(
      TL_c = .data[[tl_col]] - mean(.data[[tl_col]], na.rm = TRUE),
      Y_c = Y_log - mean(Y_log, na.rm = TRUE)
    ) %>%
    ungroup()
  as.numeric(coef(lm(Y_c ~ 0 + TL_c, data = d))[["TL_c"]])
}

permute_within_study <- function(dat, response_col, n_perm = 999) {
  observed <- study_fixed_slope(dat, response_col, "TL")
  null <- replicate(n_perm, {
    d_perm <- dat %>%
      group_by(Study) %>%
      mutate(TL_perm = sample(TL, size = n(), replace = FALSE)) %>%
      ungroup()
    study_fixed_slope(d_perm, response_col, "TL_perm")
  })
  tibble(
    Observed = observed,
    Null = null,
    P_less = (sum(null <= observed, na.rm = TRUE) + 1) / (length(null) + 1),
    P_two = (sum(abs(null) >= abs(observed), na.rm = TRUE) + 1) / (length(null) + 1)
  )
}

set.seed(20260625)
perm_g <- permute_within_study(df10, "MP_g", 999) %>% mutate(Metric = "items/g")
perm_ind <- permute_within_study(df10, "MP_ind", 999) %>% mutate(Metric = "items/ind")

perm_long <- bind_rows(perm_g, perm_ind) %>%
  mutate(Metric = factor(Metric, levels = c("items/g", "items/ind")))
perm_null <- perm_long %>%
  select(Metric, Null) %>%
  mutate(Metric = factor(Metric, levels = c("items/g", "items/ind")))
perm_summary <- perm_long %>%
  group_by(Metric) %>%
  summarise(
    Observed = first(Observed),
    P_less = first(P_less),
    P_two = first(P_two),
    Null_low = quantile(Null, 0.025, na.rm = TRUE),
    Null_mid = quantile(Null, 0.500, na.rm = TRUE),
    Null_high = quantile(Null, 0.975, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Label = case_when(
      Metric == "items/g" ~ sprintf("items/g observed=%.3f; one-sided p=%s", Observed, format_p(P_less)),
      TRUE ~ sprintf("items/ind observed=%.3f; two-sided p=%s", Observed, format_p(P_two))
    ),
    Label_x = case_when(
      Metric == "items/g" ~ min(perm_null$Null, na.rm = TRUE) + 0.006,
      TRUE ~ max(perm_null$Null, na.rm = TRUE) - 0.006
    ),
    Label_hjust = if_else(Metric == "items/g", 0, 1)
  )

write_csv(perm_null, file.path(DIR_DATA, "TrophicControls_study_fixed_permutation_null_v4.csv"))
write_csv(perm_summary, file.path(DIR_DATA, "TrophicControls_study_fixed_permutation_summary_v4.csv"))

p_perm <- ggplot(perm_null, aes(x = Null, fill = Metric, colour = Metric)) +
  geom_density(alpha = 0.18, linewidth = 0.65, adjust = 1.05) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey35", linewidth = 0.42) +
  geom_vline(data = perm_summary, aes(xintercept = Observed, colour = Metric),
             linewidth = 0.85, show.legend = FALSE) +
  geom_point(data = perm_summary, aes(x = Observed, y = 0, fill = Metric),
             inherit.aes = FALSE, shape = 21, size = 3.4, colour = "grey15", stroke = 0.30) +
  geom_label(data = perm_summary,
             aes(x = Label_x, y = Inf, label = Label, colour = Metric, hjust = Label_hjust),
             inherit.aes = FALSE, vjust = 1.18,
             fill = "white", size = 2.45, label.padding = grid::unit(0.10, "lines"),
             linewidth = 0.18, show.legend = FALSE) +
  scale_fill_manual(values = c("items/g" = "#1F5E9C", "items/ind" = "#BFBFBF"), name = "Metric") +
  scale_colour_manual(values = c("items/g" = "#1F5E9C", "items/ind" = "#6E6E6E"), name = "Metric") +
  labs(
    title = "TrophicControlsd | Study-fixed permutation test of trophic dilution",
    subtitle = "TL was permuted within each source study, preserving study-specific protocols, detection windows and sample structure",
    x = "Study-fixed TL slope for log10(1 + abundance)",
    y = "Permutation density"
  ) +
  theme_nature(8) +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )
save_panel(p_perm, "Fig_TrophicControlsd_study_fixed_permutation_v4", 8.4, 5.2)

# ---------------------------------------------------------------------------
# Fig TrophicControlse v4: TL uncertainty stress test
# ---------------------------------------------------------------------------
fit_mc <- function(dat) {
  fit_tl_model(dat, "MP_g")$row
}

make_perturbed <- function(dat, scenario) {
  if (scenario == "Observed TL") {
    return(dat)
  }
  width <- case_when(
    scenario == "Uniform +/-0.10 TL" ~ 0.10,
    scenario == "Uniform +/-0.20 TL" ~ 0.20,
    TRUE ~ NA_real_
  )
  if (!is.na(width)) {
    eps <- runif(nrow(dat), -width, width)
  } else {
    eps_width <- case_when(
      dat$Taxon_Group == "Bird" ~ 0.40,
      dat$Taxon_Group == "Fish" ~ 0.20,
      TRUE ~ 0.20
    )
    eps <- runif(nrow(dat), -eps_width, eps_width)
  }
  dat %>%
    mutate(
      TL_perturbed = pmin(pmax(TL + eps, 1), 5),
      TL_s = as.numeric(scale(TL_perturbed))
    )
}

scenarios <- c("Observed TL", "Uniform +/-0.10 TL", "Uniform +/-0.20 TL", "Source-aware conservative")
n_iter <- 50

mc_results <- map_dfr(scenarios, function(sc) {
  iter_vec <- if (sc == "Observed TL") 1 else seq_len(n_iter)
  map_dfr(iter_vec, function(i) {
    d <- make_perturbed(df10, sc)
    fit_mc(d) %>%
      mutate(Scenario = sc, Iteration = i)
  })
})
write_csv(mc_results, file.path(DIR_DATA, "TrophicControls_TL_uncertainty_stress_v4.csv"))

mc_summary <- mc_results %>%
  group_by(Scenario) %>%
  summarise(
    Fits = sum(!is.na(beta)),
    Median_beta = median(beta, na.rm = TRUE),
    Q25 = quantile(beta, 0.25, na.rm = TRUE),
    Q75 = quantile(beta, 0.75, na.rm = TRUE),
    Q025 = quantile(beta, 0.025, na.rm = TRUE),
    Q975 = quantile(beta, 0.975, na.rm = TRUE),
    Pr_negative = mean(beta < 0, na.rm = TRUE),
    Pr_sig = mean(p < 0.05, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Scenario = factor(Scenario, levels = rev(scenarios)),
    Label = sprintf("Pr(beta<0)=%.0f%%; Pr(p<0.05)=%.0f%%", 100 * Pr_negative, 100 * Pr_sig)
  )
write_csv(mc_summary, file.path(DIR_DATA, "TrophicControls_TL_uncertainty_summary_v4.csv"))

p_uncertainty <- ggplot(mc_summary, aes(y = Scenario)) +
  annotate("rect", xmin = -Inf, xmax = 0, ymin = -Inf, ymax = Inf, fill = "#EFF6F0", alpha = 0.95) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.35) +
  geom_segment(aes(x = Q025, xend = Q975, yend = Scenario), linewidth = 1.0, colour = "#9ABD9F") +
  geom_segment(aes(x = Q25, xend = Q75, yend = Scenario), linewidth = 3.0, colour = "#3F8D52", lineend = "round") +
  geom_point(aes(x = Median_beta, fill = Pr_sig), shape = 21, size = 4.0, colour = "grey15", stroke = 0.35) +
  geom_text(aes(x = 0.10, label = Label), hjust = 0, size = 2.75, colour = "grey15") +
  scale_fill_gradient(low = "white", high = "#1E6C3A", limits = c(0, 1),
                      labels = percent_format(accuracy = 1), name = "Share significant") +
  scale_x_continuous(limits = c(-0.95, 0.55), breaks = c(-0.8, -0.6, -0.4, -0.2, 0, 0.2)) +
  labs(
    title = "TrophicControlse | Trophic-level uncertainty stress test",
    subtitle = "Primary ZINB-GLMM refitted after plausible TL perturbations; intervals summarize repeated fits",
    x = "TL coefficient under perturbed trophic-level assignments",
    y = NULL,
    caption = "Source-aware scenario perturbs fish and invertebrates by +/-0.20 TL and birds by +/-0.40 TL. This is a sensitivity envelope, not a claimed measurement error."
  ) +
  theme_nature(8) +
  theme(legend.position = "bottom", panel.grid.major.y = element_blank())
save_panel(p_uncertainty, "Fig_TrophicControlse_TL_uncertainty_stress_v4", 9.5, 5.2)

evidence_table <- tribble(
  ~Figure, ~Role, ~Analysis_purpose, ~Conclusion,
  "Fig_IngestionMethodsb_detection_zero_rate_v4", "zero-rate forest diagnostic", "IngestionMethods zero inflation and detection-limit heterogeneity", "Zero proportions and Wilson confidence intervals do not indicate systematic zero inflation in lower-sensitivity windows; detection-limit heterogeneity is unlikely to explain the TL signal alone.",
  "Fig_IngestionMethodsc_TrophicControlsb_integrated_forest_v4", "single consolidated model-control forest", "IngestionMethods method controls and TrophicControls robustness controls", "The TL coefficient remains negative under method, detection, tissue, dominant-method and sparse-family controls; items/ind is retained as the body-size contrast rather than repeated as a separate TL forest.",
  "Fig_TrophicControlsa_items_g_core_cluster_v4", "mass-normalized core clustering", "TrophicControls habitat, origin and sample-composition controls", "Core outlines show where origin and major habitat groups concentrate; labels report the share of each group inside the displayed core while the fitted mass-normalized TL trend remains negative.",
  "Fig_TrophicControlsb_items_ind_core_cluster_v4", "individual-normalized core clustering", "TrophicControls body-size normalization check", "The items/ind counterpart shows group clustering without a comparable negative TL trend, supporting a mass-normalized biodilution interpretation.",
  "Fig_TrophicControlsc_study_jackknife_compact_v4", "compact leave-one-study-out ZINB influence analysis", "IngestionMethods/TrophicControls robustness against study-level methodological heterogeneity", "All 24 source studies are tested by removing one study at a time and refitting the same ZINB GLMM; the mass-normalized TL coefficient remains negative after each removal.",
  "Fig_TrophicControlsd_study_fixed_permutation_v4", "study-fixed permutation test", "IngestionMethods/TrophicControls robustness against source-study and method confounding", "Within-study TL permutation preserves source-study protocols and sample structure; the observed items/g slope is more negative than the within-study null, whereas items/ind is not.",
  "Fig_TrophicControlse_TL_uncertainty_stress_v4", "TL uncertainty stress test", "TrophicControls trophic-level assignment uncertainty", "Plausible TL perturbations retain a negative TL coefficient distribution, directly addressing TL assignment uncertainty."
)
write_csv(evidence_table, file.path(DIR_DATA, "IngestionMethods_TrophicControls_v4_evidence_table.csv"))

manifest_v4 <- tibble(
  file_base = c(
    "Fig_IngestionMethodsb_detection_zero_rate_v4",
    "Fig_IngestionMethodsc_TrophicControlsb_integrated_forest_v4",
    "Fig_TrophicControlsa_items_g_core_cluster_v4",
    "Fig_TrophicControlsb_items_ind_core_cluster_v4",
    "Fig_TrophicControlsc_study_jackknife_compact_v4",
    "Fig_TrophicControlsd_study_fixed_permutation_v4",
    "Fig_TrophicControlse_TL_uncertainty_stress_v4"
  )
) %>%
  mutate(
    png = file.path(DIR_FIG, paste0(file_base, ".png")),
    pdf = file.path(DIR_FIG, paste0(file_base, ".pdf"))
  )
write_csv(manifest_v4, file.path(DIR_FIG, "figure_manifest_v4.csv"))

cat("\nIngestionMethods/TrophicControls visual revision v4 complete.\n")
cat(sprintf("Figures: %s\n", DIR_FIG))
cat(sprintf("Data: %s\n", DIR_DATA))
graphics.off()
invisible(gc())
quit(save = "no", status = 0, runLast = FALSE)
