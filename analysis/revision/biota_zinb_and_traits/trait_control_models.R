suppressPackageStartupMessages({
  library(readxl)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(glmmTMB)
})

ROOT <- Sys.getenv("LAKE_MP_REPO_ROOT", normalizePath(getwd(), winslash = "/", mustWork = TRUE))
DATA_ROOT <- Sys.getenv("LAKE_MP_DATA_ROOT", file.path(ROOT, "data"))
OUTPUT_DIR <- Sys.getenv("LAKE_MP_OUTPUT_DIR", file.path(ROOT, "outputs"))
data_dir <- file.path(OUTPUT_DIR, "biota_zinb_and_traits", "data")
input_path <- file.path(DATA_ROOT, "evidence", "biota_zinb_and_traits", "Supplementary_Data_2_WITH_PRED_MP.xlsx")
trait_path <- file.path(DATA_ROOT, "evidence", "biota", "R1C10_fish_trait_metadata_v1.csv")

stopifnot(file.exists(input_path), file.exists(trait_path))

raw <- read_excel(input_path)
traits <- read_csv(trait_path, show_col_types = FALSE)
stopifnot(nrow(raw) == 120L, nrow(traits) == 109L)

# The trait audit preserves the fish-record order from the collapsed source data.
fish_rows <- which(raw$Species_or_Sample_Type %in% traits$Original_Label)
if (length(fish_rows) != nrow(traits) ||
    !all(raw$Species_or_Sample_Type[fish_rows] == traits$Original_Label) ||
    !all(raw$Reference[fish_rows] == traits$Study_Reference)) {
  stop("Trait rows do not align one-to-one with the fish records in the modelling workbook.")
}

fish <- raw[fish_rows, ] %>%
  mutate(Record_ID = traits$Record_ID) %>%
  left_join(
    traits %>%
      select(
        Record_ID, Accepted_Name, SpecCode, Max_Length_cm, Max_Length_Type,
        Body_Size_Source, Feeding_Guild, Feeding_Evidence, Feeding_Source,
        Trait_Version
      ),
    by = "Record_ID"
  ) %>%
  transmute(
    Record_ID,
    Study = factor(Reference),
    Species_or_Sample_Type,
    Accepted_Name,
    SpecCode,
    TL = Trophic_level,
    Pred_MP = Predicted_Lake_MP,
    MP_g = Measured_Abundance_g,
    MP_ind = Measured_Abundance_ind,
    Origin = factor(Origin),
    Habitat = factor(`Habitat dependency`, levels = c("F-Only", "FM", "FT", "FMT")),
    Max_Length_cm,
    Max_Length_Type,
    Body_Size_Source,
    Feeding_Guild,
    Feeding_Evidence,
    Feeding_Source,
    Trait_Version
  )

guild_levels <- c(
  "Generalist/variable", "Herbivore/browser",
  "Planktivore", "Macrofauna predator"
)

model_data <- fish %>%
  filter(
    !is.na(TL), !is.na(Pred_MP), !is.na(MP_g), MP_g >= 0,
    !is.na(MP_ind), MP_ind >= 0, !is.na(Origin), !is.na(Habitat),
    !is.na(Max_Length_cm), Max_Length_cm > 0,
    Feeding_Guild %in% guild_levels
  ) %>%
  mutate(
    Study = droplevels(Study),
    Origin = droplevels(Origin),
    Habitat = droplevels(Habitat),
    Feeding_Guild = factor(Feeding_Guild, levels = guild_levels),
    Log_Max_Length = log(Max_Length_cm),
    Pred_s = as.numeric(scale(log1p(Pred_MP))),
    TL_s = as.numeric(scale(TL)),
    Body_Size_s = as.numeric(scale(Log_Max_Length))
  )

stopifnot(
  nrow(model_data) >= 70L,
  n_distinct(model_data$Study) >= 15L,
  all(table(model_data$Feeding_Guild) >= 8L),
  all(model_data$Trait_Version == "FishBase 24.07")
)

fit_model <- function(model_name, response, rhs, response_label) {
  dat <- model_data %>% mutate(Yc = round(.data[[response]] * 1000))
  form <- as.formula(paste("Yc ~", rhs))
  fit <- suppressWarnings(
    glmmTMB(
      form,
      ziformula = ~1,
      family = nbinom2(link = "log"),
      data = dat
    )
  )

  converged <- isTRUE(fit$fit$convergence == 0)
  pd_hessian <- isTRUE(fit$sdr$pdHess)
  if (!converged || !pd_hessian) {
    stop(sprintf("Model '%s' failed convergence/Hessian checks.", model_name))
  }

  coef_table <- as.data.frame(summary(fit)$coefficients$cond) %>%
    tibble::rownames_to_column("Term") %>%
    transmute(
      Model = model_name,
      Response = response_label,
      Component = "conditional",
      Term,
      Estimate = Estimate,
      Std_Error = `Std. Error`,
      z = `z value`,
      P = `Pr(>|z|)`,
      CI_low = Estimate - 1.96 * Std_Error,
      CI_high = Estimate + 1.96 * Std_Error
    )

  tl <- coef_table %>% filter(Term == "TL_s")
  if (nrow(tl) != 1L) stop(sprintf("TL coefficient missing from model '%s'.", model_name))

  summary_row <- tibble(
    Model = model_name,
    Response = response_label,
    Formula = paste(deparse(form), collapse = " "),
    N = nrow(dat),
    N_Studies = n_distinct(dat$Study),
    TL_beta = tl$Estimate,
    TL_SE = tl$Std_Error,
    TL_z = tl$z,
    TL_p = tl$P,
    TL_CI_low = tl$CI_low,
    TL_CI_high = tl$CI_high,
    AIC = AIC(fit),
    LogLik = as.numeric(logLik(fit)),
    Converged = converged,
    Positive_definite_Hessian = pd_hessian,
    Body_Size_Metric = "FishBase species.Length (maximum length, cm)",
    Feeding_Mode_Source = "FishBase ecology.FeedingType; fooditems fallback",
    Trait_Version = "FishBase 24.07",
    P_Test = "Wald z"
  )

  list(model = fit, summary = summary_row, coefficients = coef_table)
}

common <- "Pred_s + TL_s + Origin + Habitat"
random <- "(1 | Study)"

fits <- list(
  fit_model(
    "Matched fish baseline", "MP_g",
    paste(common, random, sep = " + "), "items/g"
  ),
  fit_model(
    "+ independent body size", "MP_g",
    paste(common, "Body_Size_s", random, sep = " + "), "items/g"
  ),
  fit_model(
    "+ feeding mode", "MP_g",
    paste(common, "Feeding_Guild", random, sep = " + "), "items/g"
  ),
  fit_model(
    "+ body size + feeding mode", "MP_g",
    paste(common, "Body_Size_s + Feeding_Guild", random, sep = " + "), "items/g"
  ),
  fit_model(
    "Items/individual + both controls", "MP_ind",
    paste(common, "Body_Size_s + Feeding_Guild", random, sep = " + "), "items/individual"
  )
)

model_summary <- bind_rows(lapply(fits, `[[`, "summary"))
coefficient_table <- bind_rows(lapply(fits, `[[`, "coefficients"))

mass_rows <- model_summary %>% filter(Response == "items/g")
stopifnot(n_distinct(mass_rows$N) == 1L, n_distinct(mass_rows$N_Studies) == 1L)

pearson_test <- cor.test(model_data$TL, model_data$Log_Max_Length, method = "pearson")
spearman_test <- suppressWarnings(cor.test(model_data$TL, model_data$Log_Max_Length, method = "spearman", exact = FALSE))

lm_design <- lm(
  log1p(MP_g) ~ Pred_s + TL_s + Origin + Habitat + Body_Size_s + Feeding_Guild,
  data = model_data
)
vif_raw <- car::vif(lm_design)
if (is.matrix(vif_raw)) {
  vif_table <- as.data.frame(vif_raw) %>%
    tibble::rownames_to_column("Term") %>%
    transmute(
      Term,
      GVIF = GVIF,
      Df = Df,
      Adjusted_GVIF = `GVIF^(1/(2*Df))`
    )
} else {
  vif_table <- tibble(
    Term = names(vif_raw),
    GVIF = as.numeric(vif_raw),
    Df = 1,
    Adjusted_GVIF = sqrt(as.numeric(vif_raw))
  )
}

guild_counts <- model_data %>%
  count(Feeding_Guild, name = "Records") %>%
  left_join(
    model_data %>%
      group_by(Feeding_Guild) %>%
      summarise(Studies = n_distinct(Study), .groups = "drop"),
    by = "Feeding_Guild"
  )

diagnostics <- bind_rows(
  tibble(
    Metric = c(
      "Trait-complete fish records",
      "Trait-complete studies",
      "TL versus log maximum length Pearson r",
      "TL versus log maximum length Pearson P",
      "TL versus log maximum length Spearman rho",
      "TL versus log maximum length Spearman P",
      "Maximum adjusted GVIF",
      "Mass-normalized TL coefficients negative",
      "Mass-normalized TL coefficients P < 0.05"
    ),
    Value = c(
      nrow(model_data),
      n_distinct(model_data$Study),
      unname(pearson_test$estimate),
      pearson_test$p.value,
      unname(spearman_test$estimate),
      spearman_test$p.value,
      max(vif_table$Adjusted_GVIF, na.rm = TRUE),
      sum(mass_rows$TL_beta < 0),
      sum(mass_rows$TL_p < 0.05)
    ),
    Details = c(
      "All four items/g models use this identical subset",
      "Random-intercept study levels",
      "Independent FishBase maximum-length proxy",
      "Two-sided correlation test",
      "Independent FishBase maximum-length proxy",
      "Two-sided correlation test",
      paste0(vif_table$Term[which.max(vif_table$Adjusted_GVIF)], "; threshold audit < 3"),
      sprintf("%d of %d matched items/g models", sum(mass_rows$TL_beta < 0), nrow(mass_rows)),
      sprintf("%d of %d matched items/g models", sum(mass_rows$TL_p < 0.05), nrow(mass_rows))
    )
  ),
  guild_counts %>%
    transmute(
      Metric = paste0("Feeding guild: ", Feeding_Guild),
      Value = Records,
      Details = paste0(Studies, " studies")
    )
)

complete_cases <- model_data %>%
  transmute(
    Record_ID,
    Study = as.character(Study),
    Species_or_Sample_Type,
    Accepted_Name,
    TL,
    Pred_MP,
    MP_g,
    MP_ind,
    Origin = as.character(Origin),
    Habitat = as.character(Habitat),
    Max_Length_cm,
    Log_Max_Length,
    Feeding_Guild = as.character(Feeding_Guild),
    Body_Size_Source,
    Feeding_Evidence,
    Feeding_Source,
    Trait_Version
  )

write_csv(model_summary, file.path(data_dir, "R1C10_independent_trait_model_summary_v1.csv"), na = "")
write_csv(complete_cases, file.path(data_dir, "R1C10_independent_trait_complete_cases_v1.csv"), na = "")
write_csv(coefficient_table, file.path(data_dir, "R1C10_independent_trait_coefficients_v1.csv"), na = "")
write_csv(diagnostics, file.path(data_dir, "R1C10_independent_trait_diagnostics_v1.csv"), na = "")
write_csv(vif_table, file.path(data_dir, "R1C10_independent_trait_vif_v1.csv"), na = "")
saveRDS(lapply(fits, `[[`, "model"), file.path(data_dir, "R1C10_independent_trait_models_v1.rds"))

print(model_summary %>% select(Model, Response, N, N_Studies, TL_beta, TL_CI_low, TL_CI_high, TL_p, AIC))
print(diagnostics)
