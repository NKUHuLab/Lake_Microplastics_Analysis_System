# --- 0. Load Required Libraries ---
# 确保这些包已安装: install.packages(c("tidyverse", "readr", "lme4", "ggeffects", "Cairo", "viridis", "ggdist", "scales", "patchwork", "nloptr", "MASS", "glmmTMB"))

library(tidyverse)
library(readr)
library(lme4)         
library(ggeffects)
library(Cairo)
library(viridis)
library(ggdist)
library(patchwork)    
library(nloptr)       
library(MASS)         
library(glmmTMB)      # (!!!) 关键包: 用于 ZINB-GLMM

# --- 1. Define File Paths and Key Columns ---
FILE_PATH <- "E:/lake-MP-W/data/Bio/validation_analysis_data1.csv"
OUTPUT_DIR <- "E:/lake-MP-W/data/Bio"

# 定义关键列名，确保一致性
COL_MEASURED_G <- "Measured_Abundance_g"
COL_MEASURED_IND <- "Measured_Abundance_ind"
COL_PREDICTED <- "Predicted_Lake_MP"
COL_TROPHIC <- "Trophic_level"
COL_ORIGIN <- "Origin"
COL_HABITAT <- "Habitat dependency" 
COL_STUDY_ID <- "Reference"

# --- 2. Step 1 (Workflow): Clean and Prepare Biota Data ---
message("Step 1: Loading and cleaning biota data...")
df_raw <- readr::read_csv(FILE_PATH, col_types = cols(.default = "c"))

df_model_data_raw <- df_raw %>%
  mutate(
    Measured_Abundance_g = as.numeric(stringr::str_extract(.data[[COL_MEASURED_G]], "(\\d+\\.?\\d*)")),
    Measured_Abundance_ind = as.numeric(stringr::str_extract(.data[[COL_MEASURED_IND]], "(\\d+\\.?\\d*)")),
    Predicted_Lake_MP = as.numeric(.data[[COL_PREDICTED]]),
    Trophic_level = as.numeric(.data[[COL_TROPHIC]]),
    Origin = as.factor(.data[[COL_ORIGIN]]),
    Habitat_dependency = as.factor(.data[[COL_HABITAT]]),
    Study_ID = as.factor(.data[[COL_STUDY_ID]])
  )

message("Step 2.5: Checking for NA in key columns (Predicted_Lake_MP, Trophic_level, Origin, Study_ID)...")
cols_to_check_for_na <- c("Predicted_Lake_MP", "Trophic_level", "Origin", "Study_ID") 

df_dropped_key_na <- df_model_data_raw %>%
  filter(if_any(all_of(cols_to_check_for_na), is.na))

if(nrow(df_dropped_key_na) > 0) {
  message(paste("--- Dropping", nrow(df_dropped_key_na), "rows due to NA in key columns: ---"))
  print(df_dropped_key_na %>% select(any_of(c("Species_or_Sample_Type", "Reference", cols_to_check_for_na))))
  message("---------------------------------------------------------------")
} else {
  message("--- No rows dropped due to NA in key columns. ---")
}

df_model_data_raw <- df_model_data_raw %>%
  drop_na(all_of(cols_to_check_for_na))

message("Step 1 complete. Raw data loaded and initial cleaning applied.")


# --- (!!!) 新增: 步骤 3 - 将整个分析封装到一个函数中 ---
message("Step 3: Defining the full analysis-to-plot function (using glmmTMB ZINB)...")

generate_analysis_plot <- function(raw_data, col_measure_name, col_unit_name) {
  
  # --- 3A: 准备特定于此测量的数据 ---
  col_measure_sym <- rlang::sym(col_measure_name)
  
  message(paste("--- Checking for NA in measurement column:", col_measure_name, "---"))
  df_model_data <- raw_data %>%
    filter(!is.na(!!col_measure_sym)) %>%
    mutate(
      measurement_original = !!col_measure_sym,
      measurement_scaled_1000 = measurement_original * 1000,
      measurement_counts = round(measurement_scaled_1000)
    ) %>%
    filter(measurement_counts >= 0)
  
  if(nrow(df_model_data) < 30) {
    warning(paste("Warning for", col_measure_name, ": Only", nrow(df_model_data), "clean data points. Model may fail to converge."))
  } else {
    message(paste("Data prep complete for", col_measure_name, ". Proceeding with", nrow(df_model_data), "valid data points."))
  }
  
  message("Step 3.1: Stabilizing GLMM. Checking for and removing 'Study_ID' groups with only 1 observation...")
  study_id_counts <- df_model_data %>% count(Study_ID, sort = TRUE)
  
  studies_to_keep <- study_id_counts %>% filter(n > 1) %>% pull(Study_ID)
  studies_to_remove <- study_id_counts %>% filter(n == 1) %>% pull(Study_ID)
  
  original_n <- nrow(df_model_data)
  
  if(length(studies_to_remove) > 0) {
    df_model_data <- df_model_data %>% filter(Study_ID %in% studies_to_keep)
    final_n <- nrow(df_model_data)
    message(paste("--- Removed", original_n - final_n, "observations from", length(studies_to_remove), "studies that had only 1 data point."))} else {
      message("--- All studies have n > 1. No observations removed. Proceeding with GLMM.")
    }
  
  
  # --- 3B: 准备预测变量 ---
  message("Step 3.2: Preparing predictors for model...")
  
  df_model_data <- df_model_data %>%
    mutate(
      Predicted_MP_scaled = scale(log1p(Predicted_Lake_MP)),
      Trophic_level_scaled = scale(Trophic_level)
    )
  message("Applied predictor scaling (log1p & scale).")
  
  trophic_mean <- attr(df_model_data$Trophic_level_scaled, "scaled:center")
  trophic_sd <- attr(df_model_data$Trophic_level_scaled, "scaled:scale")
  
  plot_range_unscaled <- c(2.0, 5.0) 
  
  plot_range_scaled <- (plot_range_unscaled - trophic_mean) / trophic_sd
  
  ggpredict_terms_str <- paste0("Trophic_level_scaled [", plot_range_scaled[1], ":", plot_range_scaled[2], ", n=100]")
  
  message(paste("--- Plotting fix: Forcing line prediction over unscaled range", 
                plot_range_unscaled[1], "to", plot_range_unscaled[2], "---"))
  
  
  # --- 3C: (!!!) 拟合 Zero-Inflated Negative Binomial GLMM (ZINB-GLMM) ---
  message("Step 3.3: Fitting GLMM (Teaching Demonstration Version)...")
  
  zero_count <- sum(df_model_data$measurement_counts == 0)
  total_count <- nrow(df_model_data)
  zero_proportion <- zero_count / total_count
  message(paste("--- Proportion of zeros in 'measurement_counts':", round(zero_proportion * 100, 2), "% ---"))
  
  use_ziformula <- FALSE
  if (zero_proportion > 0.12) {
    message("--- High proportion of zeros detected. Using Zero-Inflated Negative Binomial model (zi=~1). ---")
    use_ziformula <- TRUE
    zi_formula_to_use <- ~1 
  } else {
    message("--- Low proportion of zeros. Using standard Negative Binomial model. ---")
  }
  
  formula_complex <- measurement_counts ~ Predicted_MP_scaled + Trophic_level_scaled + Origin + (1 + Trophic_level_scaled | Study_ID)
  formula_simple <- measurement_counts ~ Predicted_MP_scaled + Trophic_level_scaled + Origin + (1 | Study_ID)
  
  glmm_final <- NULL
  model_converged <- FALSE
  
  message("--- Attempting to fit ADVANCED model (with Random Slope for Trophic_level)... ---")
  
  tryCatch({
    if (use_ziformula) {
      glmm_final <- glmmTMB::glmmTMB(
        formula = formula_complex,
        ziformula = zi_formula_to_use,
        data = df_model_data,
        family = nbinom2
      )
    } else { # 不使用零膨胀
      glmm_final <- glmmTMB::glmmTMB(
        formula = formula_complex,
        data = df_model_data,
        family = nbinom2
      )
    }
    
    if (!is.null(glmm_final$opt$message) && grepl("converged", glmm_final$opt$message, ignore.case = TRUE)) {
      model_converged <- TRUE
    } else if (is.null(glmm_final$opt$message) && !is.null(glmm_final$sdr)) { 
      model_converged <- TRUE
    }
    
    if (model_converged) {
      message("--- SUCCESS: Advanced model (Random Slope) fitted successfully. ---")
    } else {
      message("--- WARNING: Advanced model (Random Slope) did not explicitly converge. ---")
    }
    
  }, error = function(e) {
    message(paste("--- ERROR fitting Advanced model (Random Slope):", e$message, "---"))
    model_converged <<- FALSE 
  }, warning = function(w) {
    message(paste("--- WARNING during Advanced model (Random Slope) fit:", w$message, "---"))
  })
  
  if (!model_converged) {
    message("--- WARNING: Advanced model (Random Slope) failed or did not converge. ---")
    message("--- Re-fitting with SIMPLER model (Random Intercept only)... ---")
    
    tryCatch({
      if (use_ziformula) {
        glmm_final <- glmmTMB::glmmTMB(
          formula = formula_simple,
          ziformula = zi_formula_to_use,
          data = df_model_data,
          family = nbinom2
        )
      } else { # 不使用零膨胀
        glmm_final <- glmmTMB::glmmTMB(
          formula = formula_simple,
          data = df_model_data,
          family = nbinom2
        )
      }
      if (!is.null(glmm_final$opt$message) && grepl("converged", glmm_final$opt$message, ignore.case = TRUE)) {
        message("--- Simpler model (Random Intercept) fitted successfully. ---")
      } else if (is.null(glmm_final$opt$message) && !is.null(glmm_final$sdr)) {
        message("--- Simpler model (Random Intercept) fitted successfully. ---")
      } else {
        message("--- WARNING: Simpler model (Random Intercept) did not explicitly converge. Proceeding with potentially non-converged model. ---")
      }
    }, error = function(e) {
      message(paste("--- CRITICAL ERROR: Simpler model (Random Intercept) also failed:", e$message, "---"))
      stop("Both advanced and simpler GLMM models failed to fit. Please check your data or model structure.")
    })
  }
  
  message("Step 3.3 complete. Model has been fitted.")
  
  message("--- Model Coefficients (Conditional and Zero-Inflation parts) ---")
  print(summary(glmm_final)) 
  message("-------------------------------------------------------------")
  
  
  # --- 3D: Step 4 (Workflow): Calculate Predictions for Fused Plot ---
  message("Step 4: Calculating marginal effects for Fused plot...")
  
  pred_ancova <- ggeffects::ggpredict(
    glmm_final,
    terms = ggpredict_terms_str, 
    type = "count" 
  ) %>%
    mutate(
      Trophic_Level_unscaled = (x * trophic_sd) + trophic_mean,
      predicted_original = predicted / 1000,
      conf.low_original = conf.low / 1000,
      conf.high_original = conf.high / 1000
    )
  
  message("Step 4 complete. Predictions for *overall* (and wider) trend are ready.")
  
  
  # --- 3E: Generate and Save the (Innovative) Fused Plot ---
  message("Step 5: Generating and saving the final 'Fused' plot...")
  
  color_palette <- viridis::viridis_pal(option = "D")(length(unique(df_model_data$Origin)))
  shape_palette <- c(16, 17, 18, 15, 8)[1:length(unique(df_model_data$Origin))]
  
  data_max <- max(df_model_data$measurement_original, na.rm = TRUE)
  min_positive <- min(df_model_data$measurement_original[df_model_data$measurement_original > 0], na.rm = TRUE)
  
  log_breaks <- c()
  if (is.finite(min_positive) && is.finite(data_max) && min_positive > 0 && data_max > 0) {
    log_breaks <- 10^seq(floor(log10(min_positive)), ceiling(log10(data_max)), by = 1)
  }
  
  y_breaks <- sort(unique(c(0, log_breaks)))
  y_breaks <- y_breaks[y_breaks >= 0]
  
  y_scale_settings <- scale_y_continuous(
    trans = scales::pseudo_log_trans(sigma = 0.05, base = 10),
    breaks = y_breaks,
    labels = scales::label_comma()
  )
  
  
  # --- 3E-1: 创建主图表 (Main Plot) ---
  main_plot <- ggplot(data = df_model_data) +
    geom_ribbon(
      data = pred_ancova,
      aes(x = Trophic_Level_unscaled, ymin = conf.low_original, ymax = conf.high_original),
      alpha = 0.3, show.legend = FALSE, fill = "dodgerblue" 
    ) +
    geom_line(
      data = pred_ancova,
      aes(x = Trophic_Level_unscaled, y = predicted_original),
      color = "blue4", 
      linewidth = 1.3
    ) +
    geom_point(aes(x = Trophic_level, y = measurement_original, color = Origin, shape = Origin), 
               alpha = 0.6, size = 2) +  
    y_scale_settings + 
    scale_color_manual(name = "Species Origin (物种来源)", values = color_palette) + 
    scale_shape_manual(name = "Species Origin (物种来源)", values = shape_palette) + 
    labs(
      title = paste(
        ifelse(use_ziformula, "ZINB-GLMM", "NB-GLMM"),
        "Predicted Trend Line" 
      ),
      subtitle = paste("Analysis based on", col_measure_name),
      x = "Trophic Level (营养级)",
      y = paste("Measured MP Abundance (items/", col_unit_name, ") [Pseudo-Log Scale]")
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 11),
      aspect.ratio = 1, 
      legend.position = c(0.05, 0.95), 
      legend.justification = c(0, 1), 
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3) 
    )
  
  # --- 3E-2: (!!!) 新增: 恢复独立的云雨图 ---
  raincloud_plot <- ggplot(data = df_model_data) +
    ggdist::stat_halfeye(
      aes(x = 0, y = measurement_original, fill = Origin), 
      .width = c(0.5, 0.95), 
      slab_alpha = 0.5,
      position = position_dodge(width = 0.3),
      
      # (!!!) 关键修改: 
      # 'scale' 参数控制云 (slab) 的最大宽度 (您所说的“峰值高度”)
      # 默认值是 1。调小此值 (例如 0.6) 将使峰值“变平”或“变窄”
      scale = 0.2
      
    ) +
    y_scale_settings + 
    scale_fill_manual(values = color_palette, name = "Species Origin (物种来源)") + 
    labs(
      title = "Data Distribution by Origin",
      subtitle = paste("Based on", col_measure_name),
      x = NULL, y = NULL
    ) + 
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      axis.title = element_blank(),     
      axis.text.y = element_blank(),    
      axis.ticks.y = element_blank(),   
      axis.text.x = element_blank(),    
      axis.ticks.x = element_blank(),   
      panel.grid = element_blank(),     
      panel.border = element_blank(),
      legend.position = c(0.05, 0.95), 
      legend.justification = c(0, 1), 
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3)
    )
  
  
  # --- 3F: 返回图表和模型对象 ---
  return(list(
    main_plot = main_plot, 
    raincloud_plot = raincloud_plot, 
    model_summary = summary(glmm_final), 
    glmm_object = glmm_final
  ))
}

# --- (!!!) 步骤 4 - 执行两次分析并处理图表输出 ---

# --- 分析 1: items/g ---
message("\n--- STARTING ANALYSIS 1: Measured_Abundance_g (DEMO Version) ---")
results_g <- generate_analysis_plot(
  raw_data = df_model_data_raw,
  col_measure_name = COL_MEASURED_G,
  col_unit_name = "g"
)
message("--- FINISHED ANALYSIS 1 ---\n")

# --- 显示和保存 Measured_Abundance_g 的图表 ---
message("--- Displaying and saving plots for Measured_Abundance_g (as PDF) ---")

print(results_g$main_plot)
ggsave(
  filename = file.path(OUTPUT_DIR, "Fused_Plot_Abundance_g.pdf"), 
  plot = results_g$main_plot,
  width = 8, height = 8, device = "pdf" 
)

print(results_g$raincloud_plot)
ggsave(
  filename = file.path(OUTPUT_DIR, "Raincloud_Plot_Abundance_g.pdf"), 
  plot = results_g$raincloud_plot,
  width = 5, height = 7, device = "pdf" # (!!!) 已根据您上个请求调窄了宽度
)
message(paste("Saved 2 PDF plots for items/g to", OUTPUT_DIR, "\n"))


# --- 分析 2: items/ind ---
message("\n--- STARTING ANALYSIS 2: Measured_Abundance_ind (DEMO Version) ---")
results_ind <- generate_analysis_plot(
  raw_data = df_model_data_raw,
  col_measure_name = COL_MEASURED_IND,
  col_unit_name = "ind"
)
message("--- FINISHED ANALYSIS 2 ---\n")

# --- 显示和保存 Measured_Abundance_ind 的图表 ---
message("--- Displaying and saving plots for Measured_Abundance_ind (as PDF) ---")

print(results_ind$main_plot)
ggsave(
  filename = file.path(OUTPUT_DIR, "Fused_Plot_Abundance_ind.pdf"), 
  plot = results_ind$main_plot,
  width = 8, height = 8, device = "pdf"
)

print(results_ind$raincloud_plot)
ggsave(
  filename = file.path(OUTPUT_DIR, "Raincloud_Plot_Abundance_ind.pdf"), 
  plot = results_ind$raincloud_plot,
  width = 5, height = 7, device = "pdf" # (!!!) 已根据您上个请求调窄了宽度
)
message(paste("Saved 2 PDF plots for items/ind to", OUTPUT_DIR, "\n"))

message("--- For teaching demonstration, review the full model summaries above. ---")