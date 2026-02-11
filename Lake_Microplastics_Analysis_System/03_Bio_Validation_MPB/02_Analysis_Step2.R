# --- 0. Load Required Libraries ---
# 确保这些包已安装: install.packages(c("tidyverse", "readr", "lme4", "ggeffects", "Cairo", "viridis", "nlme", "patchwork", "scales"))

library(tidyverse)  # For data manipulation (dplyr, stringr) and plotting (ggplot2)
library(readr)    # For loading the CSV file
library(lme4)     # For LMM (Loaded but not primarily used, keeping for context)
library(ggeffects)  # For calculating and plotting marginal effects from models
library(Cairo)    # For high-quality, editable PDF output
library(viridis)   # For Viridis color scales
library(nlme)     # Key package: for robust LMM (Linear Mixed Model)
library(patchwork)  # 关键新增：用于组合多个 ggplot 图表

# --- 1. Define File Paths and Key Columns ---
# (!!!) 务必将 FILE_PATH 和 OUTPUT_DIR 替换为您的实际路径
FILE_PATH <- "E:/lake-MP-W/data/Bio/validation_analysis_data.csv"
OUTPUT_DIR <- "E:/lake-MP-W/data/Bio"

# 根据您的工作流定义列名
COL_MEASURED_G <- "Measured_Abundance_g"
COL_MEASURED_IND <- "Measured_Abundance_ind"
COL_PREDICTED <- "Predicted_Lake_MP"
COL_TROPHIC <- "Trophic_level"
COL_ORIGIN <- "Origin"
COL_HABITAT <- "Habitat dependency"
COL_STUDY_ID <- "Reference" # 使用 'Reference' 作为 'Study_ID' 的代理

# --- 2. Step 1 (Workflow): Clean and Prepare Biota Data ---
message("Step 1: Loading and cleaning biota data...")

df_raw <- readr::read_csv(FILE_PATH, col_types = cols(.default = "c"))

# 创建清理后的数据表
df_model_data <- df_raw %>%
  filter(!grepl("Surface water|sediment|water|air",
                .data[[COL_STUDY_ID]],
                ignore.case = TRUE)) %>%
  filter(!grepl("Surface water|sediment|water|air",
                .data[["Species_or_Sample_Type"]],
                ignore.case = TRUE)) %>%
  mutate(
    Measured_Abundance_g = as.numeric(stringr::str_extract(.data[[COL_MEASURED_G]], "(\\d+\\.?\\d*)")),
    Measured_Abundance_ind = as.numeric(stringr::str_extract(.data[[COL_MEASURED_IND]], "(\\d+\\.?\\d*)")),
    Predicted_Lake_MP = as.numeric(.data[[COL_PREDICTED]]),
    Trophic_level = as.numeric(.data[[COL_TROPHIC]]),
    Origin = as.factor(.data[[COL_ORIGIN]]),
    Habitat_dependency = as.factor(.data[[COL_HABITAT]]),
    Study_ID = as.factor(.data[[COL_STUDY_ID]])
  ) %>%
  drop_na(Predicted_Lake_MP, Trophic_level, Origin, Habitat_dependency, Study_ID) %>%
  mutate(
    Measured_Abundance_g = replace_na(Measured_Abundance_g, 0),
    Measured_Abundance_ind = replace_na(Measured_Abundance_ind, 0)
  )

if(nrow(df_model_data) < 30) {
  warning(paste("Warning: Only", nrow(df_model_data), "clean data points. Model may fail to converge."))
} else {
  message(paste("Step 1 complete. Proceeding with", nrow(df_model_data), "valid data points."))
}

# --- 3. Step 2 (Workflow): Build the Final Model (Unchanged) ---
message("Step 2: Building a robust Linear Mixed-Effects Model (LMM)...")

df_model_data_scaled <- df_model_data %>%
  mutate(
    Predicted_MP_scaled = scale(log(Predicted_Lake_MP)),
    Trophic_level_scaled = scale(Trophic_level)
  )

lme_final <- nlme::lme(
  log(1 + Measured_Abundance_g) ~ Predicted_MP_scaled + Trophic_level_scaled + Origin + Habitat_dependency,
  data = df_model_data_scaled,
  random = ~ 1 | Study_ID,
  method = "REML"
)

message("Step 2 complete. Robust LMM (lme) has been fitted.")

message("--- Model Coefficients and P-Values ---")
print(summary(lme_final)$tTable)
message("-------------------------------------")


# --- 4. Step 3: Wrangling data for 'Quadrant Count' plots (g and ind) ---

summarize_quadrant_data <- function(binned_data, abundance_col) {
  trophic_breaks <- seq(min(binned_data$Trophic_level), max(binned_data$Trophic_level), length.out = 4)
  abundance_breaks <- quantile(log10(1 + binned_data[[abundance_col]]), probs = c(0, 0.33, 0.66, 1), na.rm = TRUE)
  
  df_binned <- binned_data %>%
    mutate(
      Abundance_Value = .data[[abundance_col]],
      Abundance_Bin = cut(log10(1 + Abundance_Value), breaks = abundance_breaks, labels = c("Low", "Mid", "High"), include.lowest = TRUE),
      Trophic_Bin = cut(Trophic_level, breaks = trophic_breaks, labels = c("Low", "Mid", "High"), include.lowest = TRUE)
    )
  
  # 1. 按 Habitat dependency 计算计数
  df_summary_habitat <- df_binned %>%
    group_by(Trophic_Bin, Abundance_Bin, Habitat_dependency) %>%
    summarise(n = n(), .groups = 'drop') %>%
    drop_na(Trophic_Bin, Abundance_Bin)
  
  # 2. 计算 Origin 百分比
  df_summary_origin_pct <- df_binned %>%
    group_by(Trophic_Bin, Abundance_Bin, Origin) %>%
    summarise(n = n(), .groups = 'drop') %>%
    drop_na(Trophic_Bin, Abundance_Bin) %>%
    pivot_wider(names_from = Origin, values_from = n, values_fill = 0) %>%
    mutate(
      Total_N_Origin = coalesce(Native, 0) + coalesce(Invasive, 0),
      pct_native = coalesce(Native, 0) / Total_N_Origin
    ) %>%
    # (!!!) 【已修复】: 明确使用 dplyr::select
    dplyr::select(Trophic_Bin, Abundance_Bin, pct_native)
  
  # 3. 创建 Habitat 计数标签
  df_plot_labels <- df_summary_habitat %>%
    mutate(label = paste0(Habitat_dependency, ": ", n)) %>%
    # (!!!) 【已修复】: 明确使用 dplyr::select
    dplyr::select(Trophic_Bin, Abundance_Bin, label) %>%
    group_by(Trophic_Bin, Abundance_Bin) %>%
    summarise(Habitat_Counts = paste(label, collapse = "\n"), .groups = 'drop')
  
  # 4. 计算每个象限的总 N
  df_summary_total_n <- df_binned %>%
    drop_na(Trophic_Bin, Abundance_Bin) %>%
    count(Trophic_Bin, Abundance_Bin) %>%
    rename(Total_N = n)
  
  # 5. 创建一个包含 *所有* 9 个象限的网格
  df_all_quadrants <- expand_grid(
    Trophic_Bin = levels(df_binned$Trophic_Bin),
    Abundance_Bin = levels(df_binned$Abundance_Bin)
  )
  
  # 6. 将 *所有内容* 连接到完整的网格
  df_plot_labels_final <- df_all_quadrants %>%
    left_join(df_plot_labels, by = c("Trophic_Bin", "Abundance_Bin")) %>%
    left_join(df_summary_origin_pct, by = c("Trophic_Bin", "Abundance_Bin")) %>%
    left_join(df_summary_total_n, by = c("Trophic_Bin", "Abundance_Bin")) %>%
    mutate(
      Habitat_Label = case_when(
        !is.na(Total_N) & !is.na(Habitat_Counts) ~ paste0("Total n = ", Total_N, "\n", Habitat_Counts),
        !is.na(Total_N) & is.na(Habitat_Counts) ~ paste0("Total n = ", Total_N, "\n(n/a)"),
        is.na(Total_N) ~ "n = 0"
      )
    )
  
  return(df_plot_labels_final)
}

# (!!!) 分别运行汇总
df_plot_data_g <- summarize_quadrant_data(df_model_data, COL_MEASURED_G)
df_plot_data_ind <- summarize_quadrant_data(df_model_data, COL_MEASURED_IND)

message("Step 3 complete. Data is binned and summarized for BOTH metrics.")

# --- 5. Step 4: 创建和保存 *两个* 象限图 ---

# (!!!) 关键修改：使用 scale_fill_gradientn 实现三色渐变
create_quadrant_plot <- function(plot_data, title_prefix, title_suffix) {
  
  # 自定义颜色
  COLOR_INVASIVE <- "#e31a1c" # 亮红色 (0% Native)
  COLOR_MID <- "#fd8d3c"   # 新增中间色 (50% Native)
  COLOR_NATIVE <- "#0a3b5f" # 深蓝色 (100% Native)
  
  ggplot(plot_data, aes(x = Trophic_Bin, y = Abundance_Bin)) +
    
    # 1. 绘制 "九宫格" (geom_tile)
    geom_tile(
      aes(fill = pct_native),
      color = "black", 
      linewidth = 1,
      alpha = 0.2 # (!!!) 已添加透明度
    ) +
    
    # 2. 绘制 "栖息地数字n" (geom_text)
    geom_text(
      aes(label = Habitat_Label),
      size = 3.5,
      lineheight = 0.9,
      vjust = 0.5,
      hjust = 0.5,
      family = "sans"
    ) +
    
    # --- 3. 美化和标签 ---
    scale_x_discrete(position = "bottom", labels = c("Low", "Mid", "High")) +
    scale_y_discrete(limits = rev, labels = c("Low", "Mid", "High")) +  
    
    # (!!!) 关键修改：使用 scale_fill_gradientn 实现三色渐变
    scale_fill_gradientn(
      name = "% Native Species\n(in Quadrant)",
      colors = c(COLOR_INVASIVE, COLOR_MID, COLOR_NATIVE), # 三个颜色
      values = c(0, 0.5, 1),                # 对应的位置 (0%, 50%, 100%)
      limits = c(0, 1), 
      labels = scales::percent, 
      na.value = "grey95" 
    ) +
    
    guides(color = "none", shape = "none") +
    
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      legend.position = "right",
      legend.title = element_text(face="bold", size=10),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 11, face = "bold"),
      aspect.ratio = 1,
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    
    labs(
      title = paste0(title_prefix, title_suffix),
      x = "Trophic Level Bins (Low, Mid, High)",
      y = "Log(1+Abundance) Bins (Low, Mid, High)"
    )
}

# (!!!) 分别创建两个图 (A 和 B)
plot_quadrant_g <- create_quadrant_plot(df_plot_data_g, "A. Measured Abundance (items/g)", " (Per Gram Weight)")
plot_quadrant_ind <- create_quadrant_plot(df_plot_data_ind, "B. Measured Abundance (items/ind)", " (Per Individual)")

# --- 6. 关键修改：使用 patchwork 组合图表并保存 ---
message("Step 5: Combining plots using patchwork and saving to a single PDF...")

# 将两张图并排放置
combined_plot <- plot_quadrant_g | plot_quadrant_ind

# 设置整个页面的标题
final_plot <- combined_plot + 
  plot_annotation(
    title = 'Biota Microplastic Distribution by Trophic Level and Species Origin',
    theme = theme(plot.title = element_text(face = 'bold', size = 18, hjust = 0.5))
  )

# --- 保存为可在 Illustrator 编辑的 PDF ---
output_filename_pdf_combined <- file.path(OUTPUT_DIR, "validation_LMM_QuadrantCount_Combined_TriColor.pdf") # (!!!) 更改文件名以区分
ggsave(
  output_filename_pdf_combined,
  plot = final_plot,
  device = cairo_pdf,
  width = 20, # 增加宽度以容纳两张图
  height = 10,
  units = "in",
  family = "Arial"
)

message(paste("Final combined PDF (with tri-color gradient) saved to:", output_filename_pdf_combined))

# (Optional) Print the combined plot to the RStudio window
print(final_plot)