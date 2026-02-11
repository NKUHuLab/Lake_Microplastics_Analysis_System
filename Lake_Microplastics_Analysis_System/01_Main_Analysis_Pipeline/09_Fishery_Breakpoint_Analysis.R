# 必要的R包
library(tidyverse)
library(patchwork)
library(ggtext)
library(scales)
library(ggrepel)
library(segmented)

# --- 1. 加载并清理数据 ---
file_path <- "E:/lake-MP-W/dataset/FAO/analysis_data_sqa.csv"
final_data_raw <- read.csv(file_path)

final_data <- final_data_raw %>%
  mutate(
    Total_Fishery = Total_Inland_Aquaculture + Total_Inland_Capture,
    Aqua_Intensity = Total_Inland_Aquaculture / Lake_area,
    Capt_Intensity = Total_Inland_Capture / Lake_area,
    log_Aqua_Intensity = log10(Aqua_Intensity + 1),
    log_Capt_Intensity = log10(Capt_Intensity + 1),
    # 创建气泡大小分类
    aqua_size_category = case_when(
      log_Total_Inland_Aquaculture <= 5  ~ "0-5",
      log_Total_Inland_Aquaculture <= 10 ~ "5-10",
      log_Total_Inland_Aquaculture <= 15 ~ "10-15",
      TRUE ~ ">15"
    ),
    capt_size_category = case_when(
      log_Total_Inland_Capture <= 5  ~ "0-5",
      log_Total_Inland_Capture <= 10 ~ "5-10",
      log_Total_Inland_Capture <= 15 ~ "10-15",
      TRUE ~ ">15"
    )
  ) %>%
  mutate(
    aqua_size_category = factor(aqua_size_category, levels = c("0-5", "5-10", "10-15", ">15")),
    capt_size_category = factor(capt_size_category, levels = c("0-5", "5-10", "10-15", ">15"))
  ) %>%
  filter(
    is.finite(log_Aqua_Intensity) &
      is.finite(log_Capt_Intensity) &
      is.finite(log_Mean_MP_Prediction) &
      is.finite(log_Total_Inland_Aquaculture) &
      is.finite(log_Total_Inland_Capture) &
      is.finite(Total_Fishery)
  )

# --- 为每个图识别需要标注的国家 ---
top_y_countries <- final_data %>% arrange(desc(log_Mean_MP_Prediction)) %>% head(5) %>% pull(country)
top_aqua_x_countries <- final_data %>% arrange(desc(log_Aqua_Intensity)) %>% head(5) %>% pull(country)
countries_to_label_aqua <- unique(c(top_aqua_x_countries, top_y_countries))
label_data_aqua <- final_data %>% filter(country %in% countries_to_label_aqua)
top_capt_x_countries <- final_data %>% arrange(desc(log_Capt_Intensity)) %>% head(5) %>% pull(country)
countries_to_label_capt <- unique(c(top_capt_x_countries, top_y_countries))
label_data_capt <- final_data %>% filter(country %in% countries_to_label_capt)

# --- 函数定义 (不变) ---
get_model_annotations <- function(data, x_var, y_var) {
  lm_model <- lm(as.formula(paste(y_var, "~", x_var)), data = data)
  lm_summary <- summary(lm_model)
  lm_coefs <- coef(lm_model)
  p_value <- pf(lm_summary$fstatistic[1], lm_summary$fstatistic[2], lm_summary$fstatistic[3], lower.tail = FALSE)
  lm_label <- paste0(sprintf("<b>Linear Fit:</b><br><i>y</i> = %.2f<i>x</i> + %.2f", lm_coefs[2], lm_coefs[1]),"<br>",sprintf("<i>R</i><sup>2</sup> = %.2f; <i>p</i> %s", lm_summary$r.squared, ifelse(p_value < 0.001, "< 0.001", paste("=", round(p_value, 3)))))
  loess_model <- loess(as.formula(paste(y_var, "~", x_var)), data = data, span = 0.75)
  y_actual <- data[[y_var]]
  pseudo_r2 <- 1 - (sum(resid(loess_model)^2) / sum((y_actual - mean(y_actual))^2))
  loess_label <- sprintf("<b>LOESS Fit (span=0.75):</b><br><i>Pseudo R</i><sup>2</sup> = %.2f", pseudo_r2)
  full_label <- paste(lm_label, loess_label, sep = "<br><br>")
  return(data.frame(x_pos = -Inf, y_pos = Inf, label = full_label))
}
find_all_breakpoints <- function(x, y, num_breakpoints = 2) {
  df <- data.frame(x = x, y = y)
  lin.mod <- lm(y ~ x, data = df)
  seg.mod <- tryCatch(
    segmented(lin.mod, seg.Z = ~x, npsi = num_breakpoints),
    error = function(e) NULL
  )
  if (is.null(seg.mod) || length(seg.mod$psi) == 0) { return(NULL) }
  breakpoints_x <- seg.mod$psi[, "Est."]
  breakpoints_y <- predict(seg.mod, newdata = data.frame(x = breakpoints_x))
  results <- data.frame(x = breakpoints_x, y = breakpoints_y, label = paste0("BP", 1:length(breakpoints_x)))
  return(results)
}

# --- 数据和设置准备 ---
aqua_annotations <- get_model_annotations(final_data, "log_Aqua_Intensity", "log_Mean_MP_Prediction")
capt_annotations <- get_model_annotations(final_data, "log_Capt_Intensity", "log_Mean_MP_Prediction")
breakpoints_aqua <- find_all_breakpoints(final_data$log_Aqua_Intensity, final_data$log_Mean_MP_Prediction, num_breakpoints = 1)
breakpoints_capt <- find_all_breakpoints(final_data$log_Capt_Intensity, final_data$log_Mean_MP_Prediction, num_breakpoints = 2)
base_theme <- theme_minimal() + theme(text=element_text(size=14), plot.title=element_text(size=18, face="bold", margin=margin(b=5)), axis.title=element_text(size=14), axis.text=element_text(size=12), legend.title=element_text(size=12, face="bold"), legend.text=element_text(size=11), panel.grid.minor=element_blank(), panel.grid.major=element_line(linetype="dashed", color="grey85"), panel.border=element_rect(color="black", fill=NA, linewidth=1))
income_palette <- c("LIC"="#00847e", "LMIC"="#ffc000", "UMIC"="#883039", "HIC"="#4c6a9c")

# --- 4. 绘图 ---

# 图A: 水产养殖
plot_aqua_final <- ggplot(final_data, aes(x = log_Aqua_Intensity, y = log_Mean_MP_Prediction)) +
  # CORRECTED: Restored the size mapping to the size category variable
  geom_point(aes(fill = income, size = aqua_size_category), shape = 21, stroke = 0.5, alpha = 0.7, na.rm = TRUE) +
  geom_smooth(method = "lm", formula=y~x, color = "grey70", linetype = "dashed", linewidth = 0.8, fill = "grey85", na.rm = TRUE) +
  geom_smooth(method = "loess", formula=y~x, se = FALSE, color = "#c44e52", linewidth = 1.2, na.rm = TRUE) +
  geom_richtext(data = aqua_annotations, aes(x = x_pos, y = y_pos, label = label), hjust = 0, vjust = 1, inherit.aes = FALSE, size = 3.5, label.padding = unit(0.25, "lines"), label.r = unit(0.15, "lines"), fill = "#FFFFFFCC") +
  geom_text_repel(data = label_data_aqua, aes(label = country), size = 4, color = "black", box.padding = 0.5, max.overlaps = Inf) +
  {if (!is.null(breakpoints_aqua))
    geom_vline(data = breakpoints_aqua, aes(xintercept = x), linetype = "dotted", color = "#e41a1c", linewidth = 1.2)} +
  {if (!is.null(breakpoints_aqua))
    geom_point(data = breakpoints_aqua, aes(x = x, y = y), color = "black", fill = "#e41a1c", size = 6, shape = 23)} +
  {if (!is.null(breakpoints_aqua))
    geom_text(data = breakpoints_aqua, aes(x = x, y = y, label = label), vjust = -1.2, hjust = -0.2, color = "#e41a1c", fontface="bold")} +
  scale_fill_manual(values = income_palette, name = "Income Group") +
  # CORRECTED: Restored the parameterized size scale
  scale_size_manual(name = bquote("log"[10]*"(Aquaculture Prod.)"), values = c("0-5" = 3, "5-10" = 8, "10-15" = 16, ">15" = 25)) +
  labs(title = "A: Inland Aquaculture", x = bquote("log"[10]*"(Aquaculture Intensity + 1)"), y = bquote("log"[10]*"(MP Abundance)")) +
  coord_fixed(ratio = 1) +
  base_theme

# 图B: 捕捞
plot_capture_final <- ggplot(final_data, aes(x = log_Capt_Intensity, y = log_Mean_MP_Prediction)) +
  # CORRECTED: Restored the size mapping to the size category variable
  geom_point(aes(fill = income, size = capt_size_category), shape = 21, stroke = 0.5, alpha = 0.7, na.rm = TRUE) +
  geom_smooth(method = "lm", formula=y~x, color = "grey70", linetype = "dashed", linewidth = 0.8, fill = "grey85", na.rm = TRUE) +
  geom_smooth(method = "loess", formula=y~x, se = FALSE, color = "#4c6a9c", linewidth = 1.2, na.rm = TRUE) +
  geom_richtext(data = capt_annotations, aes(x = x_pos, y = y_pos, label = label), hjust = 0, vjust = 1, inherit.aes = FALSE, size = 3.5, label.padding = unit(0.25, "lines"), label.r = unit(0.15, "lines"), fill = "#FFFFFFCC") +
  geom_text_repel(data = label_data_capt, aes(label = country), size = 4, color = "black", box.padding = 0.5, max.overlaps = Inf) +
  {if (!is.null(breakpoints_capt))
    geom_vline(data = breakpoints_capt, aes(xintercept = x), linetype = "dotted", color = "#e41a1c", linewidth = 1.2)} +
  {if (!is.null(breakpoints_capt))
    geom_point(data = breakpoints_capt, aes(x = x, y = y), color = "black", fill = "#e41a1c", size = 6, shape = 23)} +
  {if (!is.null(breakpoints_capt))
    geom_text(data = breakpoints_capt, aes(x = x, y = y, label = label), vjust = -1.2, hjust = -0.2, color = "#e41a1c", fontface="bold")} +
  scale_fill_manual(values = income_palette, name = "Income Group") +
  # CORRECTED: Restored the parameterized size scale
  scale_size_manual(name = bquote("log"[10]*"(Capture Prod.)"), values = c("0-5" = 3, "5-10" = 8, "10-15" = 16, ">15" = 25)) +
  labs(title = "B: Inland Capture", x = bquote("log"[10]*"(Capture Intensity + 1)"), y = "") +
  coord_fixed(ratio = 1) +
  base_theme +
  theme(axis.text.y = element_blank())

# --- 5. 拼接 ---
final_plot_ultimate <- (plot_aqua_final + plot_capture_final) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(final_plot_ultimate)

# --- 6. 保存PDF文件 ---
ggsave(
  "E:/lake-MP-W/draw/FAO-MP/FAO_MP_Analysis_Final111.pdf",
  plot = final_plot_ultimate,
  device = "pdf",
  width = 18,
  height = 11, # Increased height slightly for the legends at the bottom
  useDingbats = FALSE
)














# --- 7. 数值分析与结果打印 ---
# 确保在运行此代码前，您已经成功生成了 final_data, breakpoints_aqua, 和 breakpoints_capt 对象

cat("--- 1. 渔业强度超过拐点后的斜率分析 ---\n\n")

# --- 水产养殖斜率 ---
if (!is.null(breakpoints_aqua) && nrow(breakpoints_aqua) > 0) {
  # 获取拐点阈值
  bp_aqua_1 <- breakpoints_aqua$x[1]
  
  # 筛选超过拐点的数据
  aqua_post_bp_data <- final_data %>%
    filter(log_Aqua_Intensity > bp_aqua_1)
  
  # 计算此阶段的线性模型
  lm_aqua_post_bp <- lm(log_Mean_MP_Prediction ~ log_Aqua_Intensity, data = aqua_post_bp_data)
  slope_aqua <- coef(lm_aqua_post_bp)[2]
  
  cat(sprintf(
    "水产养殖: 超过拐点 BP1 (%.2f) 后, MP丰度随强度增加的平均斜率为: %.3f\n",
    bp_aqua_1,
    slope_aqua
  ))
} else {
  cat("水产养殖: 未找到拐点，无法计算后续斜率。\n")
}

cat("\n") # 添加空行

# --- 捕捞渔业斜率 ---
if (!is.null(breakpoints_capt) && nrow(breakpoints_capt) >= 2) {
  # 对拐点进行排序，确保 bp1 < bp2
  bps_capt <- sort(breakpoints_capt$x)
  bp_capt_1 <- bps_capt[1]
  bp_capt_2 <- bps_capt[2]
  
  # 阶段 1: BP1 和 BP2 之间
  capt_bp1_bp2_data <- final_data %>%
    filter(log_Capt_Intensity > bp_capt_1 & log_Capt_Intensity <= bp_capt_2)
  
  lm_capt_1 <- lm(log_Mean_MP_Prediction ~ log_Capt_Intensity, data = capt_bp1_bp2_data)
  slope_capt_1 <- coef(lm_capt_1)[2]
  
  cat(sprintf(
    "捕捞渔业: 在拐点 BP1 (%.2f) 和 BP2 (%.2f) 之间, MP丰度随强度增加的平均斜率为: %.3f\n",
    bp_capt_1, bp_capt_2, slope_capt_1
  ))
  
  # 阶段 2: 超过 BP2
  capt_post_bp2_data <- final_data %>%
    filter(log_Capt_Intensity > bp_capt_2)
  
  lm_capt_2 <- lm(log_Mean_MP_Prediction ~ log_Capt_Intensity, data = capt_post_bp2_data)
  slope_capt_2 <- coef(lm_capt_2)[2]
  
  cat(sprintf(
    "捕捞渔业: 超过拐点 BP2 (%.2f) 后, MP丰度随强度增加的平均斜率为: %.3f\n",
    bp_capt_2, slope_capt_2
  ))
  
} else {
  cat("捕捞渔业: 未找到两个拐点，无法计算后续斜率。\n")
}

cat("\n--- 2. 超过拐点阈值的国家收入占比与MP偏离度分析 ---\n\n")

# 首先，我们需要重新拟合LOESS模型以用于预测
loess_aqua_model <- loess(log_Mean_MP_Prediction ~ log_Aqua_Intensity, data = final_data, span = 0.75)
loess_capt_model <- loess(log_Mean_MP_Prediction ~ log_Capt_Intensity, data = final_data, span = 0.75)

# 定义一个辅助函数来执行分析和打印
analyze_breakpoint_subset <- function(data, x_var, bp_val, loess_model, title) {
  
  # 筛选超过阈值的数据
  subset_data <- data %>% filter(.data[[x_var]] > bp_val)
  
  if(nrow(subset_data) == 0) {
    cat(paste0(title, "\n无国家超过此阈值。\n\n"))
    return()
  }
  
  # 计算LOESS预测值和残差 (实际值 - 预测值)
  subset_data$predicted_mp <- predict(loess_model, newdata = subset_data)
  subset_data$mp_deviation <- subset_data$log_Mean_MP_Prediction - subset_data$predicted_mp
  
  # 计算各国收入占比和平均偏离度
  summary_stats <- subset_data %>%
    group_by(income) %>%
    summarise(
      count = n(),
      avg_deviation = mean(mp_deviation)
    ) %>%
    mutate(
      proportion = count / sum(count)
    ) %>%
    arrange(desc(proportion))
  
  cat(paste0(title, sprintf(" (阈值 > %.2f, 共 %d 个国家)\n", bp_val, nrow(subset_data))))
  cat("------------------------------------------------------------------\n")
  cat("收入水平\t占比 (%)\tMP丰度平均偏离LOESS预测值\n")
  cat("------------------------------------------------------------------\n")
  
  # 打印结果
  for(i in 1:nrow(summary_stats)) {
    cat(sprintf("%-10s\t%-10.1f\t%+.3f\n",
                summary_stats$income[i],
                summary_stats$proportion[i] * 100,
                summary_stats$avg_deviation[i]
    ))
  }
  cat("------------------------------------------------------------------\n\n")
}

# --- 应用函数进行分析 ---
if (!is.null(breakpoints_aqua) && nrow(breakpoints_aqua) > 0) {
  analyze_breakpoint_subset(final_data, "log_Aqua_Intensity", breakpoints_aqua$x[1], loess_aqua_model, "水产养殖: 超过 BP1 的国家")
}

if (!is.null(breakpoints_capt) && nrow(breakpoints_capt) >= 2) {
  bps_capt_sorted <- sort(breakpoints_capt$x)
  analyze_breakpoint_subset(final_data, "log_Capt_Intensity", bps_capt_sorted[1], loess_capt_model, "捕捞渔业: 超过 BP1 的国家")
  analyze_breakpoint_subset(final_data, "log_Capt_Intensity", bps_capt_sorted[2], loess_capt_model, "捕捞渔业: 超过 BP2 的国家")
}


cat("\n--- 3. 重点标注国家的数据详情 ---\n\n")

# 提取需要展示的列
columns_to_show <- c("country", "income", "log_Aqua_Intensity", "log_Capt_Intensity", "log_Mean_MP_Prediction", "log_Total_Inland_Aquaculture", "log_Total_Inland_Capture")

# 水产养殖图的标注国家
cat("水产养殖图 (图A) 的标注国家数据:\n")
label_data_aqua_details <- final_data %>%
  filter(country %in% countries_to_label_aqua) %>%
  # --- 这里是修改点 ---
  dplyr::select(all_of(columns_to_show))
print(label_data_aqua_details)

cat("\n") # 添加空行

# 捕捞渔业图的标注国家
cat("捕捞渔业图 (图B) 的标注国家数据:\n")
label_data_capt_details <- final_data %>%
  filter(country %in% countries_to_label_capt) %>%
  # --- 这里是修改点 ---
  dplyr::select(all_of(columns_to_show))
print(label_data_capt_details)

cat("\n--- 分析结束 ---\n")












# --- 9. 精确重算捕捞渔业LOESS曲线在特定区间的平均斜率 ---

cat("\n--- 精确重算：计算捕捞渔业LOESS曲线在特定区间的平均斜率 ---\n\n")

if (!is.null(loess_capt_model) && !is.null(breakpoints_capt) && nrow(breakpoints_capt) >= 2) {
  
  # --- 阶段 1: BP1 和 BP2 之间 ---
  
  # 对拐点进行排序，确保 bp1 < bp2
  bps_capt_sorted <- sort(breakpoints_capt$x)
  bp1 <- bps_capt_sorted[1]
  bp2 <- bps_capt_sorted[2]
  
  # 在区间 [BP1, BP2] 内创建密集点
  x_points1 <- seq(from = bp1, to = bp2, by = 0.01)
  
  # 预测y值并计算逐点斜率
  predicted_y1 <- predict(loess_capt_model, newdata = data.frame(log_Capt_Intensity = x_points1))
  pointwise_slopes1 <- diff(predicted_y1) / diff(x_points1)
  average_loess_slope1 <- mean(pointwise_slopes1, na.rm = TRUE)
  
  cat(sprintf(
    "捕捞渔业LOESS曲线在区间 [%.2f, %.2f] (BP1到BP2) 内的平均斜率近似为: %.3f\n",
    bp1,
    bp2,
    average_loess_slope1
  ))
  
  # --- 阶段 2: 超过 BP2 ---
  
  # 定义区间的起始点和结束点 (结束点为该区间的最大x值)
  start_x2 <- bp2
  end_x2 <- final_data %>% 
    filter(log_Capt_Intensity > start_x2) %>%
    summarise(max_x = max(log_Capt_Intensity, na.rm = TRUE)) %>%
    pull(max_x)
  
  # 在区间 [BP2, Max] 内创建密集点
  x_points2 <- seq(from = start_x2, to = end_x2, by = 0.01)
  
  # 预测y值并计算逐点斜率
  predicted_y2 <- predict(loess_capt_model, newdata = data.frame(log_Capt_Intensity = x_points2))
  pointwise_slopes2 <- diff(predicted_y2) / diff(x_points2)
  average_loess_slope2 <- mean(pointwise_slopes2, na.rm = TRUE)
  
  cat(sprintf(
    "捕捞渔业LOESS曲线在强度超过 %.2f (BP2之后) 的区间内的平均斜率近似为: %.3f\n",
    start_x2,
    average_loess_slope2
  ))
  
} else {
  cat("错误：无法进行计算，请确保 loess_capt_model 和 breakpoints_capt 对象已成功创建。\n")
}

cat("\n--- 分析结束 ---\n")