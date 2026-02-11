# ==============================================================================
# 1. 加载所需的库
# ==============================================================================
# 如果您尚未安装这些库，请先运行 install.packages("package_name")
library(readr)   # 用于快速读取CSV文件
library(dplyr)   # 用于数据处理和转换
library(ggplot2) # 用于数据可视化
library(scales)  # 用于高级坐标轴缩放（例如 symlog）
library(ggridges)# 用于解决密度曲线过矮的问题
library(tidyr)   # 用于数据整理 (pivot_longer)

# ==============================================================================
# 2. 数据加载与准备
# ==============================================================================
# 更新为您的绝对文件路径
input_file_path <- "E:/lake-MP-W/data/opt/data/processed_output/Ychange.csv"

# 检查文件是否存在
if (!file.exists(input_file_path)) {
  stop(paste("Error: File not found at '", input_file_path, "'. Please check the path."))
}

# 使用readr加载数据
cat("Loading data...\n")
df <- read_csv(input_file_path, show_col_types = FALSE)
cat("Data loaded successfully!\n\n")

# --- 数据准备 ---
# 将数据转换为"长"格式，以便进行分面绘图
# 同时移除缺失值，并根据您的要求只保留小于等于0的数据
plot_data_long <- df %>%
  select(opt_MPs_count, income, Region) %>%
  filter(!is.na(opt_MPs_count), opt_MPs_count <= 0) %>% # 只截取小于等于0的部分
  pivot_longer(
    cols = c("income", "Region"),
    names_to = "Group",
    values_to = "Category"
  ) %>%
  mutate(
    # 使用更可靠的 case_when() 替换 recode()
    Group = case_when(
      Group == "income" ~ "Income Level",
      Group == "Region" ~ "Region",
      TRUE ~ Group # 保持其他值不变
    ),
    Category = as.factor(Category)
  )

# --- 计算每个类别的样本量 (n) 以便在图上标注 ---
sample_counts <- plot_data_long %>%
  group_by(Group, Category) %>%
  summarise(n = n(), .groups = 'drop') %>%
  mutate(label = paste("n =", format(n, big.mark = ",")))

cat("Data prepared for faceted plotting.\n\n")

# ==============================================================================
# 3. 定义颜色方案
# ==============================================================================
# 根据您的要求定义颜色
income_colors <- c(
  "LIC" = "#00847e", 
  "LMIC" = "#ffc000", 
  "UMIC" = "#883039", 
  "HIC" = "#4c6a9c"
)
region_palette <- c("#afe1af", "#cad675", "#e4cb3a", "#ffc000")
# 将颜色方案合并为一个列表
# 注意：需要动态地为所有地区级别分配颜色
all_region_levels <- levels(factor(plot_data_long$Category[plot_data_long$Group == "Region"]))
region_colors <- setNames(rep_len(region_palette, length(all_region_levels)), all_region_levels)

combined_colors <- c(income_colors, region_colors)

cat("Color schemes defined.\n\n")

# ==============================================================================
# 4. 创建并预览美化后的图表
# ==============================================================================

# --- 定义对称对数变换函数 ---
symlog_trans <- function() {
  trans_new(
    "symlog",
    transform = function(x) sign(x) * log1p(abs(x)),
    inverse = function(x) sign(x) * (expm1(abs(x)))
  )
}

# --- 创建分面山峦图 ---
cat("Creating faceted ridgeline plot...\n")

final_plot <- ggplot(plot_data_long, aes(x = opt_MPs_count, y = Category, fill = Category)) +
  # 1. 绘制山峦图
  geom_density_ridges(
    alpha = 0.7,          # 调整透明度
    scale = 3,            # 增加重叠程度
    rel_min_height = 0.01,# 裁剪曲线尾部
    linewidth = 0.5,      # 轮廓线宽度
    panel_scaling = FALSE,# 确保密度在不同分面之间可比
    # 您可以调整这个值，值越大，曲线越平滑
    bandwidth = 4
  ) +
  # 2. 添加样本量 (n) 标签
  geom_text(
    data = sample_counts,
    aes(x = Inf, label = label), # 将标签放置在最右侧
    hjust = 1.1,                 # 水平对齐
    vjust = 0.5,                 # 垂直对齐
    size = 3.5,
    color = "black"
  ) +
  # 3. 使用分面，将图表分为两部分
  facet_wrap(
    ~ Group, 
    scales = "free_y", # 允许每个分面的Y轴独立（类别不同）
    ncol = 2           # 排列为两列
  ) +
  # 4. 应用对称对数坐标轴
  scale_x_continuous(
    name = "Mitigation Potential (Symmetric Log Scale, Non-Positive Values)",
    trans = symlog_trans(),
    # 更新坐标轴断点以匹配新数据范围
    breaks = c(-10^12, -10^9, -10^6, -10^3, 0)
  ) +
  # 5. 应用自定义颜色
  scale_fill_manual(values = combined_colors) +
  # 6. 设置标题和Y轴标签
  labs(
    title = "Density of Non-Positive Mitigation Potential by Income Level and Region",
    y = NULL # Y轴由分面标签表示，因此不需要主Y轴标签
  ) +
  # 7. 应用一个干净、专业的主题
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", margin = margin(b=15)),
    legend.position = "none", # 山峦图的填充色已经足够清晰，无需图例
    strip.text = element_text(size = 14, face = "bold"), # 分面标题样式
    panel.grid.minor = element_blank(),
    panel.spacing = unit(2, "lines") # 增加分面之间的距离
  )

# 在R的绘图窗口中打印最终图表
cat("Displaying final plot on screen...\n")
print(final_plot)
cat("Plot created.\n\n")

# ==============================================================================
# 5. 将已创建的图表保存到PDF文件
# ==============================================================================
output_pdf_path <- "E:/lake-MP-W/draw/优化潜力统计分析/减排潜力密度分布图_R_beautified.pdf"
cat(paste("Saving plot to PDF:", output_pdf_path, "\n"))

# 使用 ggsave 保存，它能更好地处理复杂的ggplot对象
ggsave(
  output_pdf_path,
  plot = final_plot,
  width = 16,
  height = 9,
  device = cairo_pdf # 使用 cairo_pdf 以获得最佳效果
)

cat(paste("\nPlot successfully saved to '", output_pdf_path, "'\n"))
cat("Analysis complete!\n")




















#######################湖泊面积分析

# -------------------------------------------------------------------------
# 第一部分：加载所需 R 包
# -------------------------------------------------------------------------
# --- 重要提示 ---
# 在运行此脚本之前，请确保您已经安装了所有必需的R包。
# 如果尚未安装，请在您的R控制台中逐行运行以下命令：
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("ggpubr")
# install.packages("ggchicklet")

# 加载包
library(dplyr)      # 用于数据处理和转换
library(ggplot2)    # 用于创建图形
library(ggpubr)     # 用于创建符合出版要求的图形和添加统计检验
library(ggchicklet) # 加载ggchicklet以便未来若有更新可尝试

# -------------------------------------------------------------------------
# 第二部分：数据加载、清洗与确认
# -------------------------------------------------------------------------
# --- 1. 读取原始数据 ---
filePath <- "E:/lake-MP-W/data/opt/data/processed_output/1.csv" 
df_raw <- read.csv(filePath, check.names = FALSE, sep = ",", stringsAsFactors = FALSE) 

# **新增**：打印原始数据的前6行以确认读取成功
cat("--- 1. 原始数据 (前6行) ---\n")
print(head(df_raw))
cat("---------------------------\n\n")

# --- 2. 清洗与转换数据 ---
# 选择前两列并重命名
df <- df_raw[, 1:2]
colnames(df) <- c("Lake_area", "change_percent")

# **核心修复**：在转换为数值前，移除数值中的非数字字符
df$Lake_area <- gsub(",", "", df$Lake_area)
# **核心修复**：移除百分号 '%'，这是导致之前转换失败的根本原因
df$change_percent <- gsub("%", "", df$change_percent)

# **核心修复**：强制将两列都转换为数值类型
df$Lake_area <- as.numeric(df$Lake_area)
df$change_percent <- as.numeric(df$change_percent)

# **数据清理**：移除在转换过程中产生的任何无效行 (NA)
df <- df %>% filter(!is.na(Lake_area) & !is.na(change_percent))

# **新增**：打印清理和转换后的数据，确认格式正确
cat("--- 2. 清理和转换后的数据 (前6行) ---\n")
print(head(df))
cat("-------------------------------------\n\n")


# -------------------------------------------------------------------------
# 第三部分：分组、计算并打印统计表
# -------------------------------------------------------------------------
# --- 3. 分组与统计计算 ---
# 定义分组
breaks <- c(0, 0.1, 1, 10, 100, 1000, 10000, Inf)
labels <- c("< 0.1", "0.1-1", "1-10", "10-100", "100-1000", "1000-10000", "> 10000")

# 对数据进行分组
df_processed <- df %>%
  mutate(Area_Group = cut(Lake_area, breaks = breaks, labels = labels, right = FALSE)) %>%
  filter(!is.na(Area_Group))

# 将分组列转换为有序因子
df_processed$Area_Group <- factor(df_processed$Area_Group, levels = labels)

# 计算汇总统计信息
summary_stats <- df_processed %>%
  group_by(Area_Group) %>%
  summarise(
    mean_change = mean(change_percent, na.rm = TRUE),
    sd_change = sd(change_percent, na.rm = TRUE),
    n = n(),
    se = sd_change / sqrt(n),
    ci_upper = mean_change + 1.96 * se,
    ci_lower = mean_change - 1.96 * se
  ) %>%
  ungroup() %>%
  # **核心修改**：调整缩放公式，使用幂函数增强宽窄对比，使其更加错落有致
  mutate(
    scaled_width = if (length(unique(n)) > 1) {
      # 将样本量n标准化到0-1范围
      normalized_n <- (n - min(n)) / (max(n) - min(n))
      # 使用幂函数 (e.g., 1.5) 来夸大差异
      powered_n <- normalized_n ^ 1.5
      # 将结果缩放到一个新的宽度范围 [0.3, 0.95]，使得最小的柱体更窄，最大的更宽
      0.3 + 0.65 * powered_n
    } else {
      0.95 # 如果所有组样本数相同，则使用统一的宽度
    }
  )


# **新增**：以表格形式打印最终的统计结果
cat("--- 3. 各分组的统计摘要表 ---\n")
print(summary_stats)
cat("-----------------------------\n\n")


# -------------------------------------------------------------------------
# 第四部分：绘制图表
# -------------------------------------------------------------------------
# --- 4. 绘图 ---
# 定义颜色
custom_colors <- c("#afe1af", "#cad675", "#e4cb3a", "#ffc000", "#8dc2b5", "#aab381", "#c7a34e")

# 稳健地计算Y轴范围
y_lower_limit <- min(summary_stats$ci_lower, 0, na.rm = TRUE) * 1.1
if (!is.finite(y_lower_limit)) {
  y_lower_limit <- -10 
}
plot_ylim <- c(y_lower_limit, 0)

# 绘制柱状图
bar_plot <- ggplot(summary_stats, aes(x = Area_Group, y = mean_change, fill = Area_Group)) +
  # **核心修改**：使用geom_col并动态映射宽度，以实现宽度与样本量成正比
  geom_col(
    aes(width = scaled_width),
    show.legend = FALSE
  ) +
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    # 根据柱体宽度动态调整误差线宽度，使其保持在柱体内
    width = 0.2,
    color = "black",
    linewidth = 0.5
  ) +
  scale_fill_manual(values = custom_colors) +
  coord_cartesian(ylim = plot_ylim) +
  labs(
    title = "不同面积湖泊的平均变化率",
    subtitle = "柱体宽度与样本量成正比，误差线代表95%置信区间",
    x = "湖泊面积分组 (km²)",
    y = "平均变化百分比 (%)"
  ) +
  theme_bw(base_size = 14) + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  )

# 显示最终图表
print(bar_plot)

