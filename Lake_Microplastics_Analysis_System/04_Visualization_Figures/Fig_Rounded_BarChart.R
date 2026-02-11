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
# install.packages("ggchicklet") # 新增：用于绘制圆角柱状图

# 加载包
library(dplyr)      # 用于数据处理和转换
library(ggplot2)    # 用于创建图形
library(ggpubr)     # 用于创建符合出版要求的图形和添加统计检验
library(ggchicklet) # 新增

# -------------------------------------------------------------------------
# 第二部分：加载和准备数据
# -------------------------------------------------------------------------
# --- 实际使用时请使用以下代码加载您的数据 ---
# 注意：请确保文件路径正确，并且R可以访问。
# Windows系统下建议使用正斜杠 "/" 或双反斜杠 "\\" 作为路径分隔符。
filePath <- "E:/lake-MP-W/data/opt/data/processed_output/1.csv" 
# **更新**：明确指定逗号分隔符
df_raw <- read.csv(filePath, check.names = FALSE, sep = ",") 

# **错误修复**：仅选择前两列以避免因空列导致列名错误
df <- df_raw[, 1:2]

# **错误修复**：确保列名与后续代码一致，将 "change%" 重命名为 "change_percent" 以避免语法错误
colnames(df) <- c("Lake_area", "change_percent")

# **核心错误修复**：强制将change_percent列转换为数值类型
# 使用suppressWarnings来避免因文本转NA产生的警告信息
df$change_percent <- as.numeric(suppressWarnings(as.character(df$change_percent)))

# **数据清理**：移除转换后产生的NA值，确保后续计算的准确性
df <- df %>% filter(!is.na(change_percent))
# ----------------------------------------------------


# 2. 根据湖泊面积进行分组
# **更新**：根据您的要求，增加新的分组
breaks <- c(0, 0.1, 1, 10, 100, 1000, 10000, Inf)
labels <- c("< 0.1", "0.1-1", "1-10", "10-100", "100-1000", "1000-10000", "> 10000")

# **逻辑修正**：不再预先过滤数据，而是对所有数据进行分组
df_processed <- df %>%
  mutate(Area_Group = cut(Lake_area, breaks = breaks, labels = labels, right = FALSE)) %>%
  filter(!is.na(Area_Group)) # 过滤掉可能因分组产生的NA值

# 将 Area_Group 转换为因子，并指定顺序，确保绘图时X轴顺序正确
df_processed$Area_Group <- factor(df_processed$Area_Group, levels = labels)


# -------------------------------------------------------------------------
# 第三部分：绘制圆角柱状图
# -------------------------------------------------------------------------
# 1. 计算每个组的汇总统计信息
# **更新**：使用处理后的数据 `df_processed` 进行计算
summary_stats <- df_processed %>%
  group_by(Area_Group) %>%
  summarise(
    mean_change = mean(change_percent, na.rm = TRUE),
    sd_change = sd(change_percent, na.rm = TRUE),
    n = n(),
    se = sd_change / sqrt(n),
    ci_upper = mean_change + 1.96 * se, # 95% CI 上限
    ci_lower = mean_change - 1.96 * se  # 95% CI 下限
  ) %>%
  ungroup() # 取消分组

# **新增**：打印计算出的统计摘要表，以便调试和确认
cat("--- 计算的统计摘要 ---\n")
print(summary_stats)
cat("------------------------\n\n")


# 2. 定义颜色
# **更新**：从您提供的颜色列表中选取，确保颜色数量足够 (7个)
custom_colors <- c("#afe1af", "#cad675", "#e4cb3a", "#ffc000", "#8dc2b5", "#aab381", "#c7a34e")

# **逻辑修正**：稳健地计算Y轴范围，以防数据为空
# 确定Y轴的下限，留出10%的额外空间
y_lower_limit <- min(summary_stats$ci_lower, 0, na.rm = TRUE) * 1.1
# 如果计算出的下限不是一个有限的数字（例如，因为没有数据），则设置一个默认值
if (!is.finite(y_lower_limit)) {
  y_lower_limit <- -10 # 设置一个默认的Y轴下限
}
plot_ylim <- c(y_lower_limit, 0)


# 3. 绘制柱状图
bar_plot <- ggplot(summary_stats, aes(x = Area_Group, y = mean_change, fill = Area_Group)) +
  # **核心错误修复**：移除错误的aes(width=...)，使用固定的宽度参数
  geom_chicklet(
    width = 0.85, # 设置一个固定的、较大的宽度以缩短柱间距
    radius = unit(4, "pt"), # 设置圆角半径
    show.legend = FALSE
  ) +
  # 添加误差线
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    width = 0.25, # 使用固定的宽度
    color = "black",
    linewidth = 0.5
  ) +
  scale_fill_manual(values = custom_colors) +
  # **逻辑修正**：使用预先计算好的、安全的Y轴范围
  coord_cartesian(ylim = plot_ylim) +
  labs(
    title = "不同面积湖泊的平均变化率",
    subtitle = "误差线代表95%置信区间",
    x = "湖泊面积分组 (km²)",
    y = "平均变化百分比 (%)"
  ) +
  theme_bw(base_size = 14) + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank() # 移除垂直网格线，使图表更整洁
  )

# 显示柱状图
print(bar_plot)
