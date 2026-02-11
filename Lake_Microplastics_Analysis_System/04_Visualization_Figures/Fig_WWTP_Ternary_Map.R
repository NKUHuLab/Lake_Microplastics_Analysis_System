# -----------------------------------------------------------------------------
# STEP 1: LOAD NECESSARY PACKAGES
# -----------------------------------------------------------------------------
# Ensure packages are installed
# install.packages(c("sf", "tidyverse", "tricolore", "showtext", "ggtern", "rnaturalearth"))

library(sf)
library(tidyverse)
library(tricolore)
library(showtext)
library(ggtern)
library(rnaturalearth)

# Add and use the Arial font.
font_add("Arial", regular = "arial.ttf", bold = "arialbd.ttf")
showtext_auto()

# -----------------------------------------------------------------------------
# STEP 2: LOAD AND PREPARE DATA
# -----------------------------------------------------------------------------
shapefile_path <- "E:/lake-MP-W/draw/三元污水排放地图/shp/all8_with_waste_discharge_final.shp"
map_data_raw <- st_read(shapefile_path)

if (is.na(st_crs(map_data_raw))) {
  st_crs(map_data_raw) <- 4326
}

# Fetch country borders and REMOVE ANTARCTICA
country_borders <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(continent != "Antarctica")

map_data_processed <- map_data_raw %>%
  mutate(
    Primary_Wa = as.numeric(Primary_Wa),
    Secondary_ = as.numeric(Secondary_),
    Advanced_W = as.numeric(Advanced_W)
  ) %>%
  mutate(
    value_sum = Primary_Wa + Secondary_ + Advanced_W
  ) %>%
  mutate(
    is_invalid = (is.na(value_sum) | value_sum == 0)
  )

data_to_color <- map_data_processed %>% filter(!is_invalid)
data_to_grey <- map_data_processed %>% filter(is_invalid)

# -----------------------------------------------------------------------------
# STEP 3: CREATE THE LEGEND OBJECT
# -----------------------------------------------------------------------------
proportions <- data_to_color %>%
  st_drop_geometry() %>%
  select(Primary_Wa, Secondary_, Advanced_W) %>%
  mutate(row_sum = Primary_Wa + Secondary_ + Advanced_W + .Machine$double.eps) %>%
  transmute(
    p1 = Primary_Wa / row_sum,
    p2 = Secondary_ / row_sum,
    p3 = Advanced_W / row_sum
  )

mean_center_point <- c(
  mean(proportions$p1, na.rm = TRUE),
  mean(proportions$p2, na.rm = TRUE),
  mean(proportions$p3, na.rm = TRUE)
)
mean_center_point <- mean_center_point / sum(mean_center_point)

tric_data_centered <- Tricolore(
  data_to_color,
  p1 = 'Primary_Wa',
  p2 = 'Secondary_',
  p3 = 'Advanced_W',
  breaks = Inf,
  center = mean_center_point,
  chroma = 0.7,
  lightness = 0.85
)

legend_plot_object <- tric_data_centered$key +
  geom_point(
    data = proportions,
    aes(x = p3, y = p2, z = p1),
    inherit.aes = FALSE,
    pch = 19,
    cex = 0.1,
    alpha = 0.1,
    color = "black"
  ) +
  labs(L = 'Primary', T = 'Secondary', R = 'Advanced') +
  theme(
    text = element_text(family = "Arial", size = 30, face = "bold"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

# -----------------------------------------------------------------------------
# STEP 4: CREATE THE MAP-ONLY OBJECT
# -----------------------------------------------------------------------------
map_only_object <- ggplot() +
  geom_sf(
    data = country_borders,
    fill = "grey85",
    color = NA
  ) +
  geom_sf(
    data = data_to_grey,
    aes(geometry = geometry),
    fill = "grey93",
    color = NA
  ) +
  geom_sf(
    data = data_to_color,
    aes(fill = tric_data_centered$rgb, geometry = geometry),
    color = NA
  ) +
  scale_fill_identity() +
  geom_sf(
    data = country_borders,
    fill = NA,
    color = "gray50",
    size = 0.15
  ) +
  coord_sf(crs = "+proj=robin") +
  theme_minimal(base_family = "Arial") +
  # **** FINAL CHANGES ARE HERE ****
  theme(
    axis.title = element_blank(),
    # 1. Remove longitude/latitude labels
    axis.text = element_blank(),
    panel.grid = element_line(color = "grey80", linetype = "dashed", size = 0.8),
    plot.background = element_rect(fill = "white", colour = "white"),
    panel.background = element_rect(fill = "white", colour = "white"),
    # 2. Remove the outer rectangular frame
    panel.border = element_blank()
  )

# -----------------------------------------------------------------------------
# STEP 5: SAVE THE TWO PLOTS TO SEPARATE FILES
# -----------------------------------------------------------------------------
output_directory <- "E:/lake-MP-W/draw/三元污水排放地图"
dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)

map_filename_png <- file.path(output_directory, "Map_Final_Clean.png")
ggsave(
  filename = map_filename_png,
  plot = map_only_object,
  device = "png",
  width = 16,
  height = 9,
  units = "in",
  dpi = 1000
)

legend_filename_pdf <- file.path(output_directory, "Legend_Final_Editable.pdf")
ggsave(
  filename = legend_filename_pdf,
  plot = legend_plot_object,
  device = cairo_pdf,
  width = 8,
  height = 8,
  units = "in"
)

cat(
  "Process complete. Two files were saved to your directory:\n",
  "1. Map File (PNG): ", map_filename_png, "\n",
  "2. Legend File (Editable PDF): ", legend_filename_pdf, "\n"
)
















# -----------------------------------------------------------------------------
# STEP 1: LOAD NECESSARY PACKAGES
# -----------------------------------------------------------------------------
# 确保这些包已经安装
# install.packages(c("sf", "tidyverse", "tricolore", "ggtern", "Cairo", "png"))

library(sf)
library(tidyverse)
library(tricolore)
library(ggtern)
library(Cairo)
library(png)

# -----------------------------------------------------------------------------
# STEP 2: LOAD AND PREPARE DATA (与之前相同)
# -----------------------------------------------------------------------------
# 加载数据以计算图例的颜色和数据点分布
shapefile_path <- "E:/lake-MP-W/draw/三元污水排放地图/shp/all8_with_waste_discharge_final.shp"
map_data_raw <- st_read(shapefile_path)

# 数据预处理
map_data_processed <- map_data_raw %>%
  mutate(
    Primary_Wa = as.numeric(Primary_Wa),
    Secondary_ = as.numeric(Secondary_),
    Advanced_W = as.numeric(Advanced_W)
  ) %>%
  mutate(
    value_sum = Primary_Wa + Secondary_ + Advanced_W
  ) %>%
  mutate(
    is_invalid = (is.na(value_sum) | value_sum == 0)
  )

data_to_color <- map_data_processed %>% filter(!is_invalid)

# -----------------------------------------------------------------------------
# STEP 3: CALCULATE DATA FOR PLOTTING (与之前相同)
# -----------------------------------------------------------------------------
# 计算三元组分比例
proportions <- data_to_color %>%
  st_drop_geometry() %>%
  select(Primary_Wa, Secondary_, Advanced_W) %>%
  mutate(row_sum = Primary_Wa + Secondary_ + Advanced_W + .Machine$double.eps) %>%
  transmute(
    p1 = Primary_Wa / row_sum,
    p2 = Secondary_ / row_sum,
    p3 = Advanced_W / row_sum
  )

# 计算中心点
mean_center_point <- c(
  mean(proportions$p1, na.rm = TRUE),
  mean(proportions$p2, na.rm = TRUE),
  mean(proportions$p3, na.rm = TRUE)
)
mean_center_point <- mean_center_point / sum(mean_center_point)

# -----------------------------------------------------------------------------
# STEP 4: MANUALLY REBUILD THE LEGEND FROM SCRATCH
# -----------------------------------------------------------------------------

# 4.1: 创建一个精细的、符合要求的三角形网格
resolution <- 200
background_grid <- tidyr::crossing(
  p1 = seq(0, 1, length.out = resolution + 1),
  p2 = seq(0, 1, length.out = resolution + 1)
) %>%
  filter(p1 + p2 <= 1 + 1e-9) %>%
  mutate(p3 = 1 - p1 - p2) %>%
  # ============================================================================
# **** 终极修正: 将因浮点精度产生的微小负数强制修正为0 ****
mutate(p3 = if_else(p3 < 0, 0, p3))
# ============================================================================


# 4.2: 为网格上的每个点计算颜色
grid_tricolore <- Tricolore(
  background_grid, 'p1', 'p2', 'p3',
  center = mean_center_point,
  chroma = 0.7, lightness = 0.85
)
background_grid$col <- grid_tricolore$rgb

# 4.3: 将网格点和数据点的三元坐标转换为X-Y坐标
background_grid_xy <- background_grid %>%
  mutate(
    x = 0.5 * p2 + p3,
    y = (sqrt(3)/2) * p2
  )

data_points_xy <- proportions %>%
  mutate(
    x = 0.5 * p2 + p3,
    y = (sqrt(3)/2) * p2
  )

# 4.4: 定义坐标轴标题的位置与角度
axis_labels <- tibble(
  lab = c("Primary", "Secondary", "Advanced"),
  x = c(-0.05, 0.5, 1.05),
  y = c(0.4, sqrt(3)/2 + 0.05, 0.4),
  angle = c(60, 0, -60),
  hjust = c(1, 0.5, 0)
)

# 4.5: 使用标准的ggplot函数构建图例
legend_background_plot <- ggplot() +
  geom_tile(
    data = background_grid_xy,
    aes(x = x, y = y, fill = col)
  ) +
  scale_fill_identity() +
  geom_point(
    data = data_points_xy,
    aes(x = x, y = y),
    pch = 19, size = 0.1, alpha = 0.1, color = "black"
  ) +
  geom_segment(aes(x=0, y=0, xend=1, yend=0), color = 'black') +
  geom_segment(aes(x=0, y=0, xend=0.5, yend=sqrt(3)/2), color = 'black') +
  geom_segment(aes(x=1, y=0, xend=0.5, yend=sqrt(3)/2), color = 'black') +
  theme_void() +
  coord_fixed()

# 4.6: 将背景图保存为一个临时的PNG文件
temp_png_path <- "legend_background_temp.png"
ggsave(
  filename = temp_png_path,
  plot = legend_background_plot,
  device = "png",
  width = 8, height = 8, units = "in",
  dpi = 600
)

# -----------------------------------------------------------------------------
# STEP 5: CREATE FINAL PDF WITH RASTER BACKGROUND AND VECTOR TEXT
# -----------------------------------------------------------------------------

# 5.1: 将刚刚保存的PNG图片读入R
legend_raster_bg <- readPNG(temp_png_path)

# 5.2: 创建最终的图例
final_legend_plot <- ggplot() +
  annotation_raster(
    legend_raster_bg,
    xmin = -0.1, xmax = 1.1,
    ymin = -0.1, ymax = 1.0
  ) +
  geom_text(
    data = axis_labels,
    aes(x = x, y = y, label = lab, angle = angle, hjust = hjust),
    size = 10,
    face = "bold"
  ) +
  coord_fixed(xlim = c(-0.1, 1.1), ylim = c(-0.1, 1.0), expand = FALSE) +
  theme_void()

# -----------------------------------------------------------------------------
# STEP 6: SAVE THE FINAL COMPOSITE LEGEND AS AN EDITABLE PDF
# -----------------------------------------------------------------------------
output_directory <- "E:/lake-MP-W/draw/三元污水排放地图"
dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)

legend_filename_pdf <- file.path(output_directory, "Legend_Optimized_Editable_Final.pdf")

ggsave(
  filename = legend_filename_pdf,
  plot = final_legend_plot,
  device = cairo_pdf,
  width = 8,
  height = 8,
  units = "in"
)

# -----------------------------------------------------------------------------
# STEP 7: CLEAN UP
# -----------------------------------------------------------------------------
# 删除临时的PNG文件
file.remove(temp_png_path)

# --- 最终确认信息 ---
cat(
  "处理完成！优化的、文字可编辑的图例文件已保存至:\n",
  legend_filename_pdf, "\n"
)