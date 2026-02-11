
# 加载必要的包
library(readxl)
library(dplyr)
library(rfishbase)

# ==============================
# 1️⃣ 读取 Excel 文件中的物种名
# ==============================
#fish_path <- "G:/Emerging contaminants and nutrients/2-Sea around us (EEZ capture)/补充营养级/Fishbase.xlsx"
#fish_path <- "G:/Emerging contaminants and nutrients/2-Sea around us (EEZ capture)/补充营养级/过程文件/第二次/Fishbase1.xlsx"

# ========================= 体内浓度估计
fish_path <- "G:/Emerging contaminants and nutrients/3-物种体内浓度数据拟合/2-缺失学名.xlsx"

species <- read_excel(fish_path) %>%
  pull(scientific_name) %>%
  unique() # 确保不重复

# 基本生物信息-水环境、深度范围、体长、最大体重（可能有）
info <- species(species)
info_selected <- info %>%
  select(Species, Saltwater,
         DepthRangeShallow, DepthRangeDeep,
         DepthRangeComShallow, DepthRangeComDeep,
         Length, CommonLength, Weight)

# 生活史参数（营养级信息）
ecology_info <- ecology(species)
ecology_selected <- ecology_info %>%
  select(Species, DietTroph, DietSeTroph,FoodTroph,FoodSeTroph)

#贝叶斯拟合系数a,b-体重体长关系
Baye_info <- length_weight(species)
Baye_selected <- Baye_info %>%
  select(Species, a, b)

Baye_mean <- Baye_selected %>%
  group_by(Species) %>%                  # 按物种分组
  summarise(
    a = mean(a, na.rm = TRUE),           # 取a的均值----存在异常值的处理
    b = mean(b, na.rm = TRUE)            # 取b的均值
  ) %>%
  ungroup()                              # 解除分组

# 假设 info_selected 和 ecology_selected 已经准备好
final_data <- info_selected %>%
  left_join(ecology_selected, by = "Species") %>%
  left_join(Baye_mean, by = "Species")
#--------------------------------------------------------------------------------

library(openxlsx)
library(writexl)

#write_xlsx(final_data, "G:/Emerging contaminants and nutrients/2-Sea around us (EEZ capture)/补充营养级/过程文件/第二次/Fishbase data1.xlsx")
write_xlsx(final_data, "G:/Emerging contaminants and nutrients/3-物种体内浓度数据拟合/0-Fishbase数据/2-SeaLifebase.xlsx")
#--------------------------------------------------------------------------------











