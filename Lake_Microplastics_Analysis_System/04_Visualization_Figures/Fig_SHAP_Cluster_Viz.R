# 必要的包
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(cluster)  # k-means聚类
library(ggplot2)  # 可视化
library(explainer)  # explainer包

# 读取SHAP值数据
shap_data <- read.csv("E:/lakemicroplastic/draw/shap聚类/shap_values_S.csv")

# 检查数据头部
head(shap_data)

# 假设 SHAP 数据是以 long 格式保存
# 这里将使用 shap_Mean_long 进行聚类分析
shap_Mean_long <- shap_data  # 你需要确认这是否是你实际的 SHAP 数据

# 进行K-means聚类（假设选择3个聚类）
set.seed(123)  # 固定随机种子
num_of_clusters <- 3
kmeans_result <- kmeans(shap_Mean_long[, -1], centers = num_of_clusters, nstart = 25)  # 去掉第一列（假设为ID列）

# 查看聚类结果
table(kmeans_result$cluster)

# 将聚类结果添加到原数据中
shap_Mean_long$Cluster <- factor(kmeans_result$cluster)

# 3. 可视化聚类结果（以PCA降维后的结果为例）
pca_result <- prcomp(shap_Mean_long[, -ncol(shap_Mean_long)], scale. = TRUE)  # 不包含聚类列

# 将PCA结果与聚类结果合并
pca_data <- data.frame(pca_result$x, Cluster = shap_Mean_long$Cluster)

# 绘制PCA散点图
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = "SHAP Clustering Visualization (PCA)",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal() +
  scale_color_manual(values = c("red", "green", "blue"))

# 4. 保存聚类结果到CSV文件
write.csv(shap_Mean_long, "E:/lakemicroplastic/draw/shap聚类/shap_cluster_results_S.csv", row.names = FALSE)

# 可选：生成每个聚类的统计描述（均值和标准差）
summary_clusters <- shap_Mean_long %>%
  group_by(Cluster) %>%
  summarise(across(everything(), list(mean = mean, sd = sd), na.rm = TRUE))

# 查看每个聚类的统计描述
print(summary_clusters)

# 保存统计描述为CSV
write.csv(summary_clusters, "E:/lakemicroplastic/draw/shap聚类/shap_cluster_stats_S.csv", row.names = FALSE)
