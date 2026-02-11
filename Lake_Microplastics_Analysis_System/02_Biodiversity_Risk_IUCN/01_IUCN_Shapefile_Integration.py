import os
import geopandas as gpd
import pandas as pd

# --- 1. 配置工作路径 ---
print("--- [步骤 1/4] 初始化数据整合配置 (已更新) ---")

# 输入数据文件夹路径
# 更新：包含 CR, 移除 LC
base_input_dir = r"E:\lake-MP-W\dataset\IUCN"
iucn_categories_paths = {
    'Critically Endangered': os.path.join(base_input_dir, 'CR'),
    'Endangered': os.path.join(base_input_dir, 'EN'),
    'Vulnerable': os.path.join(base_input_dir, 'VU'),
    'Near Threatened': os.path.join(base_input_dir, 'NT')
}

# 输出文件夹和文件路径
output_dir = r"E:\lake-MP-ani\data\IUCN_consolidated"
output_gpkg_path = os.path.join(output_dir, "all_species_classified.gpkg")

# --- 2. 准备工作区 ---
print(f"--- [步骤 2/4] 准备输出目录: {output_dir} ---")
os.makedirs(output_dir, exist_ok=True)
print("-> 输出目录已成功创建或已存在。")

# --- 3. 循环读取、标记并收集数据 ---
print("\n--- [步骤 3/4] 开始遍历、读取和标记各个分类等级的物种数据 ---")

gdfs_to_merge = []

for category_name, folder_path in iucn_categories_paths.items():
    shapefile_path = os.path.join(folder_path, 'data_0.shp')

    if not os.path.exists(shapefile_path):
        print(f"[警告] 文件未找到，跳过: {shapefile_path}")
        continue

    try:
        print(f"-> 正在处理: {category_name} (路径: {shapefile_path})")
        gdf = gpd.read_file(shapefile_path, engine="pyogrio")
        gdf['redlist_category'] = category_name
        gdfs_to_merge.append(gdf)
        print(f"   - 成功加载并标记了 {len(gdf)} 条记录。")
    except Exception as e:
        print(f"[严重错误] 读取文件时发生错误: {shapefile_path}\n{e}")
        exit()

# --- 4. 合并数据并保存结果 ---
if not gdfs_to_merge:
    print("\n[错误] 未能成功加载任何shapefile数据，无法进行合并。")
else:
    print("\n--- [步骤 4/4] 合并所有已标记的数据并保存最终结果 ---")
    print("-> 正在合并所有数据集...")
    master_gdf = pd.concat(gdfs_to_merge, ignore_index=True)
    print(f"-> 合并完成！总共包含 {len(master_gdf)} 条物种记录。")

    try:
        print(f"-> 正在保存整合后的GeoPackage文件至: {output_gpkg_path}")
        master_gdf.to_file(output_gpkg_path, driver='GPKG', engine="pyogrio")
        print("\n========================================================")
        print("  第一阶段数据整合成功完成！")
        print(f"  整合后的文件已保存为: {output_gpkg_path}")
        print("========================================================")
    except Exception as e:
        print(f"\n[严重错误] 保存最终GeoPackage文件时失败。\n{e}")