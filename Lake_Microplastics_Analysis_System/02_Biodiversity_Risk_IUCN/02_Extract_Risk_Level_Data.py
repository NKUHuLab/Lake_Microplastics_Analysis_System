import os
import glob
import geopandas as gpd
import pandas as pd

# --- 1. 配置工作路径 ---
print("--- [步骤 1/3] 初始化濒危等级数据提取配置 ---")

# 输入数据的根目录
base_input_dir = r"E:\lake-MP-W\dataset\IUCN"

# 需要处理的濒危等级文件夹列表
# (CR, EN, VU, NT)
folder_names = [
    "CR", "EN", "VU", "NT"
]

# 输出CSV文件的目录
output_dir = r"E:\lake-MP-W\dataset\IUCN\bio"

# --- 2. 准备工作区 ---
print(f"--- [步骤 2/3] 准备输出目录: {output_dir} ---")
os.makedirs(output_dir, exist_ok=True)
print("-> 输出目录已成功创建或已存在。")

# --- 3. 循环处理每个文件夹并提取数据 ---
print("\n--- [步骤 3/3] 开始遍历文件夹并提取数据 ---")

# 用于存储从每个shapefile读取的DataFrame的列表
all_data_frames = []

for folder in folder_names:
    folder_path = os.path.join(base_input_dir, folder)

    if not os.path.isdir(folder_path):
        print(f"\n[警告] 文件夹不存在，跳过: {folder_path}")
        continue

    print(f"\n-> 正在处理文件夹: {folder}")

    # 在文件夹内查找所有 .shp 文件
    shapefiles = glob.glob(os.path.join(folder_path, '*.shp'))

    if not shapefiles:
        print(f"   - [警告] 在 {folder_path} 中未找到任何 .shp 文件。")
        continue

    # 循环处理找到的每一个 shapefile
    for shp_path in shapefiles:
        try:
            print(f"   - 正在读取文件: {os.path.basename(shp_path)}")
            gdf = gpd.read_file(shp_path, engine="pyogrio")

            # 移除 'geometry' 列，只保留属性数据
            df = pd.DataFrame(gdf.drop(columns='geometry'))

            # 将提取出的数据添加到列表中
            all_data_frames.append(df)
            print(f"     - 成功读取 {len(df)} 条记录。")

        except Exception as e:
            print(f"   - [严重错误] 读取或处理文件时失败: {shp_path}")
            print(f"     错误信息: {e}")

if not all_data_frames:
    print("\n[处理结束] 未能成功加载任何数据，程序退出。")
else:
    print("\n-> 所有文件读取完毕，正在合并数据...")
    # 将列表中的所有DataFrame合并成一个
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"-> 合并完成，总共 {len(combined_df)} 条记录。")

    # 基于所有列删除完全重复的行
    print("-> 正在删除完全重复的生物记录...")
    records_before_dedupe = len(combined_df)
    deduplicated_df = combined_df.drop_duplicates()
    records_after_dedupe = len(deduplicated_df)
    print(f"   - 完成！删除了 {records_before_dedupe - records_after_dedupe} 条重复记录。")
    print(f"   - 最终剩余 {records_after_dedupe} 条唯一记录。")

    # 构建最终输出路径并保存
    output_csv_path = os.path.join(output_dir, "iucn_threatened_species_attributes.csv")
    print(f"-> 正在将所有唯一记录保存至一个CSV文件: {output_csv_path}")

    try:
        # 使用 utf-8-sig 编码以确保 Excel 等软件能正确显示特殊字符
        deduplicated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print("\n==========================================================")
        print("  濒危等级数据提取与整合成功完成！")
        print(f"  所有唯一生物记录已保存至: {output_csv_path}")
        print("==========================================================")
    except Exception as e:
        print(f"\n[严重错误] 保存最终CSV文件时失败。")
        print(f"         错误信息: {e}")