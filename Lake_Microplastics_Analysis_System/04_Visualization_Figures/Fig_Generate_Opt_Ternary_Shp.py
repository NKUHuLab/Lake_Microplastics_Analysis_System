import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import os
import numpy as np

# --- 1. 设置文件路径 ---
# Shapefile的输入路径
shapefile_path = r"E:\lake-MP-W\data\all8.shp"
# 包含经纬度和原始数据的CSV文件路径
csv_file_path = r"E:\lake-MP-W\data\opt\data\processed_output\Xchange.csv"
# 输出文件的目录
output_directory = r"E:\lake-MP-W\data\shp"
# 最终输出的Shapefile文件名
output_shapefile_path = os.path.join(output_directory, "all8_categorized_summary.shp")
# 分类对照表的输出路径
lookup_csv_path = os.path.join(output_directory, "category_summary_lookup.csv")


# --- 2. 定义分类函数 (之前已为您设计好的逻辑) ---
def get_awd_level(value):
    """根据 Advanced_Waste_Discharge 的值返回其影响等级"""
    if value <= 0:
        return 0
    elif value <= 5000:
        return 1
    elif value <= 20000:
        return 2
    elif value <= 100000:
        return 3
    else:
        return 4


def get_fish_level(value):
    """根据 fish_gdp_sqkm 的值返回其影响等级"""
    if value >= 0:
        return 0
    elif value > -1000:
        return 1
    elif value > -10000:
        return 2
    elif value > -100000:
        return 3
    else:
        return 4


def get_cult_level(value):
    """根据 Cultivated_land 的值返回其影响等级"""
    if value >= 0:
        return 0
    elif value > -5:
        return 1
    elif value > -10:
        return 2
    elif value > -20:
        return 3
    else:
        return 4


# --- 3. 读取数据 ---
print("步骤 1/9: 正在读取 Shapefile 文件...")
try:
    gdf_polygons = gpd.read_file(shapefile_path, engine='pyogrio', use_arrow=True)
except Exception:
    gdf_polygons = gpd.read_file(shapefile_path)  # 如果pyogrio失败，则使用默认引擎

if gdf_polygons.crs is None:
    print("\n警告: 输入的Shapefile缺少坐标参考系(CRS)信息。\n")

print("步骤 2/9: 正在读取 CSV 文件...")
df_points = pd.read_csv(csv_file_path)

# --- 4. 将CSV坐标转换为GeoDataFrame ---
# 假设CSV中有'lon'和'lat'列
if 'lon' not in df_points.columns or 'lat' not in df_points.columns:
    print("错误: CSV文件中未找到 'lon' 或 'lat' 列，无法进行空间聚合。请检查列名。")
    exit()

print("步骤 3/9: 正在将 CSV 坐标转换为地理点要素...")
geometry = [Point(xy) for xy in tqdm(zip(df_points['lon'], df_points['lat']), total=len(df_points), desc="坐标转换中")]
gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry, crs=gdf_polygons.crs)

# --- 5. 空间聚合：计算每个面内各字段的总量 ---
print("步骤 4/9: 正在进行空间连接以匹配点和面...")
joined_gdf = gpd.sjoin(gdf_points, gdf_polygons, how="inner", predicate="intersects")

print("步骤 5/9: 正在计算每个面要素内各字段的总和...")
columns_to_aggregate = ['Advanced_Waste_Discharge', 'fish_gdp_sqkm', 'Cultivated_land']
summed_data = joined_gdf.groupby(joined_gdf.index_right)[columns_to_aggregate].sum()

print("步骤 6/9: 正在将聚合结果合并回面图层...")
gdf_final = gdf_polygons.join(summed_data)
gdf_final[columns_to_aggregate] = gdf_final[columns_to_aggregate].fillna(0)

# --- 6. 应用分类和打代号逻辑 ---
print("步骤 7/9: 正在对聚合后的数据应用分类逻辑...")
# 应用函数计算每个因素的影响等级
gdf_final['AWD_Level'] = gdf_final['Advanced_Waste_Discharge'].apply(get_awd_level)
gdf_final['Fish_Level'] = gdf_final['fish_gdp_sqkm'].apply(get_fish_level)
gdf_final['Cult_Level'] = gdf_final['Cultivated_land'].apply(get_cult_level)

# 找出每行的最高影响等级
gdf_final['Max_Level'] = gdf_final[['AWD_Level', 'Fish_Level', 'Cult_Level']].max(axis=1)

# 定义每个等级对应的代号
awd_codes = {1: 'AWD1', 2: 'AWD2', 3: 'AWD3', 4: 'AWD4'}
fish_codes = {1: 'FISH1', 2: 'FISH2', 3: 'FISH3', 4: 'FISH4'}
cult_codes = {1: 'CULT1', 2: 'CULT2', 3: 'CULT3', 4: 'CULT4'}


# 根据最高等级生成最终的组合代号
def create_final_label(row):
    max_level = row['Max_Level']
    if max_level == 0:
        return 'No_Impact'
    parts = []
    if row['AWD_Level'] == max_level: parts.append(awd_codes.get(row['AWD_Level']))
    if row['Fish_Level'] == max_level: parts.append(fish_codes.get(row['Fish_Level']))
    if row['Cult_Level'] == max_level: parts.append(cult_codes.get(row['Cult_Level']))
    return '_'.join(filter(None, parts))


gdf_final['cat_name'] = gdf_final.apply(create_final_label, axis=1)

# --- 7. 创建短代码和对照表 ---
print("步骤 8/9: 正在为类别名称创建短数字ID和对照表...")
unique_categories = gdf_final['cat_name'].unique()
category_mapping = {name: i + 1 for i, name in enumerate(unique_categories)}
gdf_final['cat_id'] = gdf_final['cat_name'].map(category_mapping)

lookup_df = pd.DataFrame(list(category_mapping.items()), columns=['Category_Name', 'Category_ID'])
lookup_df.to_csv(lookup_csv_path, index=False, encoding='utf-8-sig')

# --- 8. 清理数据并准备输出 ---
# 删除中间计算列，保留最终结果
columns_to_drop = ['AWD_Level', 'Fish_Level', 'Cult_Level', 'Max_Level']
gdf_final.drop(columns=columns_to_drop, inplace=True)

# 为了shapefile的兼容性，对原始累加列取整
for col in columns_to_aggregate:
    gdf_final[col] = gdf_final[col].round(0).astype(np.int64)

# --- 9. 保存最终结果 ---
print(f"步骤 9/9: 正在保存最终的 Shapefile 文件至: {output_shapefile_path}")
try:
    # 选择要导出的列，以保持shapefile整洁
    output_columns = ['geometry', 'cat_id', 'cat_name'] + columns_to_aggregate
    gdf_to_save = gdf_final[output_columns]

    gdf_to_save.to_file(
        output_shapefile_path,
        engine='pyogrio',
        crs=gdf_polygons.crs
    )
    print("\n--------------------")
    print("处理完成！")
    print(f"新的Shapefile已保存在: {output_shapefile_path}")
    print(f"类别名称和ID的对照表已保存在: {lookup_csv_path}")
    print("--------------------")

except Exception as e:
    print(f"\n保存文件时出错: {e}")