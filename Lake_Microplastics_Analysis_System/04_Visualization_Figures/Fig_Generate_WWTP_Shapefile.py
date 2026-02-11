import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import os
import numpy as np # 导入numpy以使用int64

# --- 1. 设置文件路径 ---
shapefile_path = r"E:\lake-MP-W\data\all8.shp"
csv_file_path = r"E:\lake-MP-W\data\wastewater\wastewater.csv"
output_directory = os.path.dirname(shapefile_path)
output_shapefile_path = os.path.join(output_directory, "all8_with_waste_discharge_final.shp")


# --- 2. 读取数据 ---
print("正在读取 Shapefile 文件...")
gdf_polygons = gpd.read_file(shapefile_path, engine='pyogrio', use_arrow=True)

# 检查并警告输入的CRS问题
if gdf_polygons.crs is None:
    print("\n警告: 输入的Shapefile 'all8.shp' 文件本身缺少坐标参考系(CRS)信息。\n      输出文件也将没有CRS，可能无法在GIS软件中正确定位。\n")

print("正在读取 CSV 文件...")
df_points = pd.read_csv(csv_file_path)

# --- 3. 坐标转换 ---
print("正在将 CSV 文件中的坐标转换为点数据...")
geometry = [Point(xy) for xy in tqdm(zip(df_points['lon'], df_points['lat']), total=len(df_points), desc="坐标转换中")]
gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry, crs=gdf_polygons.crs)

# --- 4. 空间连接 ---
print("正在进行空间连接以匹配点和面...")
joined_gdf = gpd.sjoin(gdf_points, gdf_polygons, how="inner", predicate="intersects")

# --- 5. 计算总量 ---
print("正在计算每个面元素的废物排放总量...")
columns_to_sum = ['Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge']
summed_waste_data = joined_gdf.groupby(joined_gdf.index_right)[columns_to_sum].sum()

# --- 6. 合并数据 ---
print("正在将计算结果赋值到 Shapefile 数据中...")
gdf_polygons = gdf_polygons.join(summed_waste_data)
gdf_polygons[columns_to_sum] = gdf_polygons[columns_to_sum].fillna(0)


# --- 7. 最终数据类型处理 (最关键的修改) ---
print("正在对累加列进行取整和类型转换...")
# <--- 关键修改开始 --->
# 对新添加的、含有大数值的列进行处理
for col in columns_to_sum:
    if col in gdf_polygons.columns:
        # 先四舍五入到最近的整数，然后转换为64位整型(int64)
        # 这可以完全避免因小数位数过多导致的字段宽度问题
        # 使用np.int64确保可以存储非常大的整数
        gdf_polygons[col] = gdf_polygons[col].round(0).astype(np.int64)
# <--- 关键修改结束 --->


# --- 8. 保存为新的 Shapefile 文件 ---
print(f"正在保存新的 Shapefile 文件至: {output_shapefile_path}")
# 即使gdf_polygons.crs为None，我们依然传递它，让pyogrio处理并发出警告
gdf_polygons.to_file(
    output_shapefile_path,
    engine='pyogrio',
    crs=gdf_polygons.crs
)

print("\n处理完成！")
print("注意：如果输入文件缺少CRS，输出文件也会缺少CRS。字段名长度警告是Shapefile格式的正常现象。")