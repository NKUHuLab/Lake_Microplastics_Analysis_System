# -*- coding: utf-8 -*-
"""
Training subset sensitivity — 敏感性分析公共函数
照搬 05_predict_global.py + 06_plot_map.py 逻辑: polygon面染色, NA=#E7E7E7
"""

import os, sys, warnings, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

warnings.filterwarnings("ignore")

REPO_ROOT = Path(os.environ.get("LAKE_MP_REPO_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.environ.get("LAKE_MP_DATA_ROOT", REPO_ROOT / "data"))
OUT_DIR = Path(os.environ.get("LAKE_MP_OUTPUT_DIR", REPO_ROOT / "outputs" / "training_sensitivity"))
PREDICT_CSV = Path(os.environ.get("LAKE_MP_PREDICT_DATA", DATA_ROOT / "model_products" / "feature_2022.csv.gz"))
TRAIN_CSV = Path(os.environ.get("LAKE_MP_TRAIN_DATA", DATA_ROOT / "model_products" / "train_data.csv"))
MAIN_MODEL_PATH = Path(os.environ.get("LAKE_MP_MODEL", DATA_ROOT / "model_products" / "random_forest_model.pkl"))
BORDERS_PATH = Path(os.environ.get("LAKE_MP_BORDERS_SHP", DATA_ROOT / "external" / "borders.shp"))
BASE_SHP = Path(os.environ.get("LAKE_MP_LAKES_SHP", DATA_ROOT / "external" / "predicted_lakes_2022.shp"))
FROZEN_METADATA_CSV = Path(os.environ.get("LAKE_MP_FROZEN_WATER_METADATA", DATA_ROOT / "evidence" / "training_sensitivity" / "water_metadata_quality_frozen.csv"))
plt.rcParams.update({"font.family": "Arial"})
os.makedirs(OUT_DIR, exist_ok=True)

ROBINSON_PROJ = "+proj=robin +datum=WGS84"

# === 子采样 ===
def subsample_geographic_balance(train_df, min_per_continent=25, frac_per_continent=0.45):
    """分层比例抽样: 每大洲至少保留min_per_continent个样本(若该洲有≥25个),
    对大洲按frac_per_continent比例抽样, 确保总n≥250 且 各洲均有代表性。
    V2修复: 旧版严格等量抽样导致n仅54, VIF无法可靠计算。"""
    rng = np.random.default_rng(42)
    continents = train_df['region'].dropna().unique()
    counts = train_df['region'].value_counts()
    idx_list = []
    summary = []
    for cont in continents:
        n_cont = counts.get(cont, 0)
        if n_cont <= min_per_continent:
            n_sample = n_cont  # 小洲全保留
        else:
            n_sample = max(min_per_continent, int(n_cont * frac_per_continent))
        idx = rng.choice(train_df[train_df['region'] == cont].index,
                               size=n_sample, replace=False)
        idx_list.append(idx)
        summary.append(f'{cont}: {n_cont}→{n_sample}')
    balanced_idx = np.concatenate(idx_list); rng.shuffle(balanced_idx)
    print(f"地理平衡子集 v2 (分层比例): 总计 {len(balanced_idx)} 样本 | {'; '.join(summary)}")
    return balanced_idx

def subsample_recent_years(train_df, min_year=2015):
    idx = train_df[train_df['Year']>=min_year].index
    print(f"近10年子集 ({min_year}+): {len(idx)} 样本")
    return idx

def subsample_high_quality(train_df):
    idx = train_df.index[train_df['high_quality_analysis_subset'].astype(bool)]
    print(f"高质量子集 (frozen selector): {len(idx)} 样本")
    return idx

def subsample_exact_mesh_size(train_df, mesh_size_um=333):
    """Select records measured with one exact nominal sampling mesh size."""
    if 'Mesh Size' not in train_df.columns:
        raise KeyError("Missing required metadata column: Mesh Size")
    mesh_text = train_df['Mesh Size'].astype(str)
    mesh_values = pd.to_numeric(mesh_text.str.extract(r'([-+]?\d*\.?\d+)')[0], errors='coerce')
    mask = np.isclose(mesh_values, mesh_size_um, rtol=0, atol=1e-9, equal_nan=False)
    idx = train_df.index[mask]
    print(f"精确网孔子集 (Mesh Size == {mesh_size_um} μm): {len(idx)} 样本")
    return idx

def geometric_mean_concentration_change(main_predictions, sensitivity_predictions):
    """Geometric-mean concentration change only; not a physical reservoir total."""
    main_mean = float(np.mean(np.asarray(main_predictions, dtype=float)))
    sensitivity_mean = float(np.mean(np.asarray(sensitivity_predictions, dtype=float)))
    relative_change = float(np.exp(sensitivity_mean-main_mean)-1.0)
    return dict(main_mean_ln=main_mean, sensitivity_mean_ln=sensitivity_mean,
                relative_change=relative_change, percent_change=100.0*relative_change)

# === RF训练 ===
def train_rf_subsample(X, y, random_state=42):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        bootstrap=True, max_depth=10, max_features=0.5,
        min_samples_leaf=1, min_samples_split=10, n_estimators=500,
        random_state=random_state, n_jobs=-1, oob_score=True
    )
    # Preserve training-only imputation for later prediction.
    medians = np.nanmedian(np.asarray(X, dtype=float), axis=0)
    X = np.where(np.isnan(np.asarray(X, dtype=float)), medians, X)
    model.fit(X, y)
    model.training_feature_medians_ = pd.Series(medians, index=FEATURE_LIST)
    print(f"  RF: OOB R2={model.oob_score_:.4f}")
    return model

# === 数据加载 ===
def load_train_with_metadata():
    df_supp = pd.read_csv(FROZEN_METADATA_CSV)
    df_train = pd.read_csv(TRAIN_CSV)
    if len(df_supp) != len(df_train):
        raise ValueError(f"Metadata/training row mismatch: {len(df_supp)} != {len(df_train)}")
    def extract_year(s):
        if pd.isna(s): return np.nan
        m = re.search(r'(19\d{2}|20\d{2})', str(s))
        return int(m.group(1)) if m else np.nan
    df_supp['Year'] = df_supp['Sampling Time'].apply(extract_year)
    df_train['Year'] = df_supp['Year'].values
    df_train['Sampling_Method_Category'] = df_supp['Sampling_Method_Category'].values
    df_train['Climate_Zone'] = df_supp['Climate_Zone'].values
    if int(df_supp['high_quality_analysis_subset'].sum()) != 291:
        raise ValueError("Frozen high-quality selector must contain exactly 291 rows")
    df_train['high_quality_analysis_subset'] = df_supp['high_quality_analysis_subset'].values
    df_train['Mesh Size'] = df_supp['Mesh Size'].values
    # 区域映射
    import geopandas as gpd
    from shapely.geometry import Point
    import cartopy.io.shapereader as shpreader
    try:
        shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        world = gpd.read_file(shp)
        world = world[['geometry','NAME','CONTINENT']].rename(columns={'NAME':'name','CONTINENT':'continent'})
        geometry = [Point(xy) for xy in zip(df_supp['lon'],df_supp['lat'])]
        gdf = gpd.GeoDataFrame(df_supp, geometry=geometry, crs='EPSG:4326')
        gdf = gpd.sjoin(gdf, world[['geometry','name','continent']], how='left', predicate='within')
        df_train['country'] = gdf['name'].values
        df_train['region'] = gdf['continent'].values
    except:
        def assign(lon,lat):
            if lon<-30 and lat>15: return 'North America'
            elif lon<-30 and lat<=15: return 'South America'
            elif -30<=lon<50: return 'Europe' if lat>30 else 'Africa'
            else: return 'Asia'
        df_train['region'] = df_supp.apply(lambda r: assign(r['lon'],r['lat']), axis=1)
    return df_train

# === 照搬 05_predict_global.py: 预测 + spatial join to shapefile ===
FEATURE_LIST = ['Lake_area','Shore_dev','Vol_total','Res_time',
                'Total_POP_SERVED','Average_DF','Primary_Waste_Discharge',
                'Secondary_Waste_Discharge','Advanced_Waste_Discharge',
                'RSE_paved','RSE_gravel','RSE_other','prec','emis_tyre_TSP_HEG',
                'emis_brake_TSP_HEG','PM2_5','PM10','Mismanaged','Total_Plast',
                'fish_gdp_sqkm','Cultivated_land','Artificial_surface']

def predict_and_save_shp(model, output_shp_path, feature_csv_path=None):
    """完全照搬 05_predict_global.py 逻辑"""
    if feature_csv_path is None: feature_csv_path = PREDICT_CSV
    features_df = pd.read_csv(feature_csv_path)
    features_df.columns = features_df.columns.str.strip()
    geo_df = gpd.read_file(BASE_SHP)

    X_predict = features_df[FEATURE_LIST].copy()
    X_predict = X_predict.fillna(model.training_feature_medians_)

    print(f"  预测 {len(features_df)} 数据点...")
    features_df['prediction'] = model.predict(X_predict)

    # 保存CSV
    csv_path = output_shp_path.replace('.shp', '_predictions.csv')
    features_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Spatial join → polygon
    features_df.dropna(subset=['lon','lat'], inplace=True)
    points_gdf = gpd.GeoDataFrame(features_df, geometry=gpd.points_from_xy(features_df.lon, features_df.lat))
    if geo_df.crs: points_gdf.set_crs(geo_df.crs, inplace=True)

    merged_gdf = gpd.sjoin(geo_df, points_gdf, how="left", predicate="contains")

    if merged_gdf.index.duplicated().any():
        print("  聚合多点湖泊...")
        original_geometries = geo_df.loc[merged_gdf.index.unique()].geometry
        original_crs = geo_df.crs
        merged_gdf = merged_gdf.groupby(merged_gdf.index).mean(numeric_only=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry=original_geometries, crs=original_crs)

    print(f"  空间连接: {len(merged_gdf)} 湖泊面匹配")
    merged_gdf.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"  Shapefile: {output_shp_path}")
    return merged_gdf, features_df

# === 储量CI ===
def compute_concentration_sum_tree_spread(model, feature_csv_path=None):
    """Unweighted concentration sum and tree-resampling range; not total stock or a sampling CI."""
    if feature_csv_path is None: feature_csv_path = PREDICT_CSV
    features_df = pd.read_csv(feature_csv_path)
    features_df.columns = features_df.columns.str.strip()
    X = features_df[FEATURE_LIST].copy().fillna(model.training_feature_medians_)
    tree_preds = np.array([tree.predict(X.values) for tree in model.estimators_])
    n_trees, n_lakes = tree_preds.shape
    per_lake_mean = tree_preds.mean(axis=0)
    R_point = np.exp(per_lake_mean).sum()
    rng = np.random.RandomState(42)
    R_bootstrap = [np.exp(tree_preds[rng.choice(n_trees, size=n_trees, replace=True), :].mean(axis=0)).sum()
                   for _ in range(1000)]
    ci_low, ci_high = np.percentile(R_bootstrap, [2.5, 97.5])
    print(f"  Unweighted concentration sum and tree-resampling range: {R_point:.2e} [{ci_low:.2e}, {ci_high:.2e}]")
    return R_point, ci_low, ci_high, per_lake_mean

# === 缓存 ===
def save_sensitivity_cache(key, model, merged_gdf, R_point, ci_low, ci_high):
    cache_dir = os.path.join(OUT_DIR, 'sensitivity_cache')
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f'{key}.pkl'), 'wb') as f:
        pickle.dump((model, merged_gdf, R_point, ci_low, ci_high), f)
    print(f"  缓存: {key}.pkl")

# === 主模型预测加载 ===
def load_main_shp():
    shp = BASE_SHP
    if os.path.exists(shp):
        return gpd.read_file(shp)
    raise FileNotFoundError(f"找不到: {shp}")

# === 照搬 06_plot_map.py: 面染色地图 + 图例单独PDF ===
def plot_sensitivity_map(merged_gdf, cmap_name, title, output_base):
    """完全照搬 06_plot_map.py 的地图绘制逻辑"""
    world_raw = gpd.read_file(BORDERS_PATH)

    if merged_gdf.crs is None:
        merged_gdf.set_crs(epsg=4326, inplace=True)

    merged_gdf['plot_value'] = pd.to_numeric(merged_gdf['prediction'], errors='coerce')
    vmin, vmax = merged_gdf['plot_value'].dropna().quantile([0.01, 0.99])
    norm = Normalize(vmin=vmin, vmax=vmax)

    print("  投影 + 修复几何...")
    gdf_lakes_proj = merged_gdf.to_crs(ROBINSON_PROJ)
    world_proj = world_raw.to_crs(ROBINSON_PROJ)
    world_proj['geometry'] = world_proj.geometry.buffer(0)

    name_col = next((c for c in ['NAME','name','ADMIN','SOVEREIGNT'] if c in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col]!='Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia",case=False,na=False)]

    fig, ax = plt.subplots(1,1,figsize=(20,10))
    ax.set_facecolor('white')

    # Layer 1: landmass white (海洋也是白色背景)
    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    # Layer 2: NA/no-data 湖泊面 → #E7E7E7 灰色, 绝不白色
    gdf_na = gdf_lakes_proj[gdf_lakes_proj['plot_value'].isna() | (gdf_lakes_proj['plot_value'] == 0)]
    gdf_data = gdf_lakes_proj[gdf_lakes_proj['plot_value'].notna() & (gdf_lakes_proj['plot_value'] != 0)]
    if len(gdf_na) > 0:
        gdf_na.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)

    # Layer 3: 有数据的湖泊面 → colored by prediction
    gdf_data.plot(column='plot_value', cmap=cmap_name, norm=norm,
                  linewidth=0, edgecolor='none', ax=ax, legend=False, zorder=3)

    # Layer 4: 国界线
    world_for_borders.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5, zorder=4)
    ax.set_axis_off()

    ax.text(0.5, 0.98, title, transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold')

    # 保存地图
    map_png = output_base + '.png'
    map_pdf = output_base + '.pdf'
    fig.savefig(map_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(map_pdf, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close(fig)
    print(f"  地图: {map_png}")

    # 单独图例PDF (照搬06)
    fig_legend = plt.figure(figsize=(2.5, 7))
    ax_legend = fig_legend.add_axes([0.05, 0.05, 0.2, 0.9])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax_dummy = fig_legend.add_axes([0,0,0,0], visible=False)
    im = ax_dummy.imshow(gradient, aspect='auto', cmap=cmap_name, norm=norm)
    cbar = fig_legend.colorbar(im, cax=ax_legend, orientation='vertical')
    cbar.set_label("Predicted ln(MP)", fontsize=14, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    legend_pdf = output_base + '_legend.pdf'
    fig_legend.savefig(legend_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)
    print(f"  图例: {legend_pdf}")

# === 差异地图 (面染色版本) ===
def plot_difference_map(merged_main, merged_sens, title, output_base):
    world_raw = gpd.read_file(BORDERS_PATH)

    if merged_main.crs is None: merged_main.set_crs(epsg=4326, inplace=True)
    if merged_sens.crs is None: merged_sens.set_crs(epsg=4326, inplace=True)

    diff_vals = pd.to_numeric(merged_main['prediction'], errors='coerce') - \
                pd.to_numeric(merged_sens['prediction'], errors='coerce')
    merged_main = merged_main.copy()
    merged_main['diff'] = diff_vals

    abs_max = max(abs(merged_main['diff'].quantile(0.01)), abs(merged_main['diff'].quantile(0.99)))
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    gdf_proj = merged_main.to_crs(ROBINSON_PROJ)
    world_proj = world_raw.to_crs(ROBINSON_PROJ)
    world_proj['geometry'] = world_proj.geometry.buffer(0)
    name_col = next((c for c in ['NAME','name','ADMIN','SOVEREIGNT'] if c in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col]!='Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia",case=False,na=False)]

    fig, ax = plt.subplots(1,1,figsize=(20,10))
    ax.set_facecolor('white')
    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    gdf_na = gdf_proj[gdf_proj['diff'].isna()]
    gdf_data = gdf_proj[gdf_proj['diff'].notna()]
    if len(gdf_na) > 0:
        gdf_na.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)
    gdf_data.plot(column='diff', cmap='RdYlBu_r', norm=norm,
                  linewidth=0, edgecolor='none', ax=ax, legend=False, zorder=3)
    world_for_borders.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5, zorder=4)
    ax.set_axis_off()
    ax.text(0.5, 0.98, title, transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold')

    for ext, dpi in [('.png',300),('.pdf',300)]:
        fig.savefig(output_base+ext, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  差异地图: {output_base}.png/.pdf")

    # Legend
    fig_legend = plt.figure(figsize=(2.5, 7))
    ax_legend = fig_legend.add_axes([0.05, 0.05, 0.2, 0.9])
    grad = np.linspace(-abs_max, abs_max, 256).reshape(1,-1)
    ax_dummy = fig_legend.add_axes([0,0,0,0], visible=False)
    im = ax_dummy.imshow(grad, aspect='auto', cmap='RdYlBu_r')
    cbar = fig_legend.colorbar(im, cax=ax_legend, orientation='vertical')
    cbar.set_label("Δ ln(MP)", fontsize=14, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    legend_pdf = output_base + '_legend.pdf'
    fig_legend.savefig(legend_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)
    print(f"  图例: {legend_pdf}")

# === xlsx导出 ===
def export_prediction_xlsx_from_shp(merged_gdf, output_path):
    out = merged_gdf[['prediction']].copy()
    out['Predicted_ln_MP'] = out['prediction']
    out['Predicted_MP_items_per_m3'] = np.exp(out['prediction'])
    out.to_excel(output_path, index=False)
    print(f"  xlsx: {output_path}")

if __name__ == '__main__':
    print("敏感性分析公共函数 (polygon face 版本) 已加载")
