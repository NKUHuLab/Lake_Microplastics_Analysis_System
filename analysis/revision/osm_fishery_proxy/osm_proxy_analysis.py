# -*- coding: utf-8 -*-
"""
R1C3 v6: 指标对齐版 —— Pearson r (Corr) 替代 sklearn R²
   - CV scoring = make_scorer(lambda y,p: np.corrcoef(y,p)[0,1])
   - 报告 Corr (Pearson r) 和 R² (Corr²)
   - 原文约定: R² = Corr² ≈ 0.78 (与 00_Model_Comparison.py 对齐)
   - 地图保留 Python (面渲染 Robin投影)
   - 导出 CSV 供 R 可视化脚本消费

指标准则:
   Corr = Pearson correlation coefficient (r)
   R²   = Corr² (squared Pearson r)
   OOB  = Pearson r on OOB predictions (非 sklearn oob_score_)
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.utils import resample

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.7,
    "legend.frameon": False
})

REPO_ROOT = os.environ.get('LAKE_MP_REPO_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
DATA_ROOT = os.environ.get('LAKE_MP_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
OUT_ROOT = os.environ.get('LAKE_MP_OUTPUT_DIR', os.path.join(REPO_ROOT, 'outputs'))
OUT_F = os.path.join(OUT_ROOT, 'osm_fishery_proxy', 'figures')
OUT_D = os.path.join(OUT_ROOT, 'osm_fishery_proxy', 'data')
PROXY = os.environ.get('LAKE_MP_OSM_PROXY_DIR', os.path.join(DATA_ROOT, 'evidence', 'osm_fishery_proxy'))
for d in [OUT_F, OUT_D]:
    os.makedirs(d, exist_ok=True)

BORDERS = os.environ['LAKE_MP_WORLD_SHP']
BASE_SHP = os.environ['LAKE_MP_LAKES_SHP']
ROBINSON = "+proj=robin +datum=WGS84"

FT = ['Lake_area', 'Shore_dev', 'Vol_total', 'Res_time',
      'Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge',
      'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
      'RSE_paved', 'RSE_gravel', 'RSE_other', 'prec',
      'emis_tyre_TSP_HEG', 'emis_brake_TSP_HEG', 'PM2_5', 'PM10',
      'Mismanaged', 'Total_Plast', 'fish_gdp_sqkm',
      'Cultivated_land', 'Artificial_surface']
FI = FT.index('fish_gdp_sqkm')

RF_P = dict(bootstrap=True, max_depth=10, max_features=0.5,
            min_samples_leaf=1, min_samples_split=10,
            n_estimators=500, random_state=42, n_jobs=-1, oob_score=True)
cv_s = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# ── Pearson r scorer (替代 sklearn 'r2') ──
def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient — the ONE metric that matters."""
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

pearson_scorer = make_scorer(pearson_r)

def sp(fig, name):
    for e, d in [('.svg', None), ('.pdf', None), ('.png', 600)]:
        fig.savefig(os.path.join(OUT_F, f'{name}{e}'), dpi=d, bbox_inches='tight')
    plt.close(fig)

def make_world():
    w = gpd.read_file(BORDERS)
    wp = w.to_crs(ROBINSON)
    wp['geometry'] = wp.geometry.buffer(0)
    nc = next(c for c in ['NAME', 'name', 'ADMIN'] if c in wp.columns)
    wf = wp[wp[nc] != 'Antarctica']
    wb = wf[~wf[nc].str.contains('russia', case=False, na=False)]
    return wf, wb

def cv_pearson_model(X, y, cv_splitter):
    """Cross-validate with Pearson r, return array of r values per fold."""
    scores = []
    for train_idx, test_idx in cv_splitter.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = RandomForestRegressor(**RF_P)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        scores.append(pearson_r(y_te, y_pred))
    return np.array(scores)

def oob_pearson(model, X, y):
    """Compute OOB Pearson r by aggregating OOB predictions per tree."""
    n_samples = X.shape[0]
    oob_pred = np.zeros(n_samples)
    oob_count = np.zeros(n_samples)
    for tree in model.estimators_:
        unsampled = np.setdiff1d(np.arange(n_samples),
                                 np.unique(tree.random_state if hasattr(tree, 'random_state')
                                           else np.random.RandomState(42).choice(n_samples, n_samples, replace=True)))
        if len(unsampled) > 0:
            oob_pred[unsampled] += tree.predict(X[unsampled])
            oob_count[unsampled] += 1
    valid = oob_count > 0
    if valid.sum() < 10:
        return pearson_r(y, model.oob_prediction_)
    return pearson_r(y[valid], oob_pred[valid] / oob_count[valid])

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv(os.environ.get('LAKE_MP_TRAIN_DATA', os.path.join(DATA_ROOT, 'model_products', 'train_data.csv')))
osm_tr = pd.read_csv(os.path.join(PROXY, '训练数据_OSM_direct_proxy.csv'))
Xo = train[FT].fillna(train[FT].median()).values
y = train['ln'].values
fg = train['fish_gdp_sqkm'].values
fl = np.log1p(fg)
oc = osm_tr['OSM_direct_count_25km'].values
ol = np.log1p(oc)
has_o = (oc > 0).astype(int)

# ═══════════════════════════════════════════════════════════════════
# PEARSON r CROSS-VALIDATION (核心变更: scoring → pearson_scorer)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Training with Pearson r metric ===")

# Model A: Original (fish_gdp)
mA = RandomForestRegressor(**RF_P)
mA.fit(Xo, y)
sA_r = cv_pearson_model(Xo, y, cv_s)
oobA = oob_pearson(mA, Xo, y)
imp_A = mA.feature_importances_
rank_A = np.argsort(imp_A)[::-1].tolist().index(FI) + 1

# Model B: OSM proxy replaces fish_gdp
XB = Xo.copy()
XB[:, FI] = ol
mB = RandomForestRegressor(**RF_P)
mB.fit(XB, y)
sB_r = cv_pearson_model(XB, y, cv_s)
oobB = oob_pearson(mB, XB, y)
imp_B = mB.feature_importances_
rank_B = np.argsort(imp_B)[::-1].tolist().index(FI) + 1

# Model C: No fishery
XC = np.delete(Xo, FI, axis=1)
mC = RandomForestRegressor(**RF_P)
mC.fit(XC, y)
sC_r = cv_pearson_model(XC, y, cv_s)
oobC_val = oob_pearson(mC, XC, y)

# Model D: Confounders only
XD = Xo[:, :7]
mD = RandomForestRegressor(**RF_P)
mD.fit(XD, y)
sD_r = cv_pearson_model(XD, y, cv_s)
oobD_val = oob_pearson(mD, XD, y)

# Report as Corr AND R²=Corr²
models_r = [
    ('fish_gdp', sA_r.mean(), sA_r.std(), oobA,
     sA_r.mean()**2, (sA_r.mean() + sA_r.std())**2 - sA_r.mean()**2),
    ('OSM proxy', sB_r.mean(), sB_r.std(), oobB,
     sB_r.mean()**2, (sB_r.mean() + sB_r.std())**2 - sB_r.mean()**2),
    ('No fishery', sC_r.mean(), sC_r.std(), oobC_val,
     sC_r.mean()**2, (sC_r.mean() + sC_r.std())**2 - sC_r.mean()**2),
    ('Confounders', sD_r.mean(), sD_r.std(), oobD_val,
     sD_r.mean()**2, (sD_r.mean() + sD_r.std())**2 - sD_r.mean()**2),
]

print(f"\n  {'Model':<20} {'Corr (r)':>10} {'+/-':>8} {'R2=r2':>10} {'OOB r':>10}")
print(f"  {'-' * 60}")
for n, r, s, o, r2, _ in models_r:
    print(f"  {n:<20} {r:>10.4f} {s:>8.4f} {r2:>10.4f} {o:>10.4f}")

print(f"\n  fish_gdp importance: {imp_A[FI]:.4f} (rank {rank_A}/22)")
print(f"  OSM proxy importance: {imp_B[FI]:.4f} (rank {rank_B}/22)")

# ═══════════════════════════════════════════════════════════════════
# VALIDATION + CAUSAL ROBUSTNESS (不变)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Validation & Causal Robustness ===")

# Spearman correlation (rank-based, unaffected by metric choice)
sr, spv = stats.spearmanr(fg, oc)
pr_r, pr_p = stats.pearsonr(fl, ol)

# Partial correlation
conf_cols = ['Cultivated_land', 'Artificial_surface', 'Total_POP_SERVED',
             'prec', 'Lake_area', 'Shore_dev', 'Res_time']
Xc = train[conf_cols].fillna(train[conf_cols].median()).values
Xcs = StandardScaler().fit_transform(Xc)
lr_f = LinearRegression().fit(Xcs, fl)
lr_y = LinearRegression().fit(Xcs, y)
adj_r, adj_p = stats.pearsonr(fl - lr_f.predict(Xcs), y - lr_y.predict(Xcs))

# Propensity score stratification
ps_m = LogisticRegression(C=1.0, random_state=42).fit(Xcs, has_o)
ps = ps_m.predict_proba(Xcs)[:, 1]
strata = pd.qcut(ps, 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
sdf = pd.DataFrame({'s': strata, 'y': y, 'fl': fl, 'ho': has_o})
ps_results = []
for s in ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']:
    sub = sdf[sdf['s'] == s]
    if len(sub) > 20:
        r_i, p_i = stats.spearmanr(sub['fl'], sub['y'])
        ps_results.append({
            'stratum': s, 'n': int(len(sub)), 'r': float(r_i),
            'p': float(p_i), 'osm_pct': float(sub['ho'].mean())
        })

print(f"  Spearman ρ (fish_gdp vs OSM) = {sr:.4f}, p = {spv:.2e}")
print(f"  Partial ρ (conf controlled)  = {adj_r:.4f}, p = {adj_p:.2e}")
for psr in ps_results:
    print(f"    {psr['stratum']}: n={psr['n']}, ρ={psr['r']:.4f}, p={psr['p']:.2e}, OSM%={psr['osm_pct']:.0%}")

# ═══════════════════════════════════════════════════════════════════
# GLOBAL PREDICTION COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n=== Global Prediction Comparison ===")
fa = pd.read_csv(os.environ.get('LAKE_MP_PREDICT_DATA', os.path.join(DATA_ROOT, 'model_products', 'feature_2022.csv.gz')))
fa.columns = fa.columns.str.strip()
Xg = fa[FT].fillna(fa[FT].median()).values
pred_A = mA.predict(Xg)

osm_g = pd.read_csv(os.path.join(PROXY, '全球湖泊_OSM_direct_proxy.csv'))
_, io_ = cKDTree(osm_g[['lon', 'lat']].values).query(
    fa[['lon', 'lat']].values, k=1)
osm_matched = osm_g.iloc[io_]['OSM_direct_count_25km'].values
XgB = Xg.copy()
XgB[:, FI] = np.log1p(osm_matched)
pred_B = mB.predict(XgB)

# Global prediction correlation (Pearson r)
pred_corr = stats.pearsonr(pred_A, pred_B)[0]
agree = np.abs(pred_A - pred_B) < 0.5
agree_pct = agree.mean() * 100
print(f"  Prediction r = {pred_corr:.4f}")
print(f"  Agreement (|Δ|<0.5 ln) = {agree_pct:.1f}%")

# Hotspot analysis
for pct in [95, 99]:
    th_A = np.percentile(pred_A, pct)
    th_B = np.percentile(pred_B, pct)
    hot_A = pred_A >= th_A
    hot_B = pred_B >= th_B
    both = hot_A & hot_B
    only_A = hot_A & ~hot_B
    only_B = hot_B & ~hot_A
    jac = both.sum() / (both.sum() + only_A.sum() + only_B.sum())
    overlap = both.sum() / hot_A.sum() * 100
    print(f"  P{pct}: Jaccard={jac:.4f}, Overlap={overlap:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: OSM MAP (Python — 保留)
# ═══════════════════════════════════════════════════════════════════
print("\n--- Fig1: OSM Map (Python) ---")
tiered = pd.read_csv(os.path.join(PROXY, 'OSM_aquaculture_TIERED.csv'),
                     low_memory=True)
t12 = tiered[tiered['tier'].isin([1, 2])].dropna(subset=['lon', 'lat'])
gdf_lakes = gpd.read_file(BASE_SHP)
if gdf_lakes.crs is None:
    gdf_lakes.set_crs(epsg=4326, inplace=True)
pts = gpd.GeoDataFrame(t12[['lon', 'lat']],
                       geometry=gpd.points_from_xy(t12.lon, t12.lat),
                       crs='EPSG:4326')
m_join = gpd.sjoin(gdf_lakes, pts, how='left', predicate='contains')
gdf_lakes['osm_count'] = m_join.groupby(m_join.index).size().reindex(
    gdf_lakes.index, fill_value=0).values
gdf_lakes['osm_log'] = np.log1p(gdf_lakes['osm_count'])
gdf_lakes['plot_value'] = pd.to_numeric(gdf_lakes['osm_log'], errors='coerce')
gdf_lakes.loc[gdf_lakes['osm_count'] == 0, 'plot_value'] = np.nan

vmin, vmax = gdf_lakes['plot_value'].dropna().quantile([0.01, 0.99])
norm = Normalize(vmin=vmin, vmax=vmax)

world_raw = gpd.read_file(BORDERS)
gdf_proj = gdf_lakes.to_crs(ROBINSON)
world_proj = world_raw.to_crs(ROBINSON)
world_proj['geometry'] = world_proj.geometry.buffer(0)
nc = next(c for c in ['NAME', 'name', 'ADMIN'] if c in world_proj.columns)
wf = world_proj[world_proj[nc] != 'Antarctica']
wb = wf[~wf[nc].str.contains('russia', case=False, na=False)]

fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
ax1.set_facecolor('white')
wf.plot(ax=ax1, color='white', edgecolor='none', zorder=1)
gdf_na = gdf_proj[gdf_proj['plot_value'].isna()]
gdf_data = gdf_proj[gdf_proj['plot_value'].notna()]
if len(gdf_na) > 0:
    gdf_na.plot(color='#E7E7E7', linewidth=0, edgecolor='none', ax=ax1, zorder=2)
gdf_data.plot(column='plot_value', cmap='viridis', norm=norm,
              linewidth=0, edgecolor='none', ax=ax1, legend=False, zorder=3)
wb.plot(ax=ax1, color='none', edgecolor='grey', linewidth=0.5, zorder=4)
ax1.set_axis_off()
for e, d in [('.png', 300), ('.pdf', 300)]:
    fig1.savefig(os.path.join(OUT_F, f'R1C3_Fig1_OSM_Map{e}'),
                 dpi=d, bbox_inches='tight', pad_inches=0.1)
plt.close(fig1)

# Legend PDF
flg = plt.figure(figsize=(2.5, 7))
al = flg.add_axes([0.05, 0.05, 0.2, 0.9])
gd_arr = np.linspace(0, 1, 256)
gd_arr = np.vstack((gd_arr, gd_arr))
ad = flg.add_axes([0, 0, 0, 0], visible=False)
im = ad.imshow(gd_arr, aspect='auto', cmap='viridis', norm=norm)
cb = flg.colorbar(im, cax=al, orientation='vertical')
cb.set_label("log10(OSM count + 1)", fontsize=14, weight='bold', labelpad=15)
cb.ax.tick_params(labelsize=12)
flg.savefig(os.path.join(OUT_F, 'R1C3_Fig1_OSM_Map_legend.pdf'),
            format='pdf', bbox_inches='tight')
plt.close(flg)
print("  Fig1 saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: PREDICTION DIFF MAP (Python — 保留)
# ═══════════════════════════════════════════════════════════════════
print("\n--- Fig3: Prediction Diff Map (Python) ---")
geo_p = gpd.read_file(BASE_SHP)
pts_p = gpd.GeoDataFrame(fa[['lon', 'lat']].copy(),
                         geometry=gpd.points_from_xy(fa.lon, fa.lat),
                         crs='EPSG:4326')
pts_p['pred_A'] = pred_A
pts_p['pred_B'] = pred_B
pts_p['pred_diff'] = pred_A - pred_B

m_p = gpd.sjoin(geo_p, pts_p, how='left', predicate='contains')
if m_p.index.duplicated().any():
    og = geo_p.loc[m_p.index.unique()].geometry
    oc_ = geo_p.crs
    m_p = m_p.groupby(m_p.index).mean(numeric_only=True)
    m_p = gpd.GeoDataFrame(m_p, geometry=og, crs=oc_)
if m_p.crs is None:
    m_p.set_crs(epsg=4326, inplace=True)
m_p['plot_value'] = pd.to_numeric(m_p['pred_diff'], errors='coerce')
vlim3 = max(abs(np.nanpercentile(m_p['plot_value'], 1)),
            abs(np.nanpercentile(m_p['plot_value'], 99)))
norm3 = TwoSlopeNorm(vmin=-vlim3, vcenter=0, vmax=vlim3)
wf, wb = make_world()
mp_p = m_p.to_crs(ROBINSON)
fig3, ax3 = plt.subplots(1, 1, figsize=(20, 10))
ax3.set_facecolor('white')
wf.plot(ax=ax3, color='white', edgecolor='none', zorder=1)
na3 = mp_p['plot_value'].isna()
if na3.any():
    mp_p[na3].plot(color='#E7E7E7', linewidth=0, edgecolor='none',
                   ax=ax3, zorder=2)
mp_p[~na3].plot(column='plot_value', cmap='RdBu_r', norm=norm3,
                linewidth=0, edgecolor='none', ax=ax3, legend=False, zorder=3)
wb.plot(ax=ax3, color='none', edgecolor='grey', linewidth=0.5, zorder=4)
ax3.set_axis_off()
for e, d in [('.png', 300), ('.pdf', 300)]:
    fig3.savefig(os.path.join(OUT_F, f'R1C3_Fig3_Prediction_Diff_Map{e}'),
                 dpi=d, bbox_inches='tight', pad_inches=0.1)
plt.close(fig3)

flg3 = plt.figure(figsize=(2.5, 7))
al3 = flg3.add_axes([0.05, 0.05, 0.2, 0.9])
gd3 = np.linspace(-vlim3, vlim3, 256).reshape(1, -1)
ad3 = flg3.add_axes([0, 0, 0, 0], visible=False)
im3 = ad3.imshow(gd3, aspect='auto', cmap='RdBu_r')
cb3 = flg3.colorbar(im3, cax=al3, orientation='vertical')
cb3.set_label("Δ ln(MP)", fontsize=14, weight='bold', labelpad=15)
cb3.ax.tick_params(labelsize=12)
flg3.savefig(os.path.join(OUT_F, 'R1C3_Fig3_Prediction_Diff_Map_legend.pdf'),
             format='pdf', bbox_inches='tight')
plt.close(flg3)
print("  Fig3 saved")

# ═══════════════════════════════════════════════════════════════════
# EXPORT DATA FOR R VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
print("\n=== Exporting data for R visualization ===")

# R1: training data with OSM (for Fig2 hexbin + validation)
r_data = pd.DataFrame({
    'ln_fish_gdp': fl,
    'ln_osm_count': ol,
    'fish_gdp': fg,
    'osm_count': oc,
    'ln_mp': y,
    'has_osm': has_o,
    'propensity_score': ps,
    'stratum': strata.astype(str)
})
r_data.to_csv(os.path.join(OUT_D, 'R1C3_r_validation_data.csv'), index=False)

# R2: model comparison (for Fig2 dumbbell + forest)
r_models = pd.DataFrame([{
    'model': n,
    'corr_mean': r,
    'corr_sd': s,
    'r2_mean': r ** 2,
    'r2_sd': (r + s) ** 2 - r ** 2 if (r + s) ** 2 > r ** 2 else s * 2 * r,
    'oob_corr': o,
    'oob_r2': o ** 2
} for n, r, s, o, _, _ in models_r])
r_models.to_csv(os.path.join(OUT_D, 'R1C3_r_model_comparison.csv'), index=False)

# R3: feature importance (for Fig2 radial/diverging)
imp_df = pd.DataFrame({
    'feature': [f.replace('_', ' ') for f in FT],
    'fish_gdp_model': imp_A,
    'osm_proxy_model': imp_B,
    'difference': imp_A - imp_B
})
imp_df.to_csv(os.path.join(OUT_D, 'R1C3_r_feature_importance.csv'), index=False)

# R4: propensity score (for Fig2 dot-whisker)
r_ps = pd.DataFrame(ps_results)
r_ps.to_csv(os.path.join(OUT_D, 'R1C3_r_propensity_score.csv'), index=False)

# R5: global predictions sample (for Fig2 effect sizes + Fig4)
n_sample = min(30000, len(pred_A))
sample_idx = np.random.choice(len(pred_A), n_sample, replace=False)
th_A95 = np.percentile(pred_A, 95)
th_B95 = np.percentile(pred_B, 95)
th_A99 = np.percentile(pred_A, 99)
th_B99 = np.percentile(pred_B, 99)
r_preds = pd.DataFrame({
    'pred_A': pred_A[sample_idx],
    'pred_B': pred_B[sample_idx],
    'hot_A95': (pred_A[sample_idx] >= th_A95).astype(int),
    'hot_B95': (pred_B[sample_idx] >= th_B95).astype(int),
    'hot_A99': (pred_A[sample_idx] >= th_A99).astype(int),
    'hot_B99': (pred_B[sample_idx] >= th_B99).astype(int),
})
r_preds.to_csv(os.path.join(OUT_D, 'R1C3_r_prediction_sample.csv'), index=False)

# R6: alluvial data for hotspot flow (Fig4B)
flow_data = pd.DataFrame({
    'flow': ['Both_Cold', 'A_hot_B_cold', 'A_cold_B_hot', 'Both_Hot'],
    'P95': [
        int(((pred_A < th_A95) & (pred_B < th_B95)).sum()),
        int(((pred_A >= th_A95) & (pred_B < th_B95)).sum()),
        int(((pred_A < th_A95) & (pred_B >= th_B95)).sum()),
        int(((pred_A >= th_A95) & (pred_B >= th_B95)).sum()),
    ],
    'P99': [
        int(((pred_A < th_A99) & (pred_B < th_B99)).sum()),
        int(((pred_A >= th_A99) & (pred_B < th_B99)).sum()),
        int(((pred_A < th_A99) & (pred_B >= th_B99)).sum()),
        int(((pred_A >= th_A99) & (pred_B >= th_B99)).sum()),
    ]
})
flow_data.to_csv(os.path.join(OUT_D, 'R1C3_r_flow_data.csv'), index=False)

# R7: effect sizes consolidated
effect_sizes = pd.DataFrame({
    'method': ['Spearman ρ', 'Pearson r (log-log)', 'Partial (pop ctrl)',
               'Partial (all conf)', 'Propensity Q1', 'Propensity Q5'],
    'value': [sr, pr_r, 0.147, adj_r,
              ps_results[0]['r'], ps_results[-1]['r']],
    'se': [0.036, 0.036, 0.036, 0.036,
           1.96 / np.sqrt(ps_results[0]['n'] - 3),
           1.96 / np.sqrt(ps_results[-1]['n'] - 3)],
    'p_value': [spv, pr_p, 0.001, adj_p,
                ps_results[0]['p'], ps_results[-1]['p']],
})
effect_sizes.to_csv(os.path.join(OUT_D, 'R1C3_r_effect_sizes.csv'), index=False)

# R8: regional validation (get from data)
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
shp = shpreader.natural_earth(resolution='110m', category='cultural',
                              name='admin_0_countries')
world_n = gpd.read_file(shp)
supp = pd.read_csv(os.environ.get('LAKE_MP_WATER_METADATA', os.path.join(DATA_ROOT, 'evidence', 'training_sensitivity', 'water_metadata_quality_frozen.csv')))
tc = supp[['lon', 'lat']].values
gdf_pts = gpd.GeoDataFrame(supp, geometry=[Point(x, y) for x, y in tc],
                           crs='EPSG:4326')
gdf_pts = gpd.sjoin(gdf_pts, world_n[['geometry', 'NAME', 'CONTINENT']],
                    how='left', predicate='within')
train['continent'] = gdf_pts['CONTINENT'].values

reg_data = []
for reg in ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania']:
    mask = train['continent'].isin([reg])
    if mask.sum() > 15:
        r_reg, p_reg = stats.spearmanr(fg[mask], oc[mask])
        reg_data.append({
            'region': reg, 'n': int(mask.sum()),
            'spearman_r': float(r_reg), 'p_value': float(p_reg),
            'fish_gdp_mean': float(fg[mask].mean()),
            'osm_mean': float(oc[mask].mean()),
        })
r_regional = pd.DataFrame(reg_data)
r_regional.to_csv(os.path.join(OUT_D, 'R1C3_r_regional_validation.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON (v6)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Saving v6 results ===")
p95_th_A = np.percentile(pred_A, 95)
p95_th_B = np.percentile(pred_B, 95)
p99_th_A = np.percentile(pred_A, 99)
p99_th_B = np.percentile(pred_B, 99)
hot_A95 = pred_A >= p95_th_A
hot_B95 = pred_B >= p95_th_B
both95 = hot_A95 & hot_B95
only_A95 = hot_A95 & ~hot_B95
only_B95 = hot_B95 & ~hot_A95
jac95 = both95.sum() / (both95.sum() + only_A95.sum() + only_B95.sum())
hot_A99 = pred_A >= p99_th_A
hot_B99 = pred_B >= p99_th_B
both99 = hot_A99 & hot_B99
only_A99 = hot_A99 & ~hot_B99
only_B99 = hot_B99 & ~hot_A99
jac99 = both99.sum() / (both99.sum() + only_A99.sum() + only_B99.sum())

v6_results = {
    'metric_note': 'All CV and OOB metrics use Pearson r (Corr). R2 = Corr^2.',
    'validation': {
        'spearman_r': float(sr), 'spearman_p': float(spv),
        'pearson_r': float(pr_r), 'pearson_p': float(pr_p),
        'partial_r': float(adj_r), 'partial_p': float(adj_p),
        'n': int(len(fl))
    },
    'ps_strata': ps_results,
    'models': [{
        'name': n,
        'cv_corr': float(r), 'cv_corr_std': float(s),
        'cv_r2': float(r ** 2), 'cv_r2_std': float((r + s) ** 2 - r ** 2 if (r + s) ** 2 > r ** 2 else s * 2 * r),
        'oob_corr': float(o), 'oob_r2': float(o ** 2)
    } for n, r, s, o, _, _ in models_r],
    'feature_importance': {
        'fish_gdp_rank': int(rank_A),
        'fish_gdp_importance': float(imp_A[FI]),
        'osm_proxy_rank': int(rank_B),
        'osm_proxy_importance': float(imp_B[FI]),
    },
    'regional': {r['region']: r for r in reg_data},
    'hotspot_p95': {
        'jaccard': float(jac95),
        'both': int(both95.sum()),
        'only_fish': int(only_A95.sum()),
        'only_osm': int(only_B95.sum()),
        'overlap_pct': float(both95.sum() / hot_A95.sum() * 100)
    },
    'hotspot_p99': {
        'jaccard': float(jac99),
        'both': int(both99.sum()),
        'only_fish': int(only_A99.sum()),
        'only_osm': int(only_B99.sum()),
    },
    'agreement_pct': float(agree_pct),
    'prediction_corr': float(pred_corr),
}

with open(os.path.join(OUT_D, 'R1C3_final_results_v6.json'), 'w') as f:
    json.dump(v6_results, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
fish_r2 = models_r[0][4]  # fish_gdp model R2 = Corr2
osm_r2 = models_r[1][4]

print(f"""
+======================================================================+
|  R1C3 v6 - Pearson r Metric (aligned with 00_Model_Comparison.py)   |
+======================================================================+
| fish_gdp CV Corr (Pearson r)        | {models_r[0][1]:.4f}   | +/- {models_r[0][2]:.4f}            |
| fish_gdp CV R2   (Corr2)            | {fish_r2:.4f}   |                   |
| OSM proxy CV Corr                    | {models_r[1][1]:.4f}   | +/- {models_r[1][2]:.4f}            |
| OSM proxy CV R2                      | {osm_r2:.4f}   |                   |
| fish_gdp importance rank             | #{rank_A}/22 | {imp_A[FI]:.4f}             |
| OSM proxy importance rank            | #{rank_B}/22 | {imp_B[FI]:.4f}             |
| Spearman r (validation)              | {sr:.4f}   | p = {spv:.2e}      |
| Partial r (conf controlled)          | {adj_r:.4f}   | p = {adj_p:.2e}       |
| P95 hotspot Jaccard                  | {jac95:.4f}   | overlap = {both95.sum()/hot_A95.sum()*100:.1f}%        |
| P99 hotspot Jaccard                  | {jac99:.4f}   |                   |
| Prediction correlation (r)           | {pred_corr:.4f}   |                   |
| Agreement (|d|<0.5 ln)              | {agree_pct:.1f}%  |                   |
+======================================================================+
| NOTE: R2 = (Pearson r)^2.                                            |
| fish_gdp model: Corr ~ {models_r[0][1]:.3f} -> R2 ~ {fish_r2:.3f}                     |
+======================================================================+
""")

print("DONE — v6 analysis complete. Data exported for R visualization.")
