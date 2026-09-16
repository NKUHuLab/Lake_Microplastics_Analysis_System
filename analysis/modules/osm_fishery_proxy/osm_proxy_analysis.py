# -*- coding: utf-8 -*-
"""
OSMProxy v6: report Pearson r, r², and conventional test-set R² separately.
   - 地图保留 Python (面渲染 Robin投影)
   - 导出 CSV 供 R 可视化脚本消费

Metric definitions:
   Corr = Pearson correlation coefficient (r)
   r² = squared Pearson correlation (reported for compatibility)
   R² = sklearn r2_score on held-out predictions
   OOB Corr = Pearson r on sklearn's OOB predictions
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
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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

def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient."""
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

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

def cv_model_metrics(X, y, cv_splitter):
    """Evaluate folds with an imputer fitted only on each training partition."""
    pearson_scores = []
    r2_scores = []
    for train_idx, test_idx in cv_splitter.split(X):
        imputer = SimpleImputer(strategy='median')
        X_tr = imputer.fit_transform(X[train_idx])
        X_te = imputer.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        m = RandomForestRegressor(**RF_P)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        pearson_scores.append(pearson_r(y_te, y_pred))
        r2_scores.append(r2_score(y_te, y_pred))
    return np.asarray(pearson_scores), np.asarray(r2_scores)

def oob_pearson(model, y):
    """Compute Pearson r from RandomForestRegressor's actual OOB predictions."""
    oob_predictions = np.asarray(model.oob_prediction_)
    valid = np.isfinite(oob_predictions) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    return pearson_r(np.asarray(y)[valid], oob_predictions[valid])


def partial_pearson(x, y, controls):
    """Partial Pearson correlation and two-sided t-test with control df."""
    controls = np.asarray(controls)
    x_resid = np.asarray(x) - LinearRegression().fit(controls, x).predict(controls)
    y_resid = np.asarray(y) - LinearRegression().fit(controls, y).predict(controls)
    r, _ = stats.pearsonr(x_resid, y_resid)
    control_rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(x_resid)), controls])) - 1
    df = len(x_resid) - control_rank - 2
    if df <= 0 or not np.isfinite(r):
        return r, np.nan
    t_stat = r * np.sqrt(df / max(1.0 - r ** 2, np.finfo(float).eps))
    return r, 2 * stats.t.sf(abs(t_stat), df)


def correlation_se(r, n, n_controls=0):
    """Delta-method standard error for a correlation coefficient."""
    df = n - n_controls - 3
    return (1 - r ** 2) / np.sqrt(df) if df > 0 else np.nan

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv(os.environ.get('LAKE_MP_TRAIN_DATA', os.path.join(DATA_ROOT, 'model_products', 'train_data.csv')))
osm_tr = pd.read_csv(os.path.join(PROXY, '训练数据_OSM_direct_proxy.csv'))
Xo_raw = train[FT].to_numpy()
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

imputer_A = SimpleImputer(strategy='median').fit(Xo_raw)
Xo = imputer_A.transform(Xo_raw)

# Model A: Original (fish_gdp)
mA = RandomForestRegressor(**RF_P)
mA.fit(Xo, y)
sA_r, sA_r2 = cv_model_metrics(Xo_raw, y, cv_s)
oobA = oob_pearson(mA, y)
imp_A = mA.feature_importances_
rank_A = np.argsort(imp_A)[::-1].tolist().index(FI) + 1

# Model B: OSM proxy replaces fish_gdp
XB_raw = Xo_raw.copy()
XB_raw[:, FI] = ol
imputer_B = SimpleImputer(strategy='median').fit(XB_raw)
XB = imputer_B.transform(XB_raw)
mB = RandomForestRegressor(**RF_P)
mB.fit(XB, y)
sB_r, sB_r2 = cv_model_metrics(XB_raw, y, cv_s)
oobB = oob_pearson(mB, y)
imp_B = mB.feature_importances_
rank_B = np.argsort(imp_B)[::-1].tolist().index(FI) + 1

# Model C: No fishery
XC_raw = np.delete(Xo_raw, FI, axis=1)
imputer_C = SimpleImputer(strategy='median').fit(XC_raw)
XC = imputer_C.transform(XC_raw)
mC = RandomForestRegressor(**RF_P)
mC.fit(XC, y)
sC_r, sC_r2 = cv_model_metrics(XC_raw, y, cv_s)
oobC_val = oob_pearson(mC, y)

# Model D: Confounders only
XD_raw = Xo_raw[:, :7]
imputer_D = SimpleImputer(strategy='median').fit(XD_raw)
XD = imputer_D.transform(XD_raw)
mD = RandomForestRegressor(**RF_P)
mD.fit(XD, y)
sD_r, sD_r2 = cv_model_metrics(XD_raw, y, cv_s)
oobD_val = oob_pearson(mD, y)

# Report Pearson r, r² (compatibility), and conventional held-out R² separately.
models_r = [
    ('fish_gdp', sA_r.mean(), sA_r.std(), sA_r.mean() ** 2, sA_r2.mean(), sA_r2.std(), oobA),
    ('OSM proxy', sB_r.mean(), sB_r.std(), sB_r.mean() ** 2, sB_r2.mean(), sB_r2.std(), oobB),
    ('No fishery', sC_r.mean(), sC_r.std(), sC_r.mean() ** 2, sC_r2.mean(), sC_r2.std(), oobC_val),
    ('Confounders', sD_r.mean(), sD_r.std(), sD_r.mean() ** 2, sD_r2.mean(), sD_r2.std(), oobD_val),
]

print(f"\n  {'Model':<20} {'Corr (r)':>10} {'+/-':>8} {'r²':>10} {'R²':>10} {'OOB r':>10}")
print(f"  {'-' * 60}")
for n, r, r_sd, r_squared, r2, r2_sd, oob_r in models_r:
    print(f"  {n:<20} {r:>10.4f} {r_sd:>8.4f} {r_squared:>10.4f} {r2:>10.4f} {oob_r:>10.4f}")

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
Xc = SimpleImputer(strategy='median').fit_transform(train[conf_cols])
Xcs = StandardScaler().fit_transform(Xc)
X_pop = SimpleImputer(strategy='median').fit_transform(train[['Total_POP_SERVED']])
partial_pop_r, partial_pop_p = partial_pearson(fl, y, X_pop)
adj_r, adj_p = partial_pearson(fl, y, Xcs)

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
print(f"  Partial r (population controlled) = {partial_pop_r:.4f}, p = {partial_pop_p:.2e}")
print(f"  Partial r (all confounders controlled) = {adj_r:.4f}, p = {adj_p:.2e}")
for psr in ps_results:
    print(f"    {psr['stratum']}: n={psr['n']}, ρ={psr['r']:.4f}, p={psr['p']:.2e}, OSM%={psr['osm_pct']:.0%}")

# ═══════════════════════════════════════════════════════════════════
# GLOBAL PREDICTION COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n=== Global Prediction Comparison ===")
fa = pd.read_csv(os.environ.get('LAKE_MP_PREDICT_DATA', os.path.join(DATA_ROOT, 'model_products', 'feature_2022.csv.gz')))
fa.columns = fa.columns.str.strip()
Xg_raw = fa[FT].to_numpy()
Xg = imputer_A.transform(Xg_raw)
pred_A = mA.predict(Xg)

osm_g = pd.read_csv(os.path.join(PROXY, '全球湖泊_OSM_direct_proxy.csv'))
_, io_ = cKDTree(osm_g[['lon', 'lat']].values).query(
    fa[['lon', 'lat']].values, k=1)
osm_matched = osm_g.iloc[io_]['OSM_direct_count_25km'].values
XgB_raw = Xg_raw.copy()
XgB_raw[:, FI] = np.log1p(osm_matched)
XgB = imputer_B.transform(XgB_raw)
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
    fig1.savefig(os.path.join(OUT_F, f'OSMProxy_Fig1_OSM_Map{e}'),
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
flg.savefig(os.path.join(OUT_F, 'OSMProxy_Fig1_OSM_Map_legend.pdf'),
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
    fig3.savefig(os.path.join(OUT_F, f'OSMProxy_Fig3_Prediction_Diff_Map{e}'),
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
flg3.savefig(os.path.join(OUT_F, 'OSMProxy_Fig3_Prediction_Diff_Map_legend.pdf'),
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
r_data.to_csv(os.path.join(OUT_D, 'OSMProxy_r_validation_data.csv'), index=False)

# Model comparison: Pearson correlation and predictive R2 are separate metrics.
oob_r2_by_model = dict(zip(['fish_gdp', 'OSM proxy', 'No fishery', 'Confounders'],
                           [mA.oob_score_, mB.oob_score_, mC.oob_score_, mD.oob_score_]))
r_models = pd.DataFrame([{
    'model': n,
    'corr_mean': r,
    'corr_sd': s,
    'corr_squared': r_squared,
    'r2_mean': predictive_r2,
    'r2_sd': predictive_r2_sd,
    'oob_corr': o,
    'oob_corr_squared': o ** 2,
    'oob_r2': oob_r2_by_model[n]
} for n, r, s, r_squared, predictive_r2, predictive_r2_sd, o in models_r])
r_models.to_csv(os.path.join(OUT_D, 'OSMProxy_r_model_comparison.csv'), index=False)

# R3: feature importance (for Fig2 radial/diverging)
imp_df = pd.DataFrame({
    'feature': [f.replace('_', ' ') for f in FT],
    'fish_gdp_model': imp_A,
    'osm_proxy_model': imp_B,
    'difference': imp_A - imp_B
})
imp_df.to_csv(os.path.join(OUT_D, 'OSMProxy_r_feature_importance.csv'), index=False)

# R4: propensity score (for Fig2 dot-whisker)
r_ps = pd.DataFrame(ps_results)
r_ps.to_csv(os.path.join(OUT_D, 'OSMProxy_r_propensity_score.csv'), index=False)

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
r_preds.to_csv(os.path.join(OUT_D, 'OSMProxy_r_prediction_sample.csv'), index=False)

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
flow_data.to_csv(os.path.join(OUT_D, 'OSMProxy_r_flow_data.csv'), index=False)

# R7: effect sizes consolidated
effect_sizes = pd.DataFrame({
    'method': ['Spearman ρ', 'Pearson r (log-log)', 'Partial (pop ctrl)',
               'Partial (all conf)', 'Propensity Q1', 'Propensity Q5'],
    'value': [sr, pr_r, partial_pop_r, adj_r,
              ps_results[0]['r'], ps_results[-1]['r']],
    'se': [correlation_se(sr, len(y)), correlation_se(pr_r, len(y)),
           correlation_se(partial_pop_r, len(y), 1), correlation_se(adj_r, len(y), Xcs.shape[1]),
           correlation_se(ps_results[0]['r'], ps_results[0]['n']),
           correlation_se(ps_results[-1]['r'], ps_results[-1]['n'])],
    'p_value': [spv, pr_p, partial_pop_p, adj_p,
               ps_results[0]['p'], ps_results[-1]['p']],
})
effect_sizes.to_csv(os.path.join(OUT_D, 'OSMProxy_r_effect_sizes.csv'), index=False)

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
r_regional.to_csv(os.path.join(OUT_D, 'OSMProxy_r_regional_validation.csv'), index=False)

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
    'metric_note': 'Pearson r and squared correlation are separate from predictive R2 (1-SSE/SST). OOB metrics are conditional on the final preprocessing.',
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
        'cv_corr_squared': float(r_squared),
        'cv_r2': float(predictive_r2), 'cv_r2_std': float(predictive_r2_sd),
        'oob_corr': float(o), 'oob_corr_squared': float(o ** 2), 'oob_r2': float(oob_r2_by_model[n])
    } for n, r, s, r_squared, predictive_r2, predictive_r2_sd, o in models_r],
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

with open(os.path.join(OUT_D, 'OSMProxy_final_results_v6.json'), 'w') as f:
    json.dump(v6_results, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
fish_r2 = models_r[0][4]  # fish_gdp model mean held-out predictive R2
osm_r2 = models_r[1][4]

print(f"""
+======================================================================+
|  OSM proxy validation: Pearson r and held-out predictive R2   |
+======================================================================+
| fish_gdp CV Corr (Pearson r)        | {models_r[0][1]:.4f}   | +/- {models_r[0][2]:.4f}            |
| fish_gdp held-out R2            | {fish_r2:.4f}   |                   |
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
| NOTE: R2 is 1-SSE/SST; it is not squared correlation.                                            |
| fish_gdp model: Corr ~ {models_r[0][1]:.3f} ; predictive R2 ~ {fish_r2:.3f}                     |
+======================================================================+
""")

print("DONE — v6 analysis complete. Data exported for R visualization.")
