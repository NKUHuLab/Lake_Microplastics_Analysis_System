# -*- coding: utf-8 -*-
"""
生成湖泊微塑料优化约束的完整脚本 (Final Version)
1. 土地利用约束: 基于时间序列历史波动性
2. 渔业约束: 基于FAO历史产量波动性
3. 技术前沿约束 (废水/道路/垃圾): 基于分位数回归 (Quantile Regression) 的单次稳健拟合
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告以保持输出整洁
warnings.filterwarnings("ignore")


# ==========================================
# 1. 辅助函数与配置
# ==========================================

def create_un_code_map():
    """创建一个UN Code到国家英文名的映射字典"""
    return {
        51: 'Armenia', 4: 'Afghanistan', 8: 'Albania', 12: 'Algeria', 16: 'American Samoa', 20: 'Andorra',
        24: 'Angola', 28: 'Antigua and Barbuda', 32: 'Argentina', 36: 'Australia', 40: 'Austria',
        44: 'Bahamas', 48: 'Bahrain', 52: 'Barbados', 58: 'Belgium-Luxembourg', 50: 'Bangladesh',
        60: 'Bermuda', 64: 'Bhutan', 68: 'Bolivia (Plurinational State of)', 72: 'Botswana', 76: 'Brazil',
        533: 'Aruba', 84: 'Belize', 86: 'British Indian Ocean Territory', 90: 'Solomon Islands',
        96: 'Brunei Darussalam', 100: 'Bulgaria', 104: 'Myanmar', 108: 'Burundi', 10: 'Antarctica',
        74: 'Bouvet Island', 120: 'Cameroon', 124: 'Canada', 128: 'Canton and Enderbury Islands',
        132: 'Cabo Verde', 136: 'Cayman Islands', 140: 'Central African Republic', 144: 'Sri Lanka',
        148: 'Chad', 152: 'Chile', 156: 'China', 162: 'Christmas Island', 166: 'Cocos (Keeling) Islands',
        170: 'Colombia', 174: 'Comoros', 178: 'Congo', 184: 'Cook Islands', 188: 'Costa Rica', 192: 'Cuba',
        196: 'Cyprus', 200: 'Czechoslovakia', 31: 'Azerbaijan', 204: 'Benin', 208: 'Denmark',
        212: 'Dominica', 214: 'Dominican Republic', 112: 'Belarus', 218: 'Ecuador', 818: 'Egypt',
        222: 'El Salvador', 226: 'Equatorial Guinea', 230: 'Ethiopia PDR', 233: 'Estonia',
        234: 'Faroe Islands', 238: 'Falkland Islands (Malvinas)', 242: 'Fiji', 246: 'Finland', 250: 'France',
        254: 'French Guiana', 258: 'French Polynesia', 260: 'French Southern Territories', 262: 'Djibouti',
        268: 'Georgia', 266: 'Gabon', 270: 'Gambia', 274: 'Gaza Strip', 276: 'Germany',
        70: 'Bosnia and Herzegovina', 288: 'Ghana', 292: 'Gibraltar', 296: 'Kiribati', 300: 'Greece',
        304: 'Greenland', 308: 'Grenada', 312: 'Guadeloupe', 316: 'Guam', 320: 'Guatemala', 324: 'Guinea',
        328: 'Guyana', 334: 'Heard Island and McDonald Islands', 332: 'Haiti', 336: 'Holy See',
        340: 'Honduras', 344: 'China, Hong Kong SAR', 348: 'Hungary', 191: 'Croatia', 352: 'Iceland',
        356: 'India', 360: 'Indonesia', 364: 'Iran (Islamic Republic of)', 368: 'Iraq', 372: 'Ireland',
        376: 'Israel', 380: 'Italy', 384: "Cote d'Ivoire", 398: 'Kazakhstan', 388: 'Jamaica', 392: 'Japan',
        396: 'Johnston Island', 400: 'Jordan', 417: 'Kyrgyzstan', 404: 'Kenya', 116: 'Cambodia',
        408: "Democratic People's Republic of Korea", 410: 'Republic of Korea', 414: 'Kuwait', 428: 'Latvia',
        418: "Lao People's Democratic Republic", 422: 'Lebanon', 426: 'Lesotho', 430: 'Liberia',
        434: 'Libya', 438: 'Liechtenstein', 440: 'Lithuania', 584: 'Marshall Islands',
        446: 'China, Macao SAR', 450: 'Madagascar', 454: 'Malawi', 458: 'Malaysia', 462: 'Maldives',
        466: 'Mali', 470: 'Malta', 474: 'Martinique', 478: 'Mauritania', 480: 'Mauritius', 484: 'Mexico',
        488: 'Midway Islands', 492: 'Monaco', 496: 'Mongolia', 500: 'Montserrat', 504: 'Morocco',
        508: 'Mozambique', 583: 'Micronesia (Federated States of)', 498: 'Republic of Moldova',
        516: 'Namibia', 520: 'Nauru', 524: 'Nepal', 528: 'Netherlands (Kingdom of the)',
        530: 'Netherlands Antilles', 536: 'Neutral Zone', 540: 'New Caledonia', 807: 'North Macedonia',
        548: 'Vanuatu', 554: 'New Zealand', 558: 'Nicaragua', 562: 'Niger', 566: 'Nigeria', 570: 'Niue',
        574: 'Norfolk Island', 578: 'Norway', 580: 'Northern Mariana Islands',
        582: 'Trust Territory of the Pacific Islands', 586: 'Pakistan', 591: 'Panama', 203: 'Czechia',
        598: 'Papua New Guinea', 600: 'Paraguay', 604: 'Peru', 608: 'Philippines', 612: 'Pitcairn',
        616: 'Poland', 620: 'Portugal', 624: 'Guinea-Bissau', 626: 'Timor-Leste', 630: 'Puerto Rico',
        232: 'Eritrea', 634: 'Qatar', 585: 'Palau', 716: 'Zimbabwe', 638: 'Reunion', 642: 'Romania',
        646: 'Rwanda', 643: 'Russian Federation', 891: 'Serbia and Montenegro',
        654: 'Saint Helena, Ascension and Tristan da Cunha', 659: 'Saint Kitts and Nevis',
        662: 'Saint Lucia', 666: 'Saint Pierre and Miquelon', 670: 'Saint Vincent and the Grenadines',
        674: 'San Marino', 678: 'Sao Tome and Principe', 682: 'Saudi Arabia', 686: 'Senegal',
        690: 'Seychelles', 694: 'Sierra Leone', 705: 'Slovenia', 703: 'Slovakia', 702: 'Singapore',
        706: 'Somalia', 710: 'South Africa', 724: 'Spain', 732: 'Western Sahara', 736: 'Sudan (former)',
        740: 'Suriname', 762: 'Tajikistan', 748: 'Eswatini', 752: 'Sweden', 756: 'Switzerland',
        760: 'Syrian Arab Republic', 795: 'Turkmenistan', 158: 'Taiwan Province of China',
        834: 'United Republic of Tanzania', 764: 'Thailand', 768: 'Togo', 772: 'Tokelau', 776: 'Tonga',
        780: 'Trinidad and Tobago', 512: 'Oman', 788: 'Tunisia', 792: 'Turkiye',
        796: 'Turks and Caicos Islands', 784: 'United Arab Emirates', 800: 'Uganda', 798: 'Tuvalu',
        810: 'USSR', 826: 'United Kingdom of Great Britain and Northern Ireland', 804: 'Ukraine',
        840: 'United States of America', 581: 'United States Minor Outlying Islands', 854: 'Burkina Faso',
        858: 'Uruguay', 860: 'Uzbekistan', 862: 'Venezuela (Bolivarian Republic of)', 704: 'Viet Nam',
        231: 'Ethiopia', 92: 'British Virgin Islands', 850: 'United States Virgin Islands',
        872: 'Wake Island', 876: 'Wallis and Futuna Islands', 882: 'Samoa', 402: 'West Bank',
        886: 'Yemen Arab Republic', 720: 'Yemen, Democratic', 890: 'Yugoslavia SFR', 887: 'Yemen',
        180: 'Democratic Republic of the Congo', 894: 'Zambia', 896: 'Other NEI', 56: 'Belgium',
        442: 'Luxembourg', 660: 'Anguilla', 830: 'Channel Islands', 744: 'Svalbard and Jan Mayen Islands',
        833: 'Isle of Man', 175: 'Mayotte', 239: 'South Georgia and the South Sandwich Islands',
        688: 'Serbia', 499: 'Montenegro', 836: 'United Republic of Tanzania, Zanzibar',
        532: 'Netherlands Antilles', 248: 'Aland Islands', 535: 'Bonaire, Sint Eustatius and Saba',
        831: 'Guernsey', 832: 'Jersey', 531: 'Curacao', 652: 'Saint Barthelemy',
        663: 'Saint Martin (French part)', 534: 'Sint Maarten (Dutch part)', 728: 'South Sudan', 680: 'Sark',
        729: 'Sudan', 275: 'Palestine', 0: 'European Union'
    }


def plot_final_curve(data_for_curve, group_name, x_col, y_col, model, r_squared, n_initial, output_path):
    """
    绘制并保存拟合曲线图，用于诊断和展示技术前沿。
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 生成预测线数据
    x_min, x_max = data_for_curve[x_col].min(), data_for_curve[x_col].max()
    x_range = np.linspace(x_min, x_max, 200)

    # 构造预测用的DataFrame，注意处理特殊列名
    safe_x_col_param = re.sub(r'[^A-Za-z0-9_]+', '', x_col)
    pred_input = pd.DataFrame({safe_x_col_param: x_range})

    # 预测
    try:
        y_pred = model.predict(pred_input)
    except Exception as e:
        print(f"绘图预测时出错: {e}")
        return

    # 绘制散点和曲线
    ax.scatter(data_for_curve[x_col], data_for_curve[y_col], alpha=0.5, color='gray', s=15, label='Observation')
    ax.plot(x_range, y_pred, color='#c0392b', linewidth=3, label=f'Frontier (q={model.q:.2f})')

    ax.set_title(f'Technological Frontier: {y_col} vs {x_col}\nGroup: {group_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.legend()

    # 添加统计信息文本
    stats_text = (f"Pseudo R²: {r_squared:.3f}\n"
                  f"N: {len(data_for_curve)}/{n_initial}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()

    # 安全的文件名
    safe_group = re.sub(r'[^A-Za-z0-9_]+', '_', str(group_name))
    safe_y = re.sub(r'[^A-Za-z0-9_]+', '', y_col)
    filename = f"frontier_{safe_group}_{safe_y}.pdf"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


# ==========================================
# 2. 核心处理逻辑函数
# ==========================================

def process_land_use_constraints(time_series_path):
    """
    处理土地利用（耕地/人造地表）的时间序列数据。
    逻辑：利用2000-2022年的历史波动性（标准差）来设定未来的允许变化范围。
    """
    print(f"[Step 1] 正在处理土地利用约束 (Source: {time_series_path})...")

    files = glob.glob(os.path.join(time_series_path, "20*.csv"))
    if not files:
        print(f"[Error] 未找到时间序列CSV文件。")
        return None

    # 读取并合并所有年份数据
    df_list = []
    for f in sorted(files):
        try:
            year = int(os.path.basename(f).split('.')[0])
            temp = pd.read_csv(f)
            temp['year'] = year
            temp['row_id'] = temp.index
            df_list.append(temp)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    full_df.rename(columns={'Cultivated_land': 'Cultivated_Land', 'Artificial_surface': 'Artificialsurface'},
                   inplace=True)

    # 筛选出在2022年存在的数据作为基准
    df_2022 = full_df[full_df['year'] == 2022].set_index('row_id')
    valid_ids = df_2022.index

    results = []

    # 对每个湖泊计算历史统计量
    # 注意：直接groupby可能较慢，但逻辑清晰。对于百万级数据建议向量化。
    grouped = full_df[full_df['row_id'].isin(valid_ids)].groupby('row_id')

    stats = grouped[['Cultivated_Land', 'Artificialsurface']].agg(['min', 'max', 'std', 'last'])

    for row_id, row in stats.iterrows():
        res_item = {'row_id': row_id}

        for col in ['Cultivated_Land', 'Artificialsurface']:
            init_val = row[(col, 'last')]  # 2022年值

            if init_val > 0:
                # 计算相对标准差作为波动依据
                std_val = row[(col, 'std')]
                rel_std = std_val / init_val if init_val != 0 else 0

                # 设定波动范围 (例如 +/- 1.5倍标准差)
                # 这是一个保守的物理约束，表示未来不太可能超出历史剧烈波动的范围
                buffer = rel_std * 1.5

                # 为了防止buffer过小（如历史无变化），设置最小阈值
                buffer = max(buffer, 0.1)

                lower = init_val * (1 - buffer)
                upper = init_val * (1 + buffer)
            else:
                lower = 0
                # 如果当前为0，允许微小增长至历史最大值，或默认极小值
                upper = max(row[(col, 'max')], 0.01)

            res_item[f'Constraint_LU_{col}_Lower'] = max(0, lower)
            res_item[f'Constraint_LU_{col}_Upper'] = upper

        results.append(res_item)

    print(f"土地利用约束计算完成，共处理 {len(results)} 个湖泊。")
    return pd.DataFrame(results)


def process_fishery_constraints(fao_path):
    """
    处理FAO渔业数据。
    逻辑：计算每个国家的历史渔业产量波动，设定未来渔业强度的浮动范围。
    """
    print(f"\n[Step 2] 正在处理渔业约束 (Source: {fao_path})...")

    if not os.path.exists(fao_path):
        print("[Error] FAO文件不存在。")
        return None

    fao_df = pd.read_csv(fao_path, encoding='utf-8')
    fao_df.rename(columns={'COUNTRY.UN_CODE': 'UN_Code', 'PERIOD': 'year', 'VALUE': 'production'}, inplace=True)

    # 筛选内陆水域 (Area Code 1-7)
    valid_areas = [1, 2, 3, 4, 5, 6, 7]
    fao_df = fao_df[fao_df['AREA.CODE'].isin(valid_areas)]

    # 汇总国家每年的总产量
    country_yearly = fao_df.groupby(['UN_Code', 'year'])['production'].sum().reset_index()

    results = []
    un_map = create_un_code_map()

    for un_code, group in country_yearly.groupby('UN_Code'):
        group = group.sort_values('year')

        # 计算年际变化率
        pct_changes = group['production'].pct_change().dropna()
        pct_changes = pct_changes[~pct_changes.isin([np.inf, -np.inf])]

        if len(pct_changes) > 1:
            # 允许的波动范围：均值 +/- 2.5倍标准差 (涵盖99%的情况)
            # 或者直接取历史最大跌幅和最大涨幅作为硬约束
            buffer_std = pct_changes.std() * 2.5
            lower_pct = pct_changes.min() - buffer_std
            upper_pct = pct_changes.max() + buffer_std
        else:
            # 默认宽泛范围
            lower_pct, upper_pct = -0.5, 0.5

        results.append({
            'UN_Code': un_code,
            'Country_Name': un_map.get(un_code, 'Unknown'),
            'Constraint_Fishery_Volatility_Lower_Pct': lower_pct,
            'Constraint_Fishery_Volatility_Upper_Pct': upper_pct
        })

    print(f"渔业约束计算完成，共处理 {len(results)} 个国家/地区。")
    return pd.DataFrame(results)


def fit_robust_technological_frontier(data, group_col, x_col, y_col, quantile, output_dir):
    """
    核心回归函数：拟合分位数回归曲线作为技术前沿。
    !!! 关键修改：移除迭代剔除逻辑，仅进行单次稳健拟合 !!!
    """
    # 清理列名以适应公式
    safe_y = re.sub(r'[^A-Za-z0-9_]+', '', y_col)
    safe_x = re.sub(r'[^A-Za-z0-9_]+', '', x_col)

    data_fit = data.rename(columns={y_col: safe_y, x_col: safe_x})
    formula = f"{safe_y} ~ {safe_x} + I({safe_x}**2)"

    predictions = pd.Series(index=data.index, dtype=float)

    # 按收入组别分别拟合 (High Income, Low Income...)
    for name, group in data_fit.groupby(group_col):
        # 基础清洗：去空值、去非正值(如果log需要，这里是多项式不需要严格正，但通常GDP>0)
        subset = group.dropna(subset=[safe_y, safe_x])
        subset = subset[(subset[safe_y] >= 0) & (subset[safe_x] > 0)]

        n_init = len(subset)
        if n_init < 20: continue  # 数据太少不拟合

        try:
            # === 单次拟合 ===
            # quantreg 使用 L1 Loss，本身对异常值有鲁棒性
            model = smf.quantreg(formula, subset).fit(q=quantile, max_iter=2500)

            # 预测
            original_group = data[data[group_col] == name]
            pred_input = original_group.rename(columns={y_col: safe_y, x_col: safe_x})
            predictions.loc[original_group.index] = model.predict(pred_input)

            # 绘图诊断
            plot_final_curve(
                subset.rename(columns={safe_y: y_col, safe_x: x_col}),
                name, x_col, y_col, model, model.prsquared, n_init, output_dir
            )

        except Exception as e:
            print(f"[Warning] 组别 '{name}' 拟合失败: {e}")

    return predictions


def generate_lake_constraints(feature_path, land_use_df, fishery_df, output_path):
    """
    主流程函数：整合所有数据，计算所有约束，输出最终文件。
    """
    print(f"\n[Step 3] 正在生成最终湖泊约束文件...")

    if not os.path.exists(feature_path):
        print("[Error] 特征文件不存在。")
        return

    # 1. 读取主数据
    df = pd.read_csv(feature_path)
    df['row_id'] = df.index

    # 2. 合并外部约束
    if land_use_df is not None:
        df = pd.merge(df, land_use_df, on='row_id', how='left')

    if fishery_df is not None:
        if 'UN_Code' in df.columns:
            df = pd.merge(df, fishery_df, on='UN_Code', how='left')
        else:
            print("[Warning] 特征文件中缺少 'UN_Code' 列，无法合并渔业约束。")

    # 3. 计算经济发展代理变量 (WGPC)
    # 假设列名是 'GDP under 8th lvl' 和 'pop. under 8th lvl'
    df['WGPC_8'] = df['GDP under 8th lvl'] / df['pop. under 8th lvl']
    df['WGPC_8'] = df['WGPC_8'].replace([np.inf, -np.inf], np.nan)

    # 4. 计算当前比例 (Proportions)
    # 废水处理比例
    ww_cols = ['Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge']
    df['Total_Discharge'] = df[ww_cols].sum(axis=1)
    mask_ww = df['Total_Discharge'] > 0

    df['Prop_A'] = np.nan
    df.loc[mask_ww, 'Prop_A'] = df.loc[mask_ww, 'Advanced_Waste_Discharge'] / df.loc[mask_ww, 'Total_Discharge']

    df['Prop_S_plus_A'] = np.nan
    df.loc[mask_ww, 'Prop_S_plus_A'] = (df.loc[mask_ww, 'Secondary_Waste_Discharge'] + df.loc[
        mask_ww, 'Advanced_Waste_Discharge']) / df.loc[mask_ww, 'Total_Discharge']

    df['Prop_P'] = np.nan
    df.loc[mask_ww, 'Prop_P'] = df.loc[mask_ww, 'Primary_Waste_Discharge'] / df.loc[mask_ww, 'Total_Discharge']

    # 道路铺设比例
    rse_cols = ['RSE_paved', 'RSE_gravel', 'RSE_other']
    df['Total_RSE'] = df[rse_cols].sum(axis=1)
    mask_rse = df['Total_RSE'] > 0

    df['Prop_Paved'] = np.nan
    df.loc[mask_rse, 'Prop_Paved'] = df.loc[mask_rse, 'RSE_paved'] / df.loc[mask_rse, 'Total_RSE']

    df['Prop_G_plus_P'] = np.nan
    df.loc[mask_rse, 'Prop_G_plus_P'] = (df.loc[mask_rse, 'RSE_gravel'] + df.loc[mask_rse, 'RSE_paved']) / df.loc[
        mask_rse, 'Total_RSE']

    df['Prop_Other'] = np.nan
    df.loc[mask_rse, 'Prop_Other'] = df.loc[mask_rse, 'RSE_other'] / df.loc[mask_rse, 'Total_RSE']

    # 5. 拟合技术前沿 (回归)
    plot_dir = os.path.join(output_path, "diagnostic_plots")

    # 目标列表: (新列名, Y变量, 分位数)
    # 逻辑:
    #   高级处理 Prop_A -> 98分位数 (尽可能高)
    #   未管理垃圾 Mismanaged -> 10分位数 (尽可能低，所以是下限)
    targets = [
        ('Target_Prop_A', 'Prop_A', 0.98),
        ('Target_Prop_S_plus_A', 'Prop_S_plus_A', 0.95),
        ('Lower_Bound_Prop_P', 'Prop_P', 0.05),  # 初级处理应尽可能低(转化为高级)
        ('Target_Prop_Paved', 'Prop_Paved', 0.95),
        ('Target_Prop_G_plus_P', 'Prop_G_plus_P', 0.95),
        ('Lower_Bound_Prop_Other', 'Prop_Other', 0.05),
        ('Lower_Bound_Mismanaged', 'Mismanaged', 0.10)  # 垃圾最少能减到多少
    ]

    group_col = 'income' if 'income' in df.columns else None
    if group_col is None:
        df['all_group'] = 'Global'
        group_col = 'all_group'

    print("正在进行分位数回归拟合...")
    for target_col, y_col, q in targets:
        if y_col in df.columns:
            df[target_col] = fit_robust_technological_frontier(df, group_col, 'WGPC_8', y_col, q, plot_dir)

    # 6. 生成最终约束值 (Absolute Values)
    print("正在计算绝对约束值...")

    # 废水: 上限 = 目标比例 * 总量
    # 如果当前已经高于目标(极少情况)，则不限制(inf)，或者保持现状
    # 优化逻辑通常是: Variable <= Constraint.
    # 对于想要提高的属性(如高级处理)，优化器通常是自由选择，但受限于技术前沿。
    # 这里我们定义的是 "Upper Bound" (最大可行能力)
    df['Constraint_WW_Advanced_Upper'] = np.where(df['Prop_A'] < df['Target_Prop_A'],
                                                  df['Target_Prop_A'] * df['Total_Discharge'],
                                                  np.inf)  # 如果已经很高，不做硬性上限限制，由总量控制

    df['Constraint_WW_Sec_Plus_Adv_Upper'] = np.where(df['Prop_S_plus_A'] < df['Target_Prop_S_plus_A'],
                                                      df['Target_Prop_S_plus_A'] * df['Total_Discharge'],
                                                      np.inf)

    # 初级处理: Lower Bound (最小残留量)
    df['Constraint_WW_Primary_Lower'] = np.where(df['Primary_Waste_Discharge'] == 0, 0,
                                                 df['Lower_Bound_Prop_P'] * df['Total_Discharge'])

    # 道路
    df['Constraint_RSE_Paved_Upper'] = np.where(df['Prop_Paved'] < df['Target_Prop_Paved'],
                                                df['Target_Prop_Paved'] * df['Total_RSE'], np.inf)

    df['Constraint_RSE_Gravel_Plus_Paved_Upper'] = np.where(df['Prop_G_plus_P'] < df['Target_Prop_G_plus_P'],
                                                            df['Target_Prop_G_plus_P'] * df['Total_RSE'], np.inf)

    df['Constraint_RSE_Other_Lower'] = np.where(df['RSE_other'] == 0, 0,
                                                df['Lower_Bound_Prop_Other'] * df['Total_RSE'])

    # 垃圾 (Lower Bound)
    df['Constraint_Mismanaged_Lower'] = np.where(df['Mismanaged'] == 0, 0, df['Lower_Bound_Mismanaged'])

    # 渔业 (基于波动率)
    if 'Constraint_Fishery_Volatility_Lower_Pct' in df.columns:
        df['Constraint_Fishery_GDP_Lower'] = df['fish_gdp_sqkm'] * (1 + df['Constraint_Fishery_Volatility_Lower_Pct'])
        df['Constraint_Fishery_GDP_Upper'] = df['fish_gdp_sqkm'] * (1 + df['Constraint_Fishery_Volatility_Upper_Pct'])
        # 确保不小于0
        df['Constraint_Fishery_GDP_Lower'] = df['Constraint_Fishery_GDP_Lower'].clip(lower=0)

    # 7. 清理与保存
    # 选取所有Constraint列
    constraint_cols = [c for c in df.columns if 'Constraint_' in c]

    # 填补空值为默认逻辑
    # 对于Upper，若为空则设为inf(无限制)或当前值? 设为当前值比较安全
    # 对于Lower，若为空则设为0
    for c in constraint_cols:
        if 'Lower' in c:
            df[c] = df[c].fillna(0)
        elif 'Upper' in c:
            # 如果没有算出前沿，说明数据不足，暂不设限(使用极大值)或保持现状
            # 为了优化器安全，可以使用当前值的2倍作为松弛边界，或者inf
            df[c] = df[c].fillna(np.inf)

    output_file = os.path.join(output_path, "final_lake_constraints.csv")

    # 保存包含Row_ID和所有约束列的文件
    cols_to_save = ['row_id', 'UN_Code'] + constraint_cols
    # 确保列存在
    cols_to_save = [c for c in cols_to_save if c in df.columns]

    df[cols_to_save].to_csv(output_file, index=False)
    print(f"[Success] 最终约束文件已保存至: {output_file}")


# ==========================================
# 3. 主执行入口
# ==========================================

if __name__ == "__main__":
    # --- 配置路径 ---
    LAND_USE_PATH = r"E:\lakemicroplastic\draw\全球预测\timestep"
    FEATURE_FILE_PATH = r"E:\lake-MP-W\data\feature\feature_2022.csv"
    FAO_FILE_PATH = r"E:\lake-MP-W\dataset\FAO\GlobalProduction_2024.1.0\Global_production_quantity.csv"
    OUTPUT_DIR = r"E:\lake-MP-W\data\yueshu"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=== 开始生成优化约束 ===")

    # 1. 计算土地利用约束
    lu_constraints = process_land_use_constraints(LAND_USE_PATH)

    # 2. 计算渔业约束
    fishery_constraints = process_fishery_constraints(FAO_FILE_PATH)
    if fishery_constraints is not None:
        # 保存一份中间结果备份
        fishery_constraints.to_csv(os.path.join(OUTPUT_DIR, "fishery_volatility_constraints.csv"), index=False)

    # 3. 生成最终约束文件
    generate_lake_constraints(FEATURE_FILE_PATH, lu_constraints, fishery_constraints, OUTPUT_DIR)

    print("\n=== 所有任务完成 ===")
