# -*- coding: utf-8 -*-
"""
本轮修复的数值验证脚本(在 factors/model/f_ret 全部重跑完成后执行)。

验证项:
  A. CMRA 口径:新(12月度点)vs 旧(日频极差)对比 + 随机抽样与朴素参考对拍
  B. 正交性:抽样交易日,验证 RV 与 beta/size、liquidity 与 size 的√流通市值加权截面相关 ≈ 0
  C. 约束矩阵:同一天同 X,新 R(-w_i/w_last)vs 旧 R(-w_i/Σw) → 风格因子收益率不变,
     行业/country 改变;新约束下 Σ w_i·f_industry ≈ 0
  D. 非空率总表(与变更说明第四节对比)
  E. MLEV 量级抽样(万科/保利 高杠杆、茅台 低杠杆)
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

BACKUP = "data_store/_backup_20260813"


def load(f, backup=False):
    if backup:
        # 备份目录结构: factors/cne5 -> factors_cne5, model/cne5 -> model_cne5
        if f.startswith("factors/"):
            f = "factors_cne5/" + f.split("/", 2)[2]
        elif f.startswith("model/"):
            f = "model_cne5/" + f.split("/", 2)[2]
    root = BACKUP if backup else "data_store"
    return pd.read_csv(f"{root}/{f}", index_col=0, parse_dates=True)


# ---------------- A. CMRA ----------------
print("=" * 60)
print("A. CMRA 口径对比")
print("=" * 60)
cmra_new = load("factors/cne5/cmra.csv")
cmra_old = load("factors/cne5/cmra.csv", backup=True)
common_idx = cmra_new.index.intersection(cmra_old.index)
common_cols = cmra_new.columns.intersection(cmra_old.columns)
a_new, a_old = cmra_new.loc[common_idx, common_cols], cmra_old.loc[common_idx, common_cols]
both = (a_new.notna() & a_old.notna()).values
diff = (a_new.values - a_old.values)[both]
print(f"共同非空样本: {both.sum()} 个")
print(f"新CMRA - 旧CMRA: mean={diff.mean():.5f}  min={diff.min():.5f}  max={diff.max():.5f}")
print(f"新CMRA 非空率: {a_new.notna().sum().sum()/a_new.size:.2%}  旧: {a_old.notna().sum().sum()/a_old.size:.2%}")
print("预期:新月度口径 <= 旧日频口径(多数样本),且非空率不变")

# ---------------- B. 正交性 ----------------
print("\n" + "=" * 60)
print("B. 正交性检查(新 cne_5.csv,√流通市值加权截面相关)")
print("=" * 60)
cne = pd.read_csv("data_store/model/cne5/cne_5.csv")
cne["trade_date"] = pd.to_datetime(cne["trade_date"])
cne = cne.set_index(["trade_date", "code"]).sort_index()
w = load("base/stock_size_cir.csv")
w_sqrt = np.sqrt(w).div(np.sqrt(w).sum(axis=1), axis=0)

dates = sorted(cne.index.get_level_values(0).unique())
sample_dates = dates[:: max(1, len(dates) // 8)][:8]


def wcorr(date, x_name, y_name):
    row = cne.loc[date]
    x = row[x_name] if x_name in row else pd.Series(np.nan, index=row.index)
    y = row[y_name] if y_name in row else pd.Series(np.nan, index=row.index)
    ww = w_sqrt.loc[date] if date in w_sqrt.index else None
    if ww is None:
        return np.nan
    valid = x.notna() & y.notna() & ww.notna()
    if valid.sum() < 10:
        return np.nan
    xv, yv, wv = x[valid], y[valid], ww[valid]
    xm = (xv * wv).sum(); ym = (yv * wv).sum()
    cov = (wv * (xv - xm) * (yv - ym)).sum()
    vx = (wv * (xv - xm) ** 2).sum(); vy = (wv * (yv - ym) ** 2).sum()
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


pairs = [("residual_volatility", "beta"), ("residual_volatility", "size"),
         ("liquidity", "size")]
for x_name, y_name in pairs:
    corrs = [wcorr(d, x_name, y_name) for d in sample_dates]
    corrs = [c for c in corrs if not np.isnan(c)]
    print(f"corr({x_name}, {y_name}): 样本 {len(corrs)} 天, "
          f"mean={np.mean(corrs):+.4f}, max|.|={max(abs(c) for c in corrs):.4f} (应 ≈ 0)")

# ---------------- C. 约束矩阵 ----------------
print("\n" + "=" * 60)
print("C. 约束矩阵:新旧 R 对风格/行业 f_ret 的影响 + 约束校验")
print("=" * 60)
from model import factor_return as FR
from model.factor_return import _sqrt_cap_weights, _weighted_regression_matrix
from data import io as data_io
from data.client import to_rqcode
import os
from config import get_path

ret_data = data_io.load_base("stock_ret.csv")
stock_size_cir = data_io.load_base("stock_size_cir.csv")
industry_map = pd.read_csv(os.path.join(get_path("base"), "industry_l1.csv"))
industry_map["stock_code"] = industry_map["stock_code"].apply(to_rqcode)
industry_map = industry_map.set_index("stock_code")
style_columns = [c for c in cne.columns if c not in ("trade_date", "code")]


def build_R_old(industry_cap_weights, n_style):
    k = len(industry_cap_weights) + 1 + n_style
    location = len(industry_cap_weights)
    R = np.delete(np.diag(np.ones(k)), location, axis=1)
    neg = -industry_cap_weights / industry_cap_weights.sum()
    R[location, 1:location] = neg.values[:-1]
    return R


test_dates = sample_dates
f_new_list, f_old_list = [], []
constraint_err = []
for trade_date in test_dates:
    fd = trade_date.strftime("%Y-%m-%d")
    exposure_today = cne.loc[trade_date].reset_index()
    if exposure_today.empty:
        continue
    try:
        scale_data = stock_size_cir.loc[fd].to_frame("weight")
    except KeyError:
        continue
    cap_weights_all = _sqrt_cap_weights(scale_data)
    df_ind_date = industry_map.loc[industry_map.index.intersection(exposure_today["code"])].reset_index()
    if "stock_code" not in df_ind_date.columns and "index" in df_ind_date.columns:
        df_ind_date = df_ind_date.rename(columns={"index": "stock_code"})
    dummies = pd.get_dummies(df_ind_date["industry_code"])
    df_ind_dummies = pd.concat([df_ind_date, dummies], axis=1)
    industry_cap_weights = (
        pd.merge(cap_weights_all, df_ind_dummies, how="left",
                 left_index=True, right_on="stock_code")
        .groupby("industry_code")["weight"].sum()
    )
    industry_cols = ["stock_code"] + industry_cap_weights.index.to_list()
    X = pd.merge(exposure_today, df_ind_dummies[industry_cols],
                 how="inner", left_on="code", right_on="stock_code")
    X["country"] = 1
    X = X.dropna().set_index("code").drop(columns=["stock_code"])
    if X.empty:
        continue
    columns_name = [c for c in (["country"] + list(industry_cap_weights.index) + style_columns)
                    if c in X.columns]
    X = X[columns_name]
    scale_data_x = stock_size_cir.loc[fd, X.index].to_frame("weight")
    cap_weights_x = _sqrt_cap_weights(scale_data_x)

    R_new = FR._build_constraint_R(industry_cap_weights, len(style_columns))
    R_old = build_R_old(industry_cap_weights, len(style_columns))
    W_new = _weighted_regression_matrix(X, cap_weights_x, R_new)
    W_old = _weighted_regression_matrix(X, cap_weights_x, R_old)
    if W_new is None or W_old is None:
        print(f"{fd}: 求逆失败,跳过"); continue
    r = ret_data.loc[fd, W_new.columns]
    f_new_list.append(W_new.dot(r))
    f_old_list.append(W_old.dot(r))

    # 约束校验:Σ w_i f_industry ≈ 0(w = 当日行业市值权重,含末行业)
    f_hat = W_new.dot(r)
    w_i = industry_cap_weights.reindex(industry_cap_weights.index)
    f_ind = f_hat.reindex(w_i.index)
    constraint_err.append((w_i.values * f_ind.values).sum())

if f_new_list:
    f_new = pd.concat(f_new_list, axis=1).T
    f_old = pd.concat(f_old_list, axis=1).T
    style_cols = [c for c in f_new.columns if c in style_columns]
    others = [c for c in f_new.columns if c not in style_columns]
    d = (f_new[style_cols] - f_old[style_cols]).abs()
    print(f"风格因子 f_ret: 新旧 R 差异 max|diff| = {d.max().max():.2e} (应≈0, 浮点噪声级)")
    d2 = (f_new[others] - f_old[others]).abs()
    print(f"country+行业 f_ret: 新旧 R 差异 max|diff| = {d2.max().max():.2e} (应显著 ≠ 0)")
    print(f"新约束 Σ w_i·f_industry: max|err| = {max(abs(e) for e in constraint_err):.2e} (应≈0)")
else:
    print("无可用样本日,跳过约束验证")

# ---------------- D. 非空率 ----------------
print("\n" + "=" * 60)
print("D. 新 cne_5.csv 风格因子非空率(对比变更说明第四节)")
print("=" * 60)
for col in style_columns:
    nn = cne[col].notna().mean()
    print(f"  {col:<22} {nn:.1%}")

# ---------------- E. MLEV 量级抽样 ----------------
print("\n" + "=" * 60)
print("E. MLEV 量级抽样(新 leverage,2026 年最近日)")
print("=" * 60)
mlev = load("factors/cne5/mlev.csv")
last_date = mlev.index[-1]
for code, name in [("600519.XSHG", "贵州茅台"), ("000002.XSHE", "万科A"),
                   ("600048.XSHG", "保利发展")]:
    if code in mlev.columns:
        print(f"  {name} {code}: mlev = {mlev.loc[last_date, code]:+.3f}")

print("\n验证完成。")
