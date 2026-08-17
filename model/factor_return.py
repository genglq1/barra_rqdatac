# -*- coding: utf-8 -*-
"""
因子收益率:带约束的 WLS 截面回归
================================
对每个交易日,用"国家 + 行业(哑变量) + 风格因子"做加权截面回归,得到当日各因子收益率 f_ret。

核心矩阵:
    V = diag(√流通市值 / Σ√流通市值)        市值开方加权
    R = 行业约束矩阵(消除行业共线性,末行业=其他行业线性组合)
    W = R · (R'X'VXR')⁻¹ · R'X'V            因子收益率加权矩阵
    f = W · r                                当日因子收益率

每日回归样本构建(三层防御,Barra 规范):
    ① 估计域:有行业归属 + 当日有流通市值 + 当日有收益(停牌股剔除)的股票;
    ② 缺失暴露填 0:风格因子已截面标准化,0 = 截面均值 = 中性暴露,
       不再要求 10 因子全覆盖(旧版 dropna 把样本从 ~5100 只缩到 ~1400 只,
       小行业被整体剔空导致矩阵奇异/垃圾值);
    ③ 当日全空的风格列(如 momentum 预热期)不进入当日回归;
       行业哑变量与约束 R 基于当日最终样本自洽计算(空行业天然不出现);
    ④ 求解带条件数防御(见 _weighted_regression_matrix),杜绝静默垃圾值。

主流程 get_factor_return 按日循环,委托三个 builder 子函数:
    _build_constraint_R(industry_cap_weights)        -> 约束矩阵 R
    _build_exposure_matrix(由主流程内联合构建)        -> 因子暴露矩阵 X
    _weighted_regression_matrix(X, cap_weights, R)   -> 加权矩阵 W

输出: data_store/model/{version}/f_ret.csv(index=日期, 列=[country, 行业..., 风格...])
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_model_dir, get_path
from data import io as data_io
from data.client import init_rqdatac, to_rqcode

# 接近奇异的判定阈值:条件数超过此值时 inv 结果不可信(float64 有效位约 16 位,
# 条件数 1e10 意味着结果只剩 ~6 位有效数字,再大即垃圾值区间)
COND_THRESHOLD = 1e10


def _sqrt_cap_weights(cap_series):
    """
    市值开方归一化权重:√cap / Σ√cap。
    用于截面回归的市值加权(Barra 规范),消除全市场/样本内两处重复计算。
    """
    sqrt_cap = np.sqrt(cap_series)
    return sqrt_cap / sqrt_cap.sum()


def _build_constraint_R(industry_cap_weights, n_style):
    """
    构造行业约束矩阵 R(K×(K-1)),消除行业虚拟变量的完全共线性。

    Barra 标准约束:行业因子收益率的市值加权和为 0,即
        f_last = -Σ_{i<last} (w_i / w_last) · f_i
    故 R 的最后一行(被删行业行)系数为 -w_i / w_last(旧实现误除以 Σw ≈ 1,
    等价于给最后一个行业赋权重 1,导致 country/行业 f_ret 口径与 Barra 不一致)。

    参数:
        industry_cap_weights: Series, index=行业代码, value=行业市值权重
                              (须与当日回归样本一致,空行业不应出现)
        n_style: 风格因子个数(用于确定 K 的总维度)
    返回:
        R 矩阵(K×(K-1)), K = 行业数 + 1(country) + n_style
    """
    k = len(industry_cap_weights) + 1 + n_style
    location = len(industry_cap_weights)
    R = np.delete(np.diag(np.ones(k)), location, axis=1)
    w_last = industry_cap_weights.iloc[-1]
    if w_last is None or np.isnan(w_last) or w_last <= 0:
        # 防护:末行业权重为 0 时退化为等权约束(-1),避免除零
        R[location, 1:location] = -1.0
    else:
        R[location, 1:location] = -(industry_cap_weights.iloc[:-1] / w_last).values
    return R


def _weighted_regression_matrix(X, cap_weights, R):
    """
    构建因子收益率加权矩阵 W = R · (R'X'VXR')⁻¹ · R'X'V。
    V = diag(市值开方归一化权重)。

    性能优化:V 是对角阵,用"逐行缩放 X"替代构造 n×n 全矩阵(避免 O(n²) 内存与运算),
    数学上 X'VX = Xᵀ(X⊙w),R'X'V = Rᵀ(X⊙w)ᵀ,结果与全矩阵实现完全等价。

    数值防御:裸 inv 在矩阵接近奇异时不报错而返回错误值(曾导致 f_ret 出现
    |f|>2700 的静默垃圾值)。此处先检查条件数:非有限(含 NaN/inf)返回 None;
    超过阈值或 inv 抛 LinAlgError 时改用 pinv 最小范数解并告警。

    参数:
        X: 因子暴露矩阵(index=股票, columns=[country, 行业..., 风格...])
        cap_weights: 当日回归样本的市值开方归一化权重(DataFrame, 含 weight 列, index 与 X 一致)
        R: 约束矩阵(K×(K-1))
    返回:
        W(DataFrame, index=X.columns, columns=X.index),或 None(矩阵病态无法求解)
    """
    w = cap_weights["weight"].reindex(X.index).fillna(0).values   # (n,)
    Xa = X.values.astype(float)               # 转 float(行业哑变量 bool/混合 dtype 会致矩阵 object,inv 失败)
    Xv = Xa * w[:, None]                       # V@X(逐行乘 w),等价 diag(w)@X
    XtVX = Xa.T @ Xv                           # X'VX (K×K)
    M = R.T @ XtVX @ R                         # (R'X'VXR') 待逆矩阵
    cond = np.linalg.cond(M)
    if not np.isfinite(cond):
        print("⚠️  矩阵含 NaN/inf(条件数非有限),无法求解")
        return None
    if cond > COND_THRESHOLD:
        print(f"⚠️  条件数 {cond:.2e} > {COND_THRESHOLD:.0e},改用 pinv 最小范数解")
        inv = np.linalg.pinv(M)
    else:
        try:
            inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print(f"⚠️  inv 奇异(条件数 {cond:.2e}),改用 pinv 最小范数解")
            inv = np.linalg.pinv(M)
    W = R @ inv @ (R.T @ Xv.T)                 # R·inv·R'X'V  (K×n)
    return pd.DataFrame(W, index=X.columns, columns=X.index)


def get_factor_return(start_date, end_date, version="cne5"):
    """
    计算 CNE-5(或 CNE-6)因子收益率序列,落盘 f_ret.csv。

    每日回归样本 = 估计域内股票(有行业 + 当日有市值 + 当日有收益),
    缺失风格暴露按 0(中性)填充,当日全空的风格列与空行业不进入当日回归,
    行业约束 R 与哑变量基于当日最终样本自洽计算(见模块 docstring 三层防御)。

    参数:
        start_date, end_date: 回归日期区间 "YYYY-MM-DD"
        version: "cne5" / "cne6"
    返回:
        f_ret DataFrame(index=日期, 列=country/行业/风格因子)
    """
    # ---- 读取输入数据 ----
    ret_data = data_io.load_base("stock_ret.csv")
    stock_size_cir = data_io.load_base("stock_size_cir.csv")

    # 行业归属(当前快照,行业变化缓慢适用所有交易日);Wind 代码转 rqdatac 风格以对齐
    industry_map = pd.read_csv(os.path.join(get_path("base"), "industry_l1.csv"))
    industry_map["stock_code"] = industry_map["stock_code"].apply(to_rqcode)
    industry_map = industry_map.set_index("stock_code")[["industry_code"]]

    cne_path = os.path.join(get_model_dir(version), "cne_5.csv")
    cne_df = pd.read_csv(cne_path)
    cne_df["trade_date"] = pd.to_datetime(cne_df["trade_date"])

    # 交易日(用本地 trade_cal.csv,避免联网;rqdatac get_trading_dates 易受 Quota 限制)
    trade_cal = data_io.load_base("trade_cal.csv")
    _start, _end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    trade_days = trade_cal.index[(trade_cal.index >= _start) & (trade_cal.index <= _end)]

    # 风格因子列名(cne_5.csv 中除 trade_date/code 外的列)
    style_columns = [c for c in cne_df.columns if c not in ("trade_date", "code")]

    f_ret_list = []
    for trade_date in tqdm(trade_days, desc=f"{version} 因子收益率"):
        fd = trade_date.strftime("%Y-%m-%d")

        # 当日因子暴露(长表取该日)
        exposure_today = cne_df.loc[
            cne_df["trade_date"] == trade_date, ["code"] + style_columns]
        if exposure_today.empty:
            continue
        # 当日流通市值(权重来源;缺失则该日无法加权回归)
        try:
            cap_row = stock_size_cir.loc[fd]
        except KeyError:
            continue
        # 当日收益率(回归因变量;停牌股收益缺失,参与会把 NaN 传染给整日结果)
        try:
            day_ret = ret_data.loc[fd]
        except KeyError:
            continue
        day_ret = day_ret.dropna()

        # ---- ① 估计域:行业内壳 + 当日有市值 + 当日有收益 ----
        sample = exposure_today.merge(
            industry_map, left_on="code", right_index=True, how="inner"
        ).set_index("code")
        sample = sample[sample.index.isin(day_ret.index)]
        cap_sample = cap_row.reindex(sample.index)
        keep = cap_sample.notna() & (cap_sample > 0)
        sample, cap_sample = sample[keep], cap_sample[keep]
        if sample.empty:
            continue

        # ---- ② 风格暴露:当日全空的列剔除,其余缺失填 0(标准化后 0=中性)----
        avail_styles = [c for c in style_columns if sample[c].notna().any()]
        dropped_styles = sorted(set(style_columns) - set(avail_styles))
        if dropped_styles:
            print(f"{fd}: 风格列 {dropped_styles} 当日全空,剔除后回归")
        styles = sample[avail_styles].fillna(0.0)

        # ---- ③ 行业哑变量:基于当日最终样本(空行业天然不出现)----
        dummies = pd.get_dummies(sample["industry_code"]).astype(float)
        n_factors = 1 + dummies.shape[1] + len(avail_styles)
        if len(sample) <= n_factors:
            print(f"{fd}: 回归样本 {len(sample)} 只 ≤ 因子数 {n_factors},跳过")
            continue
        X = pd.concat(
            [pd.DataFrame(1.0, index=sample.index, columns=["country"]),
             dummies, styles],
            axis=1,
        )

        # ---- 权重与约束 R(与 X 同一样本,自洽)----
        weights = _sqrt_cap_weights(cap_sample)
        industry_cap_weights = dummies.T @ weights
        R = _build_constraint_R(industry_cap_weights, len(avail_styles))

        # ---- 加权矩阵 W(带条件数防御)----
        W = _weighted_regression_matrix(X, weights.to_frame("weight"), R)
        if W is None:
            print(f"{fd}: 矩阵病态无法求解,跳过")
            continue

        # ---- 当日因子收益率 f = W · r ----
        r = day_ret.reindex(W.columns)
        f = W.dot(r).to_frame(name=fd).T
        f_ret_list.append(f)

    f_ret = pd.concat(f_ret_list) if f_ret_list else pd.DataFrame()

    out_dir = get_model_dir(version)
    os.makedirs(out_dir, exist_ok=True)
    f_ret.to_csv(os.path.join(out_dir, "f_ret.csv"))
    print(f"f_ret.csv: 更新完成 ({len(f_ret)} 个交易日)")
    return f_ret
