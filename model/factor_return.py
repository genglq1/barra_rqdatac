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

主流程 get_factor_return 按日循环,委托三个 builder 子函数:
    _build_constraint_R(industry_cap_weights)        -> 约束矩阵 R
    _build_exposure_matrix(exposure_today, ...)      -> 因子暴露矩阵 X
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
    构造因子收益率加权矩阵 W = R · (R'X'VXR')⁻¹ · R'X'V。
    V = diag(市值开方归一化权重)。

    性能优化:V 是对角阵,用"逐行缩放 X"替代构造 n×n 全矩阵(避免 O(n²) 内存与运算),
    数学上 X'VX = Xᵀ(X⊙w),R'X'V = Rᵀ(X⊙w)ᵀ,结果与全矩阵实现完全等价。

    参数:
        X: 因子暴露矩阵(index=股票, columns=[country, 行业..., 风格...])
        cap_weights: 当日回归样本的市值开方归一化权重(DataFrame, 含 weight 列, index 与 X 一致)
        R: 约束矩阵(K×(K-1))
    返回:
        W(DataFrame, index=X.columns, columns=X.index),或 None(求逆失败)
    """
    w = cap_weights["weight"].reindex(X.index).fillna(0).values   # (n,)
    Xa = X.values.astype(float)               # 转 float(行业哑变量 bool/混合 dtype 会致矩阵 object,inv 失败)
    Xv = Xa * w[:, None]                       # V@X(逐行乘 w),等价 diag(w)@X
    XtVX = Xa.T @ Xv                           # X'VX (K×K)
    try:
        inv = np.linalg.inv(R.T @ XtVX @ R)    # (R'X'VXR')⁻¹
    except np.linalg.LinAlgError:
        return None
    W = R @ inv @ (R.T @ Xv.T)                 # R·inv·R'X'V  (K×n)
    return pd.DataFrame(W, index=X.columns, columns=X.index).round(6)


def get_factor_return(start_date, end_date, version="cne5"):
    """
    计算 CNE-5(或 CNE-6)因子收益率序列,落盘 f_ret.csv。

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
    sw_ind = pd.read_csv(os.path.join(get_path("base"), "sw_l1.csv"))
    sw_ind["stock_code"] = sw_ind["stock_code"].apply(to_rqcode)
    sw_ind = sw_ind.set_index("stock_code")

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
        formatted_date = trade_date.strftime("%Y-%m-%d")

        # 当日因子暴露(长表取该日,保留 code 列便于 merge)
        exposure_today = cne_df[cne_df["trade_date"] == trade_date].copy()
        if exposure_today.empty:
            continue

        # 当日全市场市值开方权重(用于行业权重 con)
        try:
            scale_data = stock_size_cir.loc[formatted_date].to_frame("weight")
        except KeyError:
            continue
        cap_weights_all = _sqrt_cap_weights(scale_data)

        # 行业归属(当日有因子暴露的股票);reset_index 让 stock_code 成为列
        df_ind_date = sw_ind.loc[sw_ind.index.intersection(exposure_today["code"])].reset_index()
        if "stock_code" not in df_ind_date.columns and "index" in df_ind_date.columns:
            df_ind_date = df_ind_date.rename(columns={"index": "stock_code"})
        dummies = pd.get_dummies(df_ind_date["industry_code"])
        df_ind_dummies = pd.concat([df_ind_date, dummies], axis=1)

        # 各行业市值权重(归一化)
        industry_cap_weights = (
            pd.merge(cap_weights_all, df_ind_dummies, how="left",
                     left_index=True, right_on="stock_code")
            .groupby("industry_code")["weight"].sum()
        )

        # ---- 约束矩阵 R(消除行业共线性)----
        R = _build_constraint_R(industry_cap_weights, len(style_columns))

        # ---- 因子暴露矩阵 X ----
        industry_cols = ["stock_code"] + industry_cap_weights.index.to_list()
        X = pd.merge(exposure_today, df_ind_dummies[industry_cols],
                     how="inner", left_on="code", right_on="stock_code")
        X["country"] = 1
        X = X.dropna().set_index("code").drop(columns=["stock_code"])
        if X.empty:
            continue

        # 列顺序固定:country + 行业 + 风格因子(仅保留实际存在的列)
        columns_name = [c for c in (["country"] + list(industry_cap_weights.index) + style_columns)
                        if c in X.columns]
        X = X[columns_name]

        # ---- 加权矩阵 W ----
        scale_data_x = stock_size_cir.loc[formatted_date, X.index].to_frame("weight")
        cap_weights_x = _sqrt_cap_weights(scale_data_x)
        W = _weighted_regression_matrix(X, cap_weights_x, R)
        if W is None:
            print(f"{formatted_date}: 矩阵求逆失败,跳过")
            continue

        # ---- 当日因子收益率 f = W · r ----
        r = ret_data.loc[formatted_date, W.columns]
        f = W.dot(r).to_frame(name=formatted_date).T
        f_ret_list.append(f)

    f_ret = pd.concat(f_ret_list) if f_ret_list else pd.DataFrame()

    out_dir = get_model_dir(version)
    os.makedirs(out_dir, exist_ok=True)
    f_ret.to_csv(os.path.join(out_dir, "f_ret.csv"))
    print(f"f_ret.csv: 更新完成 ({len(f_ret)} 个交易日)")
    return f_ret
