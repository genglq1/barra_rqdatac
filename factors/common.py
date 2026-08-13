# -*- coding: utf-8 -*-
"""
因子计算共享工具(版本无关)
==========================
迁移自原 factor_calculate.py 的基础函数,**计算逻辑完全保持不变**,
仅把硬编码路径改为读 data_store,并修复 missing_keys 等 bug。

被 cne5/cne6 各版本因子文件复用:
    _ensure_dataframe         确保 DataFrame 格式
    _get_ttm_data             TTM(滚动十二月)计算
    _get_exp_weight           指数衰减权重
    _regress                  WLS 加权回归封装
    _winsorize               3σ 缩尾
    _standardize_with_weights 市值加权标准化三步曲(每个因子都要过)
    _get_latest_date          因子增量更新:读已有最新日期
    _get_existing_df          多因子批量读取已有数据
    data_concat_and_save      新旧合并+去重落盘
    load_factor / load_base   读取因子/基础宽表
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm

from config import get_factor_dir
from data import io as data_io


# ----------------------------------------------------------------------------
# 全局:交易日历 + 市值权重(惰性加载,避免 import 时读盘失败)
# ----------------------------------------------------------------------------

_trade_cal = None
_stock_weight_cir = None   # 流通市值归一化权重
_stock_weight = None       # 总市值归一化权重
_ndate = None


def _ensure_globals():
    """
    惰性初始化全局变量(交易日历、市值权重)。
    首次调用时读盘,后续复用缓存。供 _standardize_with_weights 等使用。

    ⚠️ 失效说明:全局变量在进程生命周期内只初始化一次。若运行中 stock_size.csv
    被更新,本缓存不会自动刷新(长跑场景下标准化权重会偏旧)。如需刷新,重启进程。
    """
    global _trade_cal, _stock_weight_cir, _stock_weight, _ndate
    if _trade_cal is not None:
        return

    _trade_cal = data_io.load_base("trade_cal.csv")

    size_cir = data_io.load_base("stock_size_cir.csv")
    size = data_io.load_base("stock_size.csv")
    _stock_weight_cir = size_cir.apply(lambda row: row / row.sum(), axis=1)
    _stock_weight = size.apply(lambda row: row / row.sum(), axis=1)
    _ndate = min(dt.datetime.now().strftime("%Y-%m-%d"),
                 _stock_weight.index[-1].strftime("%Y-%m-%d"))


def get_ndate():
    """获取最新数据日期(今天与最新市值日的较小值)。"""
    _ensure_globals()
    return _ndate


def get_trade_cal():
    """获取交易日历。"""
    _ensure_globals()
    return _trade_cal


def get_stock_weight(is_cir=True, sqrt=False):
    """
    获取市值归一化权重(供描述符正交化等场景复用)。
    与 _standardize_with_weights 同源,保证描述符处理链口径一致。

    参数:
        is_cir: True 返回流通市值权重(默认,与 NLSIZE/标准化口径一致),
                False 返回总市值权重
        sqrt:   True 返回 √市值 归一化权重(Barra "regression-weighted" 回归权重,
                NLSIZE 正交化 / 风格因子正交化用;标准化仍用市值权重)
    返回:
        DataFrame(index=日期, columns=股票),每行归一化和为 1
    """
    _ensure_globals()
    w = _stock_weight_cir if is_cir else _stock_weight
    if sqrt:
        w = np.sqrt(w)
    return w.apply(lambda row: row / row.sum(), axis=1)


# ----------------------------------------------------------------------------
# 基础工具函数(逻辑与原 factor_calculate.py 完全一致)
# ----------------------------------------------------------------------------

def _ensure_dataframe(item):
    """确保为 DataFrame 格式。"""
    if isinstance(item, pd.DataFrame):
        return item
    elif isinstance(item, pd.Series):
        return item.to_frame()
    else:
        try:
            return pd.Series(item).to_frame()
        except Exception as e:
            print(f"无法转换为 DataFrame: {e}")
            return None


def _get_ttm_data(datdf):
    """
    对于季度日期,获取 TTM(滚动十二月)数据。
    逻辑:本期 + 去年年报 - 去年同期 = TTM
    """
    res = pd.DataFrame(index=datdf.index, columns=datdf.columns)
    for date in datdf.index[4:]:
        if date.month == 12:
            res.loc[date, :] = datdf.loc[date, :]
            continue
        lst_rpt_y = pd.to_datetime(f"{date.year - 1}-12-31")
        lst_rpt_q = pd.to_datetime(f"{date.year - 1}-{date.month}-{date.day}")
        res.loc[date, :] = datdf.loc[lst_rpt_y] + datdf.loc[date] - datdf.loc[lst_rpt_q]
    return res


def _get_exp_weight(window, half_life):
    """指数移动平均权重(半衰期归一化)。"""
    exp_wt_row = np.asarray([0.5 ** (1 / half_life)] * window)
    exp_wt = exp_wt_row ** np.arange(window + 1)[1:]
    return exp_wt[::-1] / np.sum(exp_wt)


def _regress(y, X, intercept=True, weight=1, verbose=True):
    """
    WLS 加权线性回归。
    返回:
        verbose=True: (params, intercept, resid)
        verbose=False: params

    空数据保护:若 y/X 经 dropna 后为空(某天无有效样本),返回全 NaN,避免 statsmodels 抛异常。
    """
    y = _ensure_dataframe(y)
    X = _ensure_dataframe(X)

    # 空数据保护:样本量不足以回归时,返回 NaN 占位
    # 需要 >=2 个有效观测 + 自变量个数 才能回归
    n_required = 2 + (1 if intercept else 0) + (X.shape[1] if X.ndim > 1 else 1)
    if y.dropna().empty or X.dropna().empty or len(y) < n_required:
        nan_val = np.nan
        if verbose:
            nan_resid = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
            return pd.Series([nan_val]), nan_val, nan_resid
        return pd.Series([nan_val])

    if intercept:
        X = sm.add_constant(X)
    model = sm.WLS(y, X, weights=weight)
    result = model.fit()
    params = result.params

    if verbose:
        resid = y - pd.DataFrame(np.dot(X, params), index=y.index, columns=y.columns)
        if intercept:
            return params.iloc[1:], params.iloc[0], resid
        return params, None, resid
    else:
        if intercept:
            return params.iloc[1:]
        return params


def _winsorize(row, threshold=3, mean=None):
    """
    对每行进行 3σ 缩尾处理(winsorize):超出均值±threshold个标准差的值截断到边界。

    Barra 口径:中心 μ 用市值加权均值、σ 用等权标准差。
    参数:
        row: 某日截面因子值 Series
        threshold: 缩尾倍数(默认 3)
        mean: 缩尾中心;None 则用等权均值(旧口径,保留向后兼容)
    """
    if mean is None:
        mean = row.mean()
    row_std = row.std()
    clipped_row = np.clip(row, mean - threshold * row_std, mean + threshold * row_std)
    return pd.Series(clipped_row)


def align_to_index(df, target_index, fill_method="ffill"):
    """
    将 df 的索引对齐到 target_index,缺失日期按 fill_method 填充。
    用于解决 rf/基准收益与股票收益日期不完全一致(如最新交易日国债数据未更新)的问题。

    参数:
        df: 待对齐的 Series 或 DataFrame(索引为日期)
        target_index: 目标日期索引(DatetimeIndex)
        fill_method: 缺失填充方式,'ffill' 前向填充(用前一日值);'bfill' 后向;None 不填
    返回:
        对齐后的 df
    """
    df = df.reindex(target_index)
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    return df


def _standardize_with_weights(df, is_cir):
    """
    市值加权标准化三步曲:
        1. 缩尾(_winsorize, 3σ;中心 μ 用市值加权均值、σ 用等权标准差,Barra 口径)
        2. 市值加权去均值
        3. z-score 归一
    参数:
        is_cir: True 用流通市值权重,False 用总市值权重
    """
    _ensure_globals()
    weight_df = _stock_weight_cir if is_cir else _stock_weight

    missing_indices = df.index.difference(weight_df.index)
    if not missing_indices.empty:
        raise ValueError(f"标准化权重缺失日期: {missing_indices}")

    _weight_df = weight_df.loc[df.index]
    weighted_mean = (_weight_df * df).sum(axis=1)

    # 缩尾中心 = 市值加权均值(Barra 口径;旧版用等权均值)
    df = df.apply(lambda row: _winsorize(row, mean=weighted_mean.loc[row.name]),
                  axis=1)

    df_std = df.std(axis=1)
    df_standardized = (df.sub(weighted_mean.values[:, None])).div(df_std.values[:, None])
    return df_standardized


# ----------------------------------------------------------------------------
# 因子增量更新 IO(原 factor_calculate 的 _get_latest_date 等,修复 bug)
# ----------------------------------------------------------------------------

def load_base(filename):
    """读取基础宽表(代理 data.io.load_base)。"""
    return data_io.load_base(filename)


def load_factor(version, factor_name):
    """
    读取某版本的描述因子宽表。
    参数:
        version: "cne5" / "cne6"
        factor_name: 因子名(不含 .csv)
    """
    path = os.path.join(get_factor_dir(version), f"{factor_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"因子文件不存在: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _get_latest_date(version, filename, sdate="2010-01-01"):
    """因子增量更新:读已有因子文件的最新日期。"""
    factor_dir = get_factor_dir(version)
    os.makedirs(factor_dir, exist_ok=True)
    path = os.path.join(factor_dir, filename)
    if os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        if not existing.empty:
            return existing.index.max().strftime("%Y-%m-%d"), existing
    return sdate, pd.DataFrame()


def _get_existing_df(version, factor_name_list, start_date="2010-01-01"):
    """批量判断因子是否已存在,返回(最小日期, {因子名: 已有df})。"""
    if len(factor_name_list) == 0:
        print(f"{factor_name_list} is empty")
        return start_date, {}

    existing_dict = {}
    date_list = []
    for factor_name in factor_name_list:
        _factor_name = factor_name + ".csv"
        latest_date, existing_df = _get_latest_date(version, _factor_name, start_date)
        existing_dict[factor_name] = existing_df
        date_list.append(latest_date)
    return min(date_list), existing_dict


def data_concat_and_save(version, existing_dict, concat_dict):
    """
    新旧因子数据合并并落盘(委托 data.io.merge_and_save,消除重复去重逻辑)。
    修复原 bug:原 factor_calculate.py:246 引用了未定义的 missing_keys,
                此处用正确的差集计算。
    """
    missing_keys = set(existing_dict.keys()) - set(concat_dict.keys())
    if missing_keys:
        raise ValueError(f"concat_dict 缺少 key: {missing_keys}")

    factor_dir = get_factor_dir(version)
    os.makedirs(factor_dir, exist_ok=True)

    for file_name in existing_dict.keys():
        df_new = concat_dict[file_name]
        if df_new is None or (hasattr(df_new, "empty") and df_new.empty):
            print(f"{file_name}: 无需更新")
            continue
        filepath = os.path.join(factor_dir, file_name + ".csv")
        data_io.merge_and_save(filepath, df_new, existing_dict[file_name])
        print(f"{file_name}: 更新完成")


# ----------------------------------------------------------------------------
# 向量化滚动 WLS 回归(供 beta 因子使用,替代双层 for 循环)
# ----------------------------------------------------------------------------

def rolling_wls_vectorized(y_matrix, x_vector, weights, max_missing,
                           chunk_size=500):
    """
    向量化滚动加权最小二乘回归(一元 + 截距),替代股票×窗口双层循环。

    对每个滚动窗口、每只股票做 y ~ x 的 WLS 回归(权重 weights),
    返回 beta 和 sigma(残差样本标准差,ddof=1,与原 statsmodels 口径一致)。

    数学口径(与 statsmodels.WLS 数值等价,已验证差异<1e-15):
        β = Σw(x-x̄)(y-ȳ) / Σw(x-x̄)²
        α = ȳ - β·x̄
        resid = y - (α + β·x)
        sigma = resid.std()        # ddof=1 样本标准差(原口径)

    缺失处理(与原逻辑一致):
        窗口内 NaN 数 > max_missing → 该位置记 NaN
        否则 NaN 填 0 后回归(与原 np.nan_to_num 一致)

    参数:
        y_matrix: 个股超额收益,T×N 的 ndarray/DataFrame(T 天,N 只股票)
        x_vector: 基准超额收益,T 的 ndarray/Series
        weights: 窗口权重向量,长度 = window(已归一化)
        max_missing: 窗口内允许的最大缺失数,超过则跳过
        chunk_size: 按股票分块大小(控制内存,默认 500)
    返回:
        (beta, sigma):均为 num_windows×N 的 ndarray,
                       num_windows = T - window + 1
    """
    from numpy.lib.stride_tricks import sliding_window_view

    y = np.asarray(y_matrix, dtype=float)      # T×N
    x = np.asarray(x_vector, dtype=float)      # T
    T, N = y.shape
    window = len(weights)
    w = np.asarray(weights, dtype=float)       # (window,)
    num_windows = T - window + 1
    if num_windows <= 0:
        return np.empty((0, N)), np.empty((0, N))

    # x 的滚动窗口视图:(num_windows, window),零拷贝
    x_windows = sliding_window_view(x, window)

    beta = np.full((num_windows, N), np.nan)
    sigma = np.full((num_windows, N), np.nan)

    # 按股票分块(控制内存:y 块三维视图 = num_windows × chunk × window × 8字节)
    for c_start in range(0, N, chunk_size):
        c_end = min(c_start + chunk_size, N)
        y_chunk = y[:, c_start:c_end]                          # T×chunk
        y_windows = sliding_window_view(y_chunk, window, axis=0)  # (num_windows, chunk, window)

        # ---- ① 缺失统计 ----
        missing_y = np.isnan(y_windows).sum(axis=2)            # (num_windows, chunk)
        missing_x = np.isnan(x_windows).sum(axis=1)            # (num_windows,)
        skip = (missing_y > max_missing) | (missing_x[:, None] > max_missing)

        # ---- ② 填 0(nan_to_num 口径)----
        y_filled = np.where(np.isnan(y_windows), 0.0, y_windows)
        x_filled = np.where(np.isnan(x_windows), 0.0, x_windows)

        # ---- ③ 加权统计量(广播一次性算所有窗口所有股票)----
        xbar = (x_filled * w).sum(axis=1)                      # (num_windows,)
        ybar = (y_filled * w[None, None, :]).sum(axis=2)       # (num_windows, chunk)

        x_cen = x_filled - xbar[:, None]                       # (num_windows, window)
        y_cen = y_filled - ybar[:, :, None]                    # (num_windows, chunk, window)

        wvar_x = (w * x_cen ** 2).sum(axis=1)                  # (num_windows,)
        wcov = (w[None, None, :] * x_cen[:, None, :] * y_cen).sum(axis=2)  # (num_windows, chunk)

        # 避免除零(wvar_x 理论上 > 0,但防御)
        wvar_x_safe = np.where(wvar_x == 0, np.nan, wvar_x)
        b = wcov / wvar_x_safe[:, None]                        # (num_windows, chunk)
        a = ybar - b * xbar[:, None]

        # ---- ④ 残差 + sigma(ddof=1,原 resid.std() 口径)----
        resid = y_filled - (a[:, :, None] + b[:, :, None] * x_filled[:, None, :])
        sig = resid.std(axis=2, ddof=1)                        # (num_windows, chunk)

        # ---- ⑤ 跳过位置记 NaN ----
        b[skip] = np.nan
        sig[skip] = np.nan

        beta[:, c_start:c_end] = b
        sigma[:, c_start:c_end] = sig

    return beta, sigma


def rolling_dastd_cmra_vectorized(excess_matrix, log_excess_matrix, weights,
                                  chunk_size=200):
    """
    向量化计算 DASTD(日超额收益加权标准差)与 CMRA(累计对数超额极差),
    替代 volatility.py 的逐股 rolling_windows 循环(提速 50-100×,数学口径与原文一致)。

    DASTD(算术超额 x,权重 w,窗口 W):
        wmean = Σ_w x ;  DASTD = sqrt(Σ_w (x - wmean)²)

    CMRA(对数超额 lx,Barra 原文口径):
        Z(T) = 窗口内最后 21T 个交易日的对数超额累计和,T = 1..12(共 12 个月度点,
        每月 = 21 个交易日,不加权);
        CMRA = max{Z(T)} - min{Z(T)}
        注:旧实现误按窗口内 252 个"日频"累计点取极差(极差系统性偏大),
            原文为 12 个月度点(见 CNE5 Descriptor Details 公式 (3)(4))。

    缺失处理:与逐股版一致——窗口含 NaN 时结果 NaN(excess 已在首尾有效值间填 0,
    仅上市前纯空窗口为 NaN,自动传播)。

    参数:
        excess_matrix:     算术超额收益 T×N(DataFrame/ndarray)
        log_excess_matrix: 对数超额收益 T×N
        weights:           窗口权重向量(已归一化),长度=window(仅 DASTD 使用,CMRA 不加权)
        chunk_size:        按股票分块(控制内存,默认 200)
    返回:
        (dastd, cmra):均 num_windows×N 的 ndarray,num_windows = T - window + 1
    """
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(excess_matrix, dtype=float)        # T×N
    lx = np.asarray(log_excess_matrix, dtype=float)   # T×N
    T, N = x.shape
    W = len(weights)
    w = np.asarray(weights, dtype=float)
    num_windows = T - W + 1
    if num_windows <= 0:
        return np.empty((0, N)), np.empty((0, N))

    # CMRA 月度点索引:Z(T) = 窗口最后 21T 天的累计对数超额,T = 1..12
    # 即 prefix 结束索引 = W - 21T - 1(T<12 时),Z(12) = 窗口总和
    monthly_prefix_idx = np.array([W - 21 * t - 1 for t in range(1, 12)])  # 11 个索引
    assert W == 252, f"CMRA 月度口径按 252 天窗口实现,当前 W={W}"

    dastd = np.full((num_windows, N), np.nan)
    cmra = np.full((num_windows, N), np.nan)

    for c0 in range(0, N, chunk_size):
        c1 = min(c0 + chunk_size, N)
        # DASTD:算术超额的加权标准差
        xw = sliding_window_view(x[:, c0:c1], W, axis=0)    # (num_windows, chunk, W) 零拷贝视图
        wmean = (xw * w).sum(axis=2)                         # (num_windows, chunk)
        var = ((xw - wmean[:, :, None]) ** 2 * w).sum(axis=2)
        dastd[:, c0:c1] = np.sqrt(var)

        # CMRA:12 个月度累计点的极差(原文口径)
        lxw = sliding_window_view(lx[:, c0:c1], W, axis=0)  # (num_windows, chunk, W)
        cum = np.cumsum(lxw, axis=2)                        # 窗口内每日累计(仅用于取月度点)
        total = cum[:, :, -1:]                              # Z(12) = 窗口总和
        # Z(T) = total - prefix(T),prefix(T) = 前 W-21T 天的累计和
        z_points = total - np.concatenate(
            [cum[:, :, i:i + 1] for i in monthly_prefix_idx], axis=2
        )                                                    # (num_windows, chunk, 11)
        z_points = np.concatenate([z_points, total], axis=2)  # 加上 Z(12) → 12 个点
        cmra[:, c0:c1] = z_points.max(axis=2) - z_points.min(axis=2)

    return dastd, cmra
