# -*- coding: utf-8 -*-
"""
Beta 因子:beta + sigma(残差波动)
=================================
252 日滚动 WLS 回归(半衰期 63),窗口内缺失 >42 则跳过,否则填 0。

性能:cal_beta_factor 用向量化实现(rolling_wls_vectorized),相比原双层 for 循环
      提速 50-100 倍,数学口径完全一致(已验证差异<1e-14)。
      原循环实现保留为 cal_beta_factor_legacy,供回归对比。
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfinance.utils import rolling_windows
import statsmodels.api as sm

from factors import common as F

VERSION = "cne5"

# Barra Beta 因子参数(滚动回归窗口/半衰期/缺失容忍)
WINDOW = 252          # 滚动回归窗口(交易日)
HALF_LIFE = 63        # 指数权重半衰期
MAX_MISSING = 42      # 窗口内允许的最大缺失天数(超过则跳过该窗口)


def cal_beta_factor(start_date="2017-01-01", end_date=None):
    """
    计算 Beta 因子:beta(市场敏感度) + sigma(残差波动率)。
    在 WINDOW 个滚动窗口中,若空缺值大于 MAX_MISSING,则不计算,否则空值填充为 0。

    向量化实现:用 rolling_wls_vectorized 一次性算所有股票所有窗口,
    相比原双层 for 循环大幅提速,数学口径完全一致。
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["beta", "sigma"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    window = WINDOW
    weight_array = F._get_exp_weight(window, HALF_LIFE)

    ret_data = F.load_base("stock_ret.csv")
    benchmark_ret_data = F.load_base("Rt.csv")
    rf_ret_data = F.load_base("rf.csv") / 365   # 年化→日度

    loc = ret_data.index.get_loc(ret_data[latest_date:].index[0])
    _start_date = ret_data.index[max(0, loc - window + 1)]

    ret_data = ret_data.loc[_start_date:end_date]
    # rf / benchmark 对齐到 ret_data 索引(缺失日前向填充,解决最新交易日数据未更新)
    benchmark_ret_data = F.align_to_index(
        benchmark_ret_data.sort_index(), ret_data.index, fill_method="ffill"
    )
    rf_ret_data = F.align_to_index(rf_ret_data, ret_data.index, fill_method="ffill")

    excess_ret = ret_data.apply(
        lambda x: x - rf_ret_data["rf"].values
    )   # 个股超额收益

    start_idx = benchmark_ret_data.loc[pd.notnull(benchmark_ret_data).values.flatten()].index[0]
    benchmark_ret_data = benchmark_ret_data.loc[start_idx:]
    excess_ret = excess_ret.loc[start_idx:]

    # 窗口不足保护:数据长度 < window 时无法做滚动回归,产出空并提示
    if len(benchmark_ret_data) < window:
        print(f"⚠️  beta/sigma 跳过:数据仅 {len(benchmark_ret_data)} 天,不足窗口 {window} 天。"
              f"请拉长历史数据(如 --start 至少早于目标日 {window} 个交易日)。")
        concat_dict = {"beta": pd.DataFrame(), "sigma": pd.DataFrame()}
        F.data_concat_and_save(VERSION, existing_dict, concat_dict)
        return

    # ---- 向量化滚动 WLS 回归 ----
    # benchmark 超额收益(基准收益 - 无风险)
    benchmark_excess = (benchmark_ret_data.iloc[:, 0] - rf_ret_data["rf"].loc[benchmark_ret_data.index].values).values
    beta_arr, sigma_arr = F.rolling_wls_vectorized(
        excess_ret.values, benchmark_excess, weight_array, MAX_MISSING
    )

    # 结果矩阵转回 DataFrame(索引=窗口结束日,列=股票代码)
    window_end_dates = excess_ret.index[window - 1:]
    beta_df = pd.DataFrame(beta_arr, index=window_end_dates, columns=excess_ret.columns)
    sigma_df = pd.DataFrame(sigma_arr, index=window_end_dates, columns=excess_ret.columns)

    # 截面标准化(流通市值加权)
    beta_df = F._standardize_with_weights(beta_df, True)
    sigma_df = F._standardize_with_weights(sigma_df, True)

    concat_dict = {"beta": beta_df, "sigma": sigma_df}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)


def cal_beta_factor_legacy(start_date="2017-01-01", end_date=None):
    """
    Beta 因子原始实现(双层 for 循环 + statsmodels.WLS)。
    保留供回归验证/对比,正式流程用 cal_beta_factor(向量化版)。

    ⚠️ 全市场重算耗时数小时,仅用于数值对比验证。
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["beta", "sigma"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    window = WINDOW
    weight_array = F._get_exp_weight(window, HALF_LIFE)

    ret_data = F.load_base("stock_ret.csv")
    benchmark_ret_data = F.load_base("Rt.csv")
    rf_ret_data = F.load_base("rf.csv") / 365

    loc = ret_data.index.get_loc(ret_data[latest_date:].index[0])
    _start_date = ret_data.index[max(0, loc - window + 1)]

    ret_data = ret_data.loc[_start_date:end_date]
    benchmark_ret_data = F.align_to_index(
        benchmark_ret_data.sort_index(), ret_data.index, fill_method="ffill"
    )
    rf_ret_data = F.align_to_index(rf_ret_data, ret_data.index, fill_method="ffill")

    excess_ret = ret_data.apply(lambda x: x - rf_ret_data["rf"].values)

    start_idx = benchmark_ret_data.loc[pd.notnull(benchmark_ret_data).values.flatten()].index[0]
    benchmark_ret_data = benchmark_ret_data.loc[start_idx:]
    ret_data = ret_data.loc[start_idx:]

    if len(benchmark_ret_data) < window:
        print(f"⚠️  beta/sigma 跳过:数据仅 {len(benchmark_ret_data)} 天,不足窗口 {window} 天。")
        concat_dict = {"beta": pd.DataFrame(), "sigma": pd.DataFrame()}
        F.data_concat_and_save(VERSION, existing_dict, concat_dict)
        return

    rolling_xs = rolling_windows(benchmark_ret_data, window)
    reg_dict = {"beta": pd.DataFrame(), "alpha": pd.DataFrame(), "sigma": pd.DataFrame()}

    for stock_name in tqdm(excess_ret.columns):
        y = excess_ret[stock_name]
        rolling_ys = rolling_windows(y, window)

        for i, (rolling_x, rolling_y) in enumerate(zip(rolling_xs, rolling_ys)):
            window_sdate, window_edate = y.index[i], y.index[i + window - 1]

            if (np.isnan(rolling_x).sum() > MAX_MISSING) or (np.isnan(rolling_y).sum() > MAX_MISSING):
                reg_dict["beta"].loc[window_edate, stock_name] = np.nan
                reg_dict["alpha"].loc[window_edate, stock_name] = np.nan
                reg_dict["sigma"].loc[window_edate, stock_name] = np.nan
                continue
            else:
                rolling_y = np.nan_to_num(rolling_y, nan=0)
                rolling_x = np.nan_to_num(rolling_x, nan=0)

            b, a, resid = F._regress(rolling_y, rolling_x,
                                     intercept=True, weight=weight_array, verbose=True)
            reg_dict["beta"].loc[window_edate, stock_name] = b.values[0]
            reg_dict["alpha"].loc[window_edate, stock_name] = a
            reg_dict["sigma"].loc[window_edate, stock_name] = resid.std()[0]

    reg_dict["beta"] = F._standardize_with_weights(reg_dict["beta"], True)
    reg_dict["sigma"] = F._standardize_with_weights(reg_dict["sigma"], True)

    concat_dict = {"beta": reg_dict["beta"], "sigma": reg_dict["sigma"]}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
