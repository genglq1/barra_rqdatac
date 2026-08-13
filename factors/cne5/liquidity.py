
# -*- coding: utf-8 -*-
"""
Liquidity 因子:stom / stoq / stoa(换手率)
=========================================
迁移自原 factor_calculate.py 的 cal_liquidity_factor。
21/63/252 日换手率对数,一个月内超过 3 个空值都不计算。
"""

import numpy as np
import pandas as pd

from factors import common as F

VERSION = "cne5"

# Barra Liquidity 因子参数(stom/stoq/stoa 的换手率累积窗口与缺失容忍)
WINDOW_MONTH = 21       # 月度换手率窗口
WINDOW_QUARTER = 63     # 季度换手率窗口
WINDOW_ANNUAL = 252     # 年度换手率窗口
TOL_MONTH = 3           # 月度窗口允许缺失天数
TOL_QUARTER = 9         # 季度窗口允许缺失天数
TOL_ANNUAL = 36         # 年度窗口允许缺失天数


def cal_liquidity_factor(start_date="2010-01-01", end_date=None):
    """计算换手率因子 stom/stoq/stoa:一个月内超过3个空值都不计算。"""
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["stom", "stoq", "stoa"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    # 容错:stock_turnover 缺失(如 Quota 未下载)时跳过整个 Liquidity,不阻断 pipeline
    import os
    from config import get_path
    if not os.path.exists(os.path.join(get_path("base"), "stock_turnover.csv")):
        print("⚠️  stock_turnover.csv 不存在,Liquidity(stom/stoq/stoa)跳过")
        concat_dict = {"stom": pd.DataFrame(), "stoq": pd.DataFrame(), "stoa": pd.DataFrame()}
        F.data_concat_and_save(VERSION, existing_dict, concat_dict)
        return

    a_turnover_data = F.load_base("stock_turnover.csv")

    loc = a_turnover_data.index.get_loc(a_turnover_data[latest_date:end_date].index[0])
    _start_date = a_turnover_data.index[max(0, loc - WINDOW_ANNUAL + 1)]
    a_turnover_data = a_turnover_data.loc[_start_date:, :]

    # 少于阈值空值都要计算(min_periods = window - 容忍空缺数)
    a_turnover_21_data = a_turnover_data.rolling(window=WINDOW_MONTH,  min_periods=WINDOW_MONTH - TOL_MONTH).sum()
    a_turnover_63_data = a_turnover_data.rolling(window=WINDOW_QUARTER, min_periods=WINDOW_QUARTER - TOL_QUARTER).sum()
    a_turnover_252_data = a_turnover_data.rolling(window=WINDOW_ANNUAL, min_periods=WINDOW_ANNUAL - TOL_ANNUAL).sum()

    stom = np.log(a_turnover_21_data)
    stoq = np.log(a_turnover_63_data / 3)
    stoa = np.log(a_turnover_252_data / 12)
    # sum=0(长期停牌股窗口内全无交易)→ log(0)=-inf,清为 NaN;
    # 否则 -inf 进入 _standardize_with_weights 的 winsorize 会使截面 mean=-inf → 当日整列 NaN
    stom = stom.replace(-np.inf, np.nan)
    stoq = stoq.replace(-np.inf, np.nan)
    stoa = stoa.replace(-np.inf, np.nan)

    stom = stom.loc[latest_date:end_date]
    stoq = stoq.loc[latest_date:end_date]
    stoa = stoa.loc[latest_date:end_date]

    stom = F._standardize_with_weights(stom, True)
    stoq = F._standardize_with_weights(stoq, True)
    stoa = F._standardize_with_weights(stoa, True)

    concat_dict = {"stom": stom, "stoq": stoq, "stoa": stoa}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
