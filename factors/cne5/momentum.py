# -*- coding: utf-8 -*-
"""
Momentum 因子:rstr
==================
迁移自原 factor_calculate.py 的 cal_momentum_factor。
504 日加权累计超额收益,前 21 日剔除,半衰期 126。
"""

import numpy as np
import pandas as pd

from factors import common as F

VERSION = "cne5"

# Barra Momentum 因子参数
WINDOW = 504          # 累计超额收益窗口(约2年)
HALF_LIFE = 126       # 指数权重半衰期(约半年)
LAG = 21              # 剔除最近 LAG 个交易日(避免短期反转干扰)
MIN_PERIODS = 483     # 滚动窗口最小有效样本数


def cal_momentum_factor(start_date="2010-01-01", end_date=None):
    """计算动量因子 rstr:对空值不做处理。"""
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["rstr"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    ret_data = F.load_base("stock_ret.csv")
    rf_ret_data = F.load_base("rf.csv") / 365

    loc = ret_data.index.get_loc(ret_data[latest_date:].index[0])
    _start_date = ret_data.index[max(0, loc - WINDOW - LAG + 1)]   # 新开始时间索引

    ret_data = ret_data.loc[_start_date:end_date]
    # rf 对齐到 ret_data 索引(缺失日前向填充)
    rf_ret_data = F.align_to_index(rf_ret_data, ret_data.index, fill_method="ffill")

    # 每日收益率 - 无风险收益率(对数相减)
    log_excess_ret = ret_data.apply(
        lambda x: np.log(1 + x) - np.log(1 + rf_ret_data["rf"].values), axis=0
    )
    log_excess_ret = log_excess_ret.loc[ret_data.index, :]

    weight_array = F._get_exp_weight(WINDOW, HALF_LIFE)
    min_length = WINDOW + LAG - 1

    # 窗口不足保护:数据长度 < min_length 时无法计算,产出空
    if log_excess_ret.shape[0] < min_length:
        print(f"⚠️  rstr 跳过:数据仅 {log_excess_ret.shape[0]} 天,不足所需 {min_length} 天。"
              f"请拉长历史数据。")
        concat_dict = {"rstr": pd.DataFrame()}
        F.data_concat_and_save(VERSION, existing_dict, concat_dict)
        return

    rstr = pd.DataFrame(index=log_excess_ret.index, columns=log_excess_ret.columns)

    # 此处已保证数据长度 >= min_length(否则前面已 return)
    shifted_data = log_excess_ret.shift(LAG)
    for col in shifted_data.columns:
        rolling_window = shifted_data[col].rolling(window=WINDOW, min_periods=MIN_PERIODS)
        rstr[col] = rolling_window.apply(lambda x: np.nansum(x * weight_array))

    rstr = rstr.loc[latest_date:end_date]
    rstr = F._standardize_with_weights(rstr, True)

    concat_dict = {"rstr": rstr}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
