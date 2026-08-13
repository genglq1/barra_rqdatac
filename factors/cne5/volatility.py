# -*- coding: utf-8 -*-
"""
Residual Volatility 因子:dastd / cmra / hsigma
==============================================
Barra CNE-5 原文权重:0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA

- DASTD:过去252天日超额收益的加权标准差(半衰期42天)
- CMRA :过去252天累计超额【对数】收益的极差,取 12 个月度累计点(每月=21交易日):
         Z(T)=窗口内最后 21T 天对数超额累计(T=1..12),CMRA=max Z - min Z(原文公式 3/4)
- HSIGMA:个股超额收益对市场超额收益(Beta)回归的残差时序标准差,
          与 beta.py 产出的 sigma 因子口径一致,直接复用(且 sigma 已在 beta.py 标准化,
          此处不再重复标准化)

⚠️ 因子依赖:hsigma 复用 sigma,故需 beta 因子先算完(registry 声明 depends_on)。
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfinance.utils import rolling_windows

from config import get_factor_dir

from factors import common as F

VERSION = "cne5"

# Barra Residual Volatility 因子参数(dastd/cmra)
WINDOW = 252          # 滚动窗口(交易日)
HALF_LIFE = 42        # 指数权重半衰期


def cal_residualvolatility_factor(start_date="2010-01-01", end_date=None):
    """
    计算残差波动率因子:dastd / cmra / hsigma(Barra CNE-5 原文定义)。
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["dastd", "cmra", "hsigma"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    window = WINDOW
    weight_array = F._get_exp_weight(window, HALF_LIFE)

    ret_data = F.load_base("stock_ret.csv")
    rf_ret_data = F.load_base("rf.csv") / 365

    loc = ret_data.index.get_loc(ret_data[latest_date:].index[0])
    _start_date = ret_data.index[max(0, loc - 252 + 1)]

    ret_data = ret_data.sort_index()[_start_date:end_date]
    # rf 对齐到 ret_data 索引(缺失日前向填充)
    rf_ret_data = F.align_to_index(rf_ret_data, ret_data.index, fill_method="ffill")

    excess_ret = ret_data.apply(
        lambda x: x - rf_ret_data["rf"].values
    )   # 超额收益率(减法)
    # 仅对退市之前的数据进行填充
    excess_ret = excess_ret.apply(
        lambda x: x if x.isnull().all()
        else x[x.first_valid_index():x.last_valid_index()].fillna(0)
    )

    log_excess_ret = ret_data.apply(
        lambda x: np.log(x + 1) - np.log(rf_ret_data["rf"].values + 1)
    )   # 超额收益率对数相减
    log_excess_ret = log_excess_ret.apply(
        lambda x: x if x.isnull().all()
        else x[x.first_valid_index():x.last_valid_index()].fillna(0)
    )

    # 窗口不足保护:dastd/cmra 需 252 天滚动窗口
    if len(excess_ret) < window:
        print(f"⚠️  dastd/cmra 跳过:数据仅 {len(excess_ret)} 天,不足窗口 {window} 天。")
        dastd = pd.DataFrame()
        cmra = pd.DataFrame()
    else:
        # 向量化计算(替代逐股 rolling_windows 循环,提速 50-100×,数学口径完全一致)
        # DASTD=加权标准差;CMRA=对数超额累计极差
        dastd_arr, cmra_arr = F.rolling_dastd_cmra_vectorized(
            excess_ret.values, log_excess_ret.values, weight_array
        )
        window_end_dates = excess_ret.index[window - 1:]
        dastd = pd.DataFrame(dastd_arr, index=window_end_dates, columns=excess_ret.columns)
        cmra = pd.DataFrame(cmra_arr, index=window_end_dates, columns=excess_ret.columns)

    # HSIGMA:个股超额收益对市场超额收益(Beta)回归的残差时序标准差
    # 按 Barra 原文,HSIGMA = 残差波动率。beta.py 在计算 beta 时已产出该残差标准差(即 sigma 因子),
    # 两者口径完全一致,故直接复用 sigma 因子,无需重复回归。
    # 注1:原实现误将 sigma 对 [beta,size] 做截面回归取残差,那是对波动率做市值中性化,非 Barra HSIGMA。
    # 注2:sigma 在 beta.py 落盘前已做截面标准化,此处直接切片复用,不再二次标准化。
    hsigma = pd.DataFrame()
    beta_path = os.path.join(get_factor_dir(VERSION), "beta.csv")
    if not os.path.exists(beta_path) or os.path.getsize(beta_path) < 50:
        print("⚠️  hsigma 跳过:依赖的 sigma(随 beta 产出)不存在或为空(需先有足够历史数据算 beta)。")
    else:
        sigma = F.load_factor(VERSION, "sigma")
        hsigma = sigma.loc[latest_date:end_date]
        if hsigma.empty:
            print("⚠️  hsigma 跳过:sigma 为空。")

    # 标准化(若 dastd/cmra/hsigma 为空,标准化会报错,故分别判断)
    if not dastd.empty:
        dastd = dastd.loc[latest_date:end_date]
        dastd = F._standardize_with_weights(dastd, True)
    if not cmra.empty:
        cmra = cmra.loc[latest_date:end_date]
        cmra = F._standardize_with_weights(cmra, True)
    # hsigma 复用已标准化的 sigma,无需再标准化

    concat_dict = {"dastd": dastd, "cmra": cmra, "hsigma": hsigma}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
