# -*- coding: utf-8 -*-
"""
Size 因子:lncap(规模) + nlsize(非线性市值)
===========================================
- lncap:ln(总市值)经流通市值加权标准化,即 Size 因子暴露
- nlsize:Barra CNE-5 非线性市值,在标准化的 Size 暴露上做立方回归取残差(原文口径)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from factors import common as F

VERSION = "cne5"


def cal_size_factor(start_date="2010-01-01", end_date=None):
    """
    计算规模因子 lncap:log(总市值),流通市值加权标准化。
    对空值不做处理。
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["lncap"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    stock_size_df = F.load_base("stock_size.csv")
    stock_size_df = stock_size_df.loc[latest_date:end_date]

    lncap = np.log(stock_size_df)
    lncap = F._standardize_with_weights(lncap, True)

    concat_dict = {"lncap": lncap}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)


def cal_nonlinearsize_factor(start_date="2010-01-01", end_date=None):
    """
    计算非线性市值因子 nlsize(Barra CNE-5 原文口径)。

    Barra 定义:在【标准化的 Size 暴露值】上做立方回归取残差,而非原始 log(市值):
        1. lncap = ln(总市值)
        2. 标准化(缩尾 + 流通市值加权去均值 + z-score)→ Size 暴露 LNCAP
        3. NLS 残差 = LNCAP³ 对 LNCAP 做【市值加权 WLS】正交回归的残差(原文 regression-weighted)
        4. 残差再标准化 → nlsize

    注:此前版本误对未标准化的原始 log(市值) 做立方回归,与 Barra 原文不符。
        标准化后的 LNCAP 均值≈0、标准差≈1,立方值在合理范围(±27),
        残差能均衡反映市值维度的非线性成分,且天然与 Size 因子正交。
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["nlsize"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    size_data = F.load_base("stock_size.csv")
    size_data = size_data.loc[latest_date:end_date]
    lncap_raw = np.log(size_data)

    # 步骤1:标准化得到 Size 因子暴露 LNCAP(与 cal_size_factor 口径一致)
    size_exposure = F._standardize_with_weights(lncap_raw, True)

    # 步骤2:在标准化的 Size 暴露上做立方,再对 Size 做【市值加权】正交(原文 regression-weighted)
    # 回归权重口径 = √流通市值(与 Barra 截面回归权重一致,与标准化权重不同源);
    # 标准化(步骤1)仍用流通市值权重(_standardize_with_weights is_cir=True)
    weight_df = F.get_stock_weight(is_cir=True, sqrt=True)
    nlsize = pd.DataFrame()
    for idx in tqdm(size_exposure.index):
        x = size_exposure.loc[idx, :].dropna()
        if x.empty:
            continue   # 当天无有效数据,跳过
        # 权重与 x 对齐:取当天有市值权重的股票交集,避免缺权重的股票扭曲回归
        w_row = weight_df.loc[idx]
        common = x.index.intersection(w_row.dropna().index)
        if len(common) == 0:
            continue
        x = x.loc[common]
        x_3 = x ** 3
        w = w_row.loc[common].values
        beta, alpha, resid = F._regress(x_3, x, intercept=True, weight=w, verbose=True)
        nlsize = pd.concat([nlsize, resid.T], axis=0)

    # 步骤3:残差再标准化
    nlsize = F._standardize_with_weights(nlsize, True)
    concat_dict = {"nlsize": nlsize}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
