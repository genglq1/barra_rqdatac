# -*- coding: utf-8 -*-
"""
Earnings Yield 因子:CETOP / ETOP / EPIBS
=========================================
Barra CNE-5 原文权重:0.68·EPIBS + 0.11·ETOP + 0.21·CETOP

三个分量均用 rqdatac 现成的估值/预期字段,无需手算 TTM 与日期对齐:
    CETOP = 1 / pcf_ratio_ttm(经营现金流TTM/市值,严格匹配 Barra 经营现金流口径)
    ETOP  = ep_ratio_ttm(归母净利润TTM/总市值)
    EPIBS = comp_con_eps_ftm / 不复权收盘价(一致预期EPS_FTM / 当前真实价)

注:CETOP 取 pcf_ratio_ttm(经营市现率)的倒数;不用 cfp_ratio_ttm(其分子含三大活动
    现金流,与 Barra 只用经营现金流的定义不符)。
"""

import os
import numpy as np
import pandas as pd

from config import get_path
from factors import common as F

VERSION = "cne5"


def cal_earningsyield_factor(start_date="2010-01-01", end_date=None):
    """计算盈利因子 cetop / etop / epibs(Barra CNE-5 三分量)。"""
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["cetop", "etop", "epibs"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    # ---- CETOP = 1/pcf_ratio_ttm(经营现金流TTM/市值)----
    # 边界:pcf≤0(经营现金流为负)→1/pcf为负(合理);pcf=0→inf,replace为NaN
    # 容错:pcf_ratio_ttm 缺失(如 Quota 未下载)时 CETOP 跳过,不影响 EPIBS/ETOP
    pcf_path = os.path.join(get_path("base"), "pcf_ratio_ttm.csv")
    if os.path.exists(pcf_path):
        pcf = F.load_base("pcf_ratio_ttm.csv").loc[latest_date:end_date]
        cetop = 1 / pcf
        cetop = cetop.replace([np.inf, -np.inf], np.nan)
        cetop = F._standardize_with_weights(cetop, True)
    else:
        print("⚠️  pcf_ratio_ttm.csv 不存在,CETOP 跳过(EPIBS/ETOP 不受影响)")
        cetop = pd.DataFrame()

    # ---- ETOP = ep_ratio_ttm(归母净利润TTM/总市值)----
    ep = F.load_base("ep_ratio_ttm.csv").loc[latest_date:end_date]
    etop = F._standardize_with_weights(ep, True)

    # ---- EPIBS = 一致预期EPS(FTM) / 不复权收盘价(当前真实价)----
    # 一致预期 EPS 基于当前股本,须用当前真实价(不复权);后复权价会随除权累积放大、
    # 系统性低估老股 EPIBS。comp_con_eps_ftm 覆盖分析师跟踪股票,小盘股缺失→NaN。
    con_eps = F.load_base("comp_con_eps_ftm.csv").loc[latest_date:end_date]
    # 优先用不复权价;若尚未下载(stock_close_unadjusted.csv 缺失),fallback 后复权价并警告
    unadj_path = os.path.join(get_path("base"), "stock_close_unadjusted.csv")
    if os.path.exists(unadj_path):
        price = F.load_base("stock_close_unadjusted.csv").loc[latest_date:end_date]
    else:
        print("⚠️  stock_close_unadjusted.csv 不存在,EPIBS 暂用后复权价"
              "(老股 EPIBS 会被低估,建议重跑 data 层 price 下载不复权价)")
        price = F.load_base("stock_close.csv").loc[latest_date:end_date]
    epibs = con_eps / price
    epibs = F._standardize_with_weights(epibs, True)

    concat_dict = {"cetop": cetop, "etop": etop, "epibs": epibs}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
