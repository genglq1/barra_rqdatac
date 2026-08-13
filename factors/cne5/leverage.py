# -*- coding: utf-8 -*-
"""
Leverage 因子:mlev / dtoa / blev
================================
Barra CNE-5 原文权重:0.38·MLEV + 0.35·DTOA + 0.27·BLEV

- MLEV:市场杠杆 = (ME + PE + LD) / ME
- DTOA:资产负债比 = TD / TA
- BLEV:账面杠杆 = (BE + PE + LD) / BE

变量:ME=普通股市值, PE=优先股, LD=长期负债(非流动负债),
      TD=总负债, TA=总资产, BE=普通股账面权益(归母权益)。

注:rqdatac get_factor 返回的财务数据已是日频且 PIT 对齐(基于最新已披露财报填充),
    无需再做季频→日频转换。原实现误调用 _transform_to_daily 导致日频数据被破坏为全空。
"""

import os

import numpy as np
import pandas as pd

from config import get_path
from factors import common as F

VERSION = "cne5"


def cal_leverage_factor(start_date="2010-01-01", end_date=None):
    """计算杠杆因子 mlev / dtoa / blev(Barra CNE-5 定义)。"""
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["mlev", "dtoa", "blev"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    me = F.load_base("stock_size.csv")                           # ME:普通股市值
    ld = F.load_base("total_ncl.csv")                            # LD:长期负债(非流动负债)
    td = F.load_base("total_liab.csv")                           # TD:总负债
    ta = F.load_base("total_assets.csv")                         # TA:总资产
    be = F.load_base("total_hldr_eqy_exc_min_int.csv")           # BE:归母权益(账面)

    # PE:优先股。rqdatac 因子库不提供 oth_eqt_tools_p_shr,缺失按 0(绝大多数股票无优先股)
    pe_path = os.path.join(get_path("base"), "oth_eqt_tools_p_shr.csv")
    if os.path.exists(pe_path) and os.path.getsize(pe_path) > 50:
        pe = F.load_base("oth_eqt_tools_p_shr.csv")
    else:
        pe = pd.DataFrame(0.0, index=me.index, columns=me.columns)
        print("优先股数据:rqdatac 不提供,按 0 处理")

    # 统一切片到计算区间
    # ME 用当日总市值(原文 "the market value of common equity on the last trading day");
    # 旧实现误用 shift(1) 取前一日市值,已按原文修正
    me = me.loc[latest_date:end_date]
    pe = pe.loc[latest_date:end_date]
    # 财务宽表自 2005 起存(供 TTM/5年回归),而标准化权重(流通市值)自 2020 起;
    # 统一对齐到 ME 索引(2020+),避免 2010-2019 无权重行触发标准化缺日期报错
    ld = ld.reindex(me.index)
    td = td.reindex(me.index)
    ta = ta.reindex(me.index)
    be = be.reindex(me.index)

    # ---- MLEV = (ME + PE + LD) / ME ----
    # PE/LD 缺失(如银行股无非流动负债科目)填 0;ME 为 0 时结果置 NaN
    pe_mlev = pe.fillna(0)
    ld_mlev = ld.fillna(0)
    df_mlev = (me + pe_mlev + ld_mlev) / me
    df_mlev = df_mlev.replace([np.inf, -np.inf], np.nan)

    # ---- DTOA = TD / TA ----
    df_dtoa = td / ta
    df_dtoa = df_dtoa.replace([np.inf, -np.inf], np.nan)

    # ---- BLEV = (BE + PE + LD) / BE ----
    be_blev = be
    pe_blev = pe.fillna(0)
    ld_blev = ld.fillna(0)
    df_blev = (be_blev + pe_blev + ld_blev) / be_blev
    df_blev = df_blev.replace([np.inf, -np.inf], np.nan)

    # 截面标准化
    df_mlev = F._standardize_with_weights(df_mlev, True)
    df_dtoa = F._standardize_with_weights(df_dtoa, True)
    df_blev = F._standardize_with_weights(df_blev, True)

    concat_dict = {"mlev": df_mlev, "dtoa": df_dtoa, "blev": df_blev}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
