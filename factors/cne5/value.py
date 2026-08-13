# -*- coding: utf-8 -*-
"""
Value 因子:btop(账面市值比)
==========================
迁移自原 factor_calculate.py 的 cal_booktoprice_factor。
btop = 1 / PB,流通市值加权标准化。
"""

import pandas as pd

from factors import common as F

VERSION = "cne5"


def cal_booktoprice_factor(start_date="2010-01-01", end_date=None):
    """计算账面市值比因子 btop:对空值不做处理。"""
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["btop"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    a_pb_data = F.load_base("stock_pb.csv")
    a_pb_data = a_pb_data.loc[start_date:end_date]

    btop = 1 / a_pb_data
    btop = F._standardize_with_weights(btop, True)

    concat_dict = {"btop": btop}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
