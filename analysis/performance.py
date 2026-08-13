# -*- coding: utf-8 -*-
"""
业绩指标库
==========
迁移自原 performance_index_module.py,计算收益/风险/收益风险比/胜率类指标。

改进:
    - 交易日历改为读 data_store/base/trade_cal.csv
    - 修复 pandas 2.x 废弃的 fillna(method='ffill') -> .ffill()
    - trade_cal 需含 is_week_start_end 列(-1 表示周末);若无则内部派生

依赖输入:
    净值 DataFrame(index=日期, 列=净值),由调用方传入
"""

import pandas as pd
import numpy as np
import dateutil.relativedelta

from data import io as data_io


def _load_trade_cal():
    """读取交易日历,补充周标记列(若无)。"""
    cal = data_io.load_base("trade_cal.csv")
    if "is_week_start_end" not in cal.columns:
        # 派生:周末标记为 -1(周五/最后交易日),其他为 0
        # 简化:按周一为周首,周五为周末
        cal["weekday"] = cal.index.weekday
        cal["is_week_start_end"] = np.where(cal["weekday"] == 4, -1, 0)
    return cal


def maxdrawdown(df, column_name):
    """
    计算最大回撤。
    返回: dict {最大回撤, 最大回撤开始时间, 最大回撤结束时间}
    """
    md_dict = {}
    k = np.argmax(
        (np.maximum.accumulate(df[column_name]) - df[column_name])
        / np.maximum.accumulate(df[column_name])
    )
    if k > 0:
        m = np.argmax(df[column_name][:k])
        md = (1 - df[column_name][k] / df[column_name][m])
        md_dict["最大回撤开始时间"] = df.index[m]
        md_dict["最大回撤结束时间"] = df.index[k]
    else:
        md = 0
        md_dict["最大回撤开始时间"] = "--"
        md_dict["最大回撤结束时间"] = "--"
    md_dict["最大回撤"] = -md
    return md_dict


def ret_idx(df_nav, column_name, period):
    """
    计算总收益/年化/夏普/最大回撤/周胜率等。
    period: 'd'(日度) 或 'w'(周度)
    返回: 指标字典
    """
    trade_calendar = _load_trade_cal()
    df = df_nav.copy()
    ret_dict_base = {}
    df = df.sort_index()

    start_date = df.index[0]
    end_date = df.index[-1]
    trade_calendar_term = trade_calendar[start_date:end_date]
    df = pd.merge(trade_calendar_term, df, how="left", left_index=True, right_index=True)

    if period == "d":
        if pd.isnull(df[column_name][0]):
            print(f"******警告:{column_name}起始日无净值,计算无效!******")
        elif df[df[column_name].notna()].index.size / df.index.size < 0.8:
            print(f"******警告:{column_name}日度数据缺失率超过20%!******")
        df[column_name] = df[column_name].ffill()   # 修复:fillna(method='ffill') -> .ffill()

        period_sum = df.index.size - 1
        df["ret_day"] = df[column_name] / df[column_name].shift(1) - 1

        ret_t = df.loc[df.index[-1], column_name] / df.loc[df.index[0], column_name] - 1
        ret_a = pow(ret_t + 1, 250 / period_sum) - 1

        ret_m = {}
        for i in [1, 3, 6, 12]:
            item = df.loc[(df.index[-1] + dateutil.relativedelta.relativedelta(months=-i)):df.index[-1], column_name]
            ret_m[f"近{i}月收益率"] = item[-1] / item[0] - 1

        item = df.loc[str(df.index[-1].year), column_name]
        ret_ytd = item[-1] / item[0] - 1

        ar_ann = np.sqrt(250) * df["ret_day"].std()
        df["f"] = np.where(df["ret_day"] < 0, 1, 0)
        df["dr"] = (df["ret_day"]) ** 2 * df["f"]
        dr_ann = np.sqrt(df["dr"].sum()) * np.sqrt(250 / (period_sum - 1))

        week_win_list = []
        week_ret_list = []
        week_end_list = df[df["is_week_start_end"] == -1].index.to_list()
        for i in range(len(week_end_list)):
            if i == 0:
                week_ret = df.loc[week_end_list[i], column_name] / df[column_name][0] - 1
            else:
                week_ret = df.loc[week_end_list[i], column_name] / df.loc[week_end_list[i - 1], column_name] - 1
            week_ret_list.append(week_ret)
            week_win_list.append(1 if week_ret > 0 else 0)

        week_num = len(week_win_list)
        week_win_num = sum(week_win_list)
        week_loss_num = week_num - week_win_num
        week_win_ratio = sum(week_win_list) / len(week_win_list)
        neg_sum = abs(sum([x for x in week_ret_list if x < 0]))
        profit_rate = (sum([x for x in week_ret_list if x > 0])) / neg_sum if neg_sum != 0 else 9999
        week_mean_ret = np.mean(week_ret_list)
        min_week_ret = min(week_ret_list)
        ar_week_ann = np.sqrt(52) * np.std(week_ret_list, ddof=1)

        dd_m = []
        for i in range(len(week_win_list)):
            if i == 0:
                dd_m.append(1 if week_win_list[0] == 0 else 0)
            else:
                dd_m.append(dd_m[i - 1] + 1 if week_win_list[i] == 0 else 0)
        max_long_week = max(dd_m)

    elif period == "w":
        df = df[df["is_week_start_end"] == -1]
        if df[df[column_name].notna()].index.size / df.index.size < 0.8:
            print("******警告:周度数据缺失率超过20%!******")
        df[column_name] = df[column_name].ffill()   # 修复

        period_sum = df.index.size - 1
        df["ret_week"] = df[column_name] / df[column_name].shift(1) - 1

        ret_t = df.loc[df.index[-1], column_name] / df.loc[df.index[0], column_name] - 1
        ret_a = pow(ret_t + 1, 52 / period_sum) - 1

        ret_m = {}
        for i in [1, 3, 6, 12]:
            item = df.loc[(df.index[-1] + dateutil.relativedelta.relativedelta(months=-i)):df.index[-1], column_name]
            ret_m[f"近{i}月收益率"] = item[-1] / item[0] - 1

        item = df.loc[str(df.index[-1].year), column_name]
        ret_ytd = item[-1] / item[0] - 1

        ar_ann = np.sqrt(52) * df["ret_week"].std()
        df["f"] = np.where(df["ret_week"] < 0, 1, 0)
        df["dr"] = (df["ret_week"]) ** 2 * df["f"]
        dr_ann = np.sqrt(df["dr"].sum()) * np.sqrt(52 / (period_sum - 1))

        week_win_list = []
        week_ret_list = []
        week_end_list = df[df["is_week_start_end"] == -1].index.to_list()
        for i in range(1, len(week_end_list)):
            week_ret = df.loc[week_end_list[i], column_name] / df.loc[week_end_list[i - 1], column_name] - 1
            week_ret_list.append(week_ret)
            week_win_list.append(1 if week_ret > 0 else 0)

        week_num = len(week_win_list)
        week_win_num = sum(week_win_list)
        week_loss_num = week_num - week_win_num
        week_win_ratio = sum(week_win_list) / len(week_win_list)
        neg_mean = abs(np.mean([x for x in week_ret_list if x < 0]))
        profit_rate = (np.mean([x for x in week_ret_list if x > 0])) / neg_mean
        week_mean_ret = np.mean(week_ret_list)
        min_week_ret = min(week_ret_list)
        ar_week_ann = np.sqrt(52) * np.std(week_ret_list, ddof=1)

        dd_m = []
        for i in range(len(week_win_list)):
            if i == 0:
                dd_m.append(1 if week_win_list[0] == 0 else 0)
            else:
                dd_m.append(dd_m[i - 1] + 1 if week_win_list[i] == 0 else 0)
        max_long_week = max(dd_m)
    else:
        raise ValueError("period 必须为 'd' 或 'w'")

    # 指标汇总
    ret_dict_base["累计收益率"] = ret_t
    ret_dict_base["年化收益率"] = ret_a
    ret_dict_base["周均收益率"] = week_mean_ret
    ret_dict_base.update(ret_m)
    ret_dict_base["年初至今收益率"] = ret_ytd

    if period == "d":
        ret_dict_base["年化波动率(日度收益)"] = ar_ann
        ret_dict_base["年化下行波动率(日度收益)"] = dr_ann
    ret_dict_base["年化波动率(周度收益)"] = ar_week_ann
    if period == "w":
        ret_dict_base["年化下行波动率(周度收益)"] = dr_ann

    md = maxdrawdown(df, column_name)
    ret_dict_base.update(md)
    ret_dict_base["最低单周回报率"] = min_week_ret
    ret_dict_base["最长连续下跌周数"] = max_long_week

    ret_dict_base["夏普比率(Sharpe)"] = (ret_a - 0.03) / ar_ann
    ret_dict_base["卡玛比例(Calmar)"] = -ret_a / md["最大回撤"] if md["最大回撤"] != 0 else 9999
    ret_dict_base["索提诺比率(Sortino)"] = (ret_a - 0.03) / dr_ann if dr_ann != 0 else 9999

    ret_dict_base["交易周数"] = week_num
    ret_dict_base["正收益周数"] = week_win_num
    ret_dict_base["负收益周数"] = week_loss_num
    ret_dict_base["周胜率"] = week_win_ratio
    ret_dict_base["盈亏比(周度)"] = profit_rate

    return ret_dict_base
