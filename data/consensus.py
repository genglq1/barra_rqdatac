# -*- coding: utf-8 -*-
"""
一致预期数据下载(分析师预测,alternative-data)
================================================
rqdatac consensus 属于另类数据,与基础财务(get_factor)不同源,故独立模块。

提供两类因子所需的一致预期字段:
    EPIBS(Earnings Yield):comp_con_eps_ftm = 一致预期每股收益(FTM,元/股)
    EGIBS/EGIBS_s(Growth):comp_con_net_profit_growth_ratio_t3 / _ftm = 一致预期
        净利润同比增长率(长期 T+3 / 短期 FTM);rqdatac 无 EPS 增长率字段,
        用净利润增长代理 earnings growth(股本变动时有偏差,数据固有限制)。
    数据频率:日度(每个交易日更新),覆盖分析师跟踪的股票(大中盘为主)

产出(data_store/base/):见下方 CONSENSUS_FIELDS(字段名.csv,日频日期×股票宽表)
"""

import datetime as dt
import pandas as pd
from tqdm import tqdm

from config import get_batch_size
from .client import init_rqdatac
from . import io, universe

# 一致预期字段 -> 产出文件名
# - comp_con_eps_ftm            : EPIBS 因子用(预期EPS_FTM / 价格)
# - comp_con_net_profit_growth_ratio_t3  : Growth 的 EGIBS 用(长期净利润增长预测)
# - comp_con_net_profit_growth_ratio_ftm : Growth 的 EGIBS_s 用(短期净利润增长预测)
CONSENSUS_FIELDS = {
    "comp_con_eps_ftm": "comp_con_eps_ftm.csv",                              # 一致预期 EPS(FTM) → EPIBS
    "comp_con_net_profit_growth_ratio_t3":  "comp_con_net_profit_growth_ratio_t3.csv",   # 预期净利润增长率(长期) → EGIBS
    "comp_con_net_profit_growth_ratio_ftm": "comp_con_net_profit_growth_ratio_ftm.csv",  # 预期净利润增长率(短期) → EGIBS_s
}


def update_consensus_eps(start_date="2010-01-01", end_date=None):
    """
    下载一致预期指标(EPS_FTM + 净利润增长率 t3/ftm),日频,增量更新落盘。
    按 CONSENSUS_FIELDS 逐字段取数,供 EPIBS / EGIBS / EGIBS_s 因子使用。

    注:consensus 数据覆盖分析师跟踪的股票(约 80% 大中盘),小盘股可能缺失,
        缺失值在各因子计算时按截面非空标准化处理。
    """
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    existing = {}
    latest_dates = []
    for field, fname in CONSENSUS_FIELDS.items():
        ld, edf = io.get_latest_date(fname, start_date)
        latest_dates.append(ld)
        existing[fname] = edf

    latest_date = min(latest_dates)
    if latest_date >= end_date:
        print("一致预期数据: 已是最新,无需更新")
        return

    stocks = universe.get_stock_list_rq()
    batch_size = get_batch_size()
    print(f"一致预期更新区间: {latest_date} ~ {end_date},共 {len(stocks)} 只股票")

    # 逐字段分批取数(consensus 接口签名:get_comp_indicators(order_book_ids, start, end, fields, report_range))
    for field, fname in CONSENSUS_FIELDS.items():
        frames = []
        for i in tqdm(range(0, len(stocks), batch_size), desc=f"一致预期-{field}"):
            batch = stocks[i: i + batch_size]
            try:
                df = rqdatac.consensus.get_comp_indicators(
                    batch, start_date=latest_date, end_date=end_date,
                    fields=[field], report_range=0, market="cn",
                )
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"批次 {i} 取 {field} 失败: {e}")
                continue

        if not frames:
            print(f"警告: {field} 无数据,跳过 {fname}")
            continue

        raw = pd.concat(frames)
        # MultiIndex(order_book_id, date) → 日期×股票宽表
        df_new = io.pivot_multiindex(raw, field)
        io.concat_and_save(fname, df_new, existing[fname])
